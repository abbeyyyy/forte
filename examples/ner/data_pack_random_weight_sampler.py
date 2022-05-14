#  Copyright 2020 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import unittest
from typing import List, Iterator
import random
from collections import Counter
import numpy as np
import numpy.random
import torch
from typing import Dict, List, Optional, Tuple

from forte.data.base_pack import PackType
from forte.data.data_pack import DataPack
from forte.data.readers.conll03_reader import CoNLL03Reader
from forte.data.data_pack_weight_sampler import DataPackWeightSampler
from forte.data.mix_up_dataset import MixupIterator
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Sentence, EntityMention, Token
from forte.models.ner.utils import normalize_digit_word
from forte.processors.misc import Alphabet



def data_pack_entity_weighting(pack: DataPack, tid: int) -> float:
    entity_num = 0.
    sentence = pack.get_entry(tid)
    for _ in pack.get(EntityMention, sentence):
        entity_num += 1
    return entity_num


def data_pack_random_entity(pack: DataPack, tid:int, num_entity: int) -> int:
    rand_idx = random.randint(0, num_entity-1)
    sentence = pack.get_entry(tid)
    for idx, entity in enumerate(pack.get(EntityMention, sentence)):
        if rand_idx == idx:
            return entity.tid

def normalize_func(x):
    return normalize_digit_word(x)

def beta_mix_ratio(alpha):
    return numpy.random.beta(alpha, alpha)


class DataPackWeightSamplerTest(unittest.TestCase):
    def setUp(self):
        root_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir,
                os.pardir,
            )
        )

        file_path: str = os.path.join(
            root_path, "data_samples/conll03"
        )
        reader = CoNLL03Reader()
        context_type = Sentence
        request = {Sentence: []}
        skip_k = 0

        self.input_files = ["conll03.conll"]
        self.feature_schemes = {}

        train_pl: Pipeline = Pipeline()
        train_pl.set_reader(reader)
        train_pl.initialize()
        self.pack_iterator: Iterator[PackType] = train_pl.process_dataset(file_path)

        self.data_sampler: DataPackWeightSampler = DataPackWeightSampler(
            self.pack_iterator, data_pack_entity_weighting, context_type, request, skip_k
        )

        self.mixup_iterator = MixupIterator(
            self.pack_iterator,
            data_pack_entity_weighting,
            data_pack_random_entity,
            {"context_type": context_type,
             "request": request,
             "skip_k": skip_k},
        )

        self.min_frequency = 0
        self.normalize_digit: bool = True
        self.embedding_path: str = "."

        self.word_cnt: Counter = Counter()
        self.char_cnt: Counter = Counter()
        self.pos_cnt: Counter = Counter()
        self.chunk_cnt: Counter = Counter()
        self.ner_cnt: Counter = Counter()
        self._process()

    def _process(self):
        """
        Process the data pack to collect vocabulary information.

        Args:
            data_pack: The ner data to create vocabulary with.

        Returns:

        """
        # for data_pack in input_pack:
        for data_pack in self.pack_iterator:
            for instance in data_pack.get_data(
                context_type=Sentence, request={Token: ["chunk", "pos", "ner"]}
            ):
                for token in instance["Token"]["text"]:
                    for char in token:
                        self.char_cnt[char] += 1
                    word = normalize_func(token)
                    self.word_cnt[word] += 1

                for pos in instance["Token"]["pos"]:
                    self.pos_cnt[pos] += 1
                for chunk in instance["Token"]["chunk"]:
                    self.chunk_cnt[chunk] += 1
                for ner in instance["Token"]["ner"]:
                    self.ner_cnt[ner] += 1
        self.word_alphabet = Alphabet("word", self.word_cnt)
        self.char_alphabet = Alphabet("character", self.char_cnt)
        self.ner_alphabet = Alphabet("ner", self.ner_cnt)


    def test_data_pack_iterator(self):
        tokenizer = NerTokenizer(self.word_alphabet)
        instances = []
        for d_a, d_b in self.mixup_iterator:
            print(d_a.text)
            input_ids_a, input_ids_b = tokenizer.tokenize(d_a.text), tokenizer.tokenize(d_b.text)
            for augment_entry in d_a.get(EntityMention):
                begin_a, end_a = augment_entry.begin, augment_entry.end
            for augment_entry in d_b.get(EntityMention):
                begin_b, end_b = augment_entry.begin, augment_entry.end
            instances.append((input_ids_a, input_ids_b, beta_mix_ratio(8), (begin_a, end_a, begin_b, end_b)))
        batch_data = self.get_batch_tensor(instances, torch.device('cpu'))
        print(batch_data[0])

    def get_batch_tensor(
        self,
        data: List[Tuple[List[int], List[List[int]], int, Tuple[int, int, int, int]]],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(data)
        batch_length = max([len(d[0]) for d in data])


        wid_inputs_a = np.empty([batch_size, batch_length], dtype=np.int64)
        wid_inputs_b = np.empty([batch_size, batch_length], dtype=np.int64)
        masks_a = np.zeros([batch_size, batch_length], dtype=np.float32)
        masks_b = np.zeros([batch_size, batch_length], dtype=np.float32)
        mix_ratios = np.zeros([batch_size], dtype=np.float32)
        mix_idxes = np.zeros([batch_size, 4], dtype=np.int64)

        for i, inst in enumerate(data):
            wids_a, wids_b, ratio, idxes = inst

            inst_size = len(wids_a)
            wid_inputs_a[i, :inst_size] = wids_a
            wid_inputs_a[i, inst_size:] = self.word_alphabet.pad_id
            masks_a[i, :inst_size] = 1.0
            inst_size = len(wids_b)
            wid_inputs_b[i, :inst_size] = wids_b
            wid_inputs_b[i, inst_size:] = self.word_alphabet.pad_id
            masks_b[i, :inst_size] = 1.0
            mix_ratios[i] = ratio
            mix_idxes[i] = idxes

        words_a = torch.from_numpy(wid_inputs_a).to(device)
        words_b = torch.from_numpy(wid_inputs_b).to(device)
        masks_a = torch.from_numpy(masks_a).to(device)
        masks_b = torch.from_numpy(masks_b).to(device)
        mix_ratios = torch.from_numpy(mix_ratios).to(device)
        mix_idxes = torch.from_numpy(mix_idxes).to(device)

        return words_a, words_b, masks_a, masks_b, mix_ratios, mix_idxes



class NerTokenizer():
    def __init__(self, word_alphabet):
        self.word_alphabet = word_alphabet

    def tokenize(self, text):
        word_ids = []
        for word in text.split(' '):
            word = normalize_func(word)
            word_ids.append(self.word_alphabet.get_index(word))
        return word_ids


if __name__ == "__main__":
    unittest.main()
