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
"""
Provide data across multiple data packs during training. A data pack iterator
iterates over each single data example across multiple data packs. A data pack
data set represents the dataset of a bunch of data packs. A raw example
represents a single data point in the dataset. A feature collection represents
an extracted feature corresponding to an input data point.
"""
from typing import Dict, Iterator, Type, Optional, List, Tuple, Union, Any, Callable

from forte.utils.utils import get_class
from forte.common.configurable import Configurable
from forte.datasets.conll.conll_utils import get_tag
from forte.data.data_pack import DataPack
from forte.common.configuration import Config
from forte.data.types import DataRequest
from ft.onto.base_ontology import Token, Sentence, EntityMention
from forte.data.types import DataRequest
from forte.utils import create_import_error_msg
from forte.data.data_pack_weight_sampler import DataPackWeightSampler

try:
    import torch
except ImportError as e:
    raise ImportError(
        create_import_error_msg("torch", "extractor", "data pack dataset")
    ) from e

try:
    from texar.torch.data import IterDataSource, DatasetBase, Batch
except ImportError as e:
    raise ImportError(
        create_import_error_msg(
            "texar-pytorch", "extractor", "data pack dataset"
        )
    ) from e


class MixupIterator(Configurable):
    def __init__(
            self,
            pack_iterator: Iterator[DataPack],
            data_pack_weighting_fn: Callable[[DataPack, Sentence], float],
            segment_annotate_fn: Callable[[DataPack, Sentence, int], int],
            configs: Union[Config, Dict[str, Any]],
    ):
        self.configs = self.make_configs(configs)
        self._data_request = {
            Token: {"fields": ["ner"]},
            }
        self._augment_entry = get_class(self.configs["augment_entry"])
        self._segment_pool_sampler: DataPackWeightSampler = DataPackWeightSampler(
            pack_iterator, data_pack_weighting_fn, self.configs["context_type"], self._data_request, self.configs["skip_k"]
        )
        self._segment_pool = self._segment_pool_sampler.generate_weighted_samples(self.configs['augment_size'])
        self._segment_annotate_fn = segment_annotate_fn
        self._annotated_data_iterator = self._annotate_segment_pool()

    def _annotate_segment_pool(self):
        segment_pool = []
        for sentence, pack, seg_num, tag in self._segment_pool:
            seg_tid = self._segment_annotate_fn(pack, sentence, seg_num)
            annotated_pack = DataPack()
            annotated_pack.set_text(sentence.text)
            segment = pack.get_entry(seg_tid)
            type = getattr(segment, 'ner_type')
            start, end = segment.begin - sentence.begin, segment.end - sentence.begin
            entry = self._augment_entry(annotated_pack, start, end)
            setattr(entry, 'ner_type', type)
            setattr(annotated_pack, 'tag', tag)
            annotated_pack.add_entry(entry)
            segment_pool.append(annotated_pack)
        return iter(segment_pool)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data_a = next(self._annotated_data_iterator)
            data_b = next(self._annotated_data_iterator)
        except StopIteration as e:
            raise StopIteration from e
        return data_a, data_b

    # change argument name (docs )
    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {"augment_entry": "ft.onto.base_ontology.EntityMention",
                "context_type": Sentence,
                "skip_k": 0,
                "augment_size": 10,
                }

    def _tokenize_sentence(self, tokenizer, label_map, datapack, tags, begin, end):
        text_list = datapack.text.split(' ')
        # label_list = example.label_a
        tokens = []
        labels = []
        valid = []
        label_mask = []
        mix_start, mix_end, word_len, token_len = 0, 0, 0, 0
        tokens.append('[CLS]')
        valid.append(1)
        label_mask.append(1)
        labels.append(label_map['[CLS]'])
        for i, word in enumerate(text_list):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            for idx in range(len(token)):
                labels.append(label_map[tags[i]])
                label_mask.append(1)
                if idx == 0:
                    valid.append(1)
                else:
                    valid.append(0)
            # print(word_len, begin, end)
            if word_len == begin:
                mix_start = token_len
            word_len += len(word)+1
            token_len += len(token)
            if word_len == end+1:
                mix_end = token_len
        tokens.append('[SEP]')
        valid.append(1)
        label_mask.append(1)
        labels.append(label_map['[SEP]'])
        return tokens, valid, labels, label_mask, mix_start, mix_end

    def _adjust_segment_length(self, input_ids, mix_end, valid,
                               labels, label_map, adjust_len):
        input_ids[mix_end:mix_end] = [0] * adjust_len
        valid[mix_end:mix_end] = [0] * adjust_len
        labels[mix_end:mix_end] = [label_map['O']] * adjust_len
        return input_ids, valid, labels


    def get_data_batch(self, mixup_iterator, tokenizer, mix_ratio_fn, label_map,
                       batch_size=1):
        instances = []
        tags_a, tags_b = [], []
        begin_a, end_a, begin_b, end_b = 0, 0, 0, 0
        for d_a, d_b in mixup_iterator:
            for augment_entry in d_a.get(self._augment_entry):
                begin_a, end_a = augment_entry.begin, augment_entry.end
                tags_a = d_a.tag
            input_ids_a, valid_a, labels_a, label_mask_a, mix_start_a, mix_end_a = self._tokenize_sentence(
                tokenizer, label_map, d_a, tags_a, begin_a, end_a)

            for augment_entry in d_b.get(self._augment_entry):
                begin_b, end_b = augment_entry.begin, augment_entry.end
                tags_b = d_b.tag
            input_ids_b, valid_b, labels_b, label_mask_b, mix_start_b, mix_end_b = self._tokenize_sentence(
                tokenizer, label_map, d_b, tags_b, begin_b, end_b)


            instances.append((input_ids_a, input_ids_b, mix_ratio_fn(8),
                              (begin_a, end_a, begin_b, end_b)))
            if len(instances) == batch_size:
                yield instances
                instances = []


def mix_embeddings(embedding_a, embedding_b, mix_ratios, mix_idxes):
    batch_size, _, _ = embedding_a.shape

    for b in range(batch_size):
        if mix_ratios[b] > 0:
            mix_ratio = mix_ratios[b]
            for idx in range(mix_idxes[b][0], mix_idxes[b][1]):

                embedding_a[b][idx] = embedding_a[b][idx] * mix_ratio + \
                                      embedding_b[b][idx - mix_idxes[b][0]] * (1 - mix_ratio)
    return embedding_a
