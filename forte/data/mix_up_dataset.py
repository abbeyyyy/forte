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
from typing import Dict, Iterator, Type, Optional, List, Tuple, Union, Any, \
    Callable
import copy

import torch
from torch.utils.data import (DataLoader, TensorDataset, RandomSampler,
                              SequentialSampler)

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


class Instance(object):
    def __init__(self, input_ids, valid, labels):
        self.input_ids = input_ids
        self.valid = valid
        self.labels = labels


class MixInstance(object):
    def __init__(self, input_ids_a, input_ids_b, valid_a, valid_b,
                 mix_ratio, labels, mix_idx):
        self.input_ids_a = input_ids_a
        self.input_ids_b = input_ids_b
        self.valid_a = valid_a
        self.valid_b = valid_b
        self.mix_ratio = mix_ratio
        self.labels = labels
        self.mix_idx = mix_idx


class MixupIterator(Configurable):
    def __init__(
            self,
            pack_iterator: Iterator[DataPack],
            data_pack_weighting_fn: Callable[[DataPack, Sentence], float],
            segment_annotate_fn: Callable[[DataPack, Sentence, int], int],
            configs: Union[Config, Dict[str, Any]],
            train_iterator: Iterator[DataPack],
            eval_iterator: Iterator[DataPack] = None,
            test_iterator: Iterator[DataPack] = None,
    ):
        self.configs = self.make_configs(configs)
        self._data_request = {
            Token: {"fields": ["ner"]},
        }
        self._augment_entry = get_class(self.configs["augment_entry"])
        self._segment_pool_sampler: DataPackWeightSampler = DataPackWeightSampler(
            pack_iterator, data_pack_weighting_fn,
            self.configs["context_type"],
            self._data_request, self.configs["skip_k"]
        )
        self._segment_pool = self._segment_pool_sampler.generate_weighted_samples(
            self.configs['augment_size'])
        self._segment_annotate_fn = segment_annotate_fn
        self._train_iterator = train_iterator
        self._eval_iterator = eval_iterator
        self._test_iterator = test_iterator

        self._iter_mode_mixed = False
        self._train_mode = 'TRAIN'

    def _iter_original(self):
        if self._train_mode == 'TRAIN':
            original_data_iterator = self._train_iterator
        elif self._train_mode == 'EVAL':
            original_data_iterator = self._eval_iterator
        elif self._train_mode == 'TEST':
            original_data_iterator = self._test_iterator
        else:
            raise Exception
        self._curr_data_pack = next(original_data_iterator)
        self._context_iter = self._curr_data_pack.get(
            Sentence
        )
        try:
            sent = next(self._context_iter)
            tag = get_tag(self._curr_data_pack, sent, Token, EntityMention,
                          "ner_type")
        except StopIteration:
            self._curr_data_pack = next(original_data_iterator)
            self._context_iter = self._curr_data_pack.get(
                Sentence
            )
            sent = next(self._context_iter)
            tag = get_tag(self._curr_data_pack, sent, Token, EntityMention,
                          "ner_type")
        d = DataPack()
        d.set_text(sent.text)
        setattr(d, 'tag', tag)
        yield d

    def _iter_mixed(self):
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
            yield annotated_pack

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter_mode_mixed:
            try:
                data_a = self._iter_mixed()
                data_b = self._iter_mixed()
            except StopIteration as e:
                raise StopIteration from e
            return next(data_a, data_b)
        else:
            return next(self._iter_original())

    # change argument name (docs )
    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {"augment_entry": "ft.onto.base_ontology.EntityMention",
                "context_type": Sentence,
                "skip_k": 0,
                "augment_size": 10,
                "max_len": 128,
                }

    @classmethod
    def _tokenize_sentence(cls, tokenizer, label_map, data_pack, tags, begin=-1,
                           end=-1):
        text_list = data_pack.text.split(' ')
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
            word_len += len(word) + 1
            token_len += len(token)
            if word_len == end + 1:
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
        return input_ids, valid, labels, mix_end + adjust_len

    @classmethod
    def _adjust_seq_length(cls, sequence, max_len, padding=0):
        if len(sequence) > max_len:
            sequence = sequence[:max_len]
        else:
            sequence += [padding] * (max_len - len(sequence))
        assert (len(sequence) == max_len)

    def get_original_data_batch(self, tokenizer, label_map, batch_size=1,
                                mode="TRAIN"):
        instances = []
        self._iter_mode_mixed = False
        self._train_mode = mode
        for d in self:
            tokens, valid, labels, label_mask, _, _ = self._tokenize_sentence(
                tokenizer, label_map, d, d.tag)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            max_len = self.configs["max_len"]
            for seq in [input_ids, valid, labels]:
                self._adjust_seq_length(seq, max_len)
            instances.append(Instance(input_ids, valid, labels))
            if len(instances) == batch_size:
                yield self._batch_fn(instances)
                instances = []

    def get_mixed_data_batch(self, tokenizer, mix_ratio_fn, label_map,
                             batch_size=1):
        instances = []
        tags_a, tags_b = [], []
        begin_a, end_a, begin_b, end_b = 0, 0, 0, 0
        self._iter_mode_mixed = True
        for d_a, d_b in self:
            for augment_entry in d_a.get(self._augment_entry):
                begin_a, end_a = augment_entry.begin, augment_entry.end
                tags_a = d_a.tag
            tokens_a, valid_a, labels_a, label_mask_a, mix_start_a, mix_end_a = self._tokenize_sentence(
                tokenizer, label_map, d_a, tags_a, begin_a, end_a)

            for augment_entry in d_b.get(self._augment_entry):
                begin_b, end_b = augment_entry.begin, augment_entry.end
                tags_b = d_b.tag
            tokens_b, valid_b, labels_b, label_mask_b, mix_start_b, mix_end_b = self._tokenize_sentence(
                tokenizer, label_map, d_b, tags_b, begin_b, end_b)

            input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
            input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)

            len_a, len_b = end_a - begin_a, end_b - begin_b
            if len_b > len_a:
                input_ids_a, valid_a, labels_a, end_a = self._adjust_segment_length(
                    input_ids_a, end_a, valid_a, labels_a, label_map,
                    len_b - len_a)
            else:
                input_ids_b, valid_b, labels_b, end_b = self._adjust_segment_length(
                    input_ids_b, end_b, valid_b, labels_b, label_map,
                    len_a - len_b)

            max_len = self.configs["max_len"]
            for seq in [input_ids_a, input_ids_b, valid_a, valid_b, labels_a,
                        labels_b]:
                self._adjust_seq_length(seq, max_len)

            mix_ratio = mix_ratio_fn(8)
            mixed_label = self.mix_sequence(labels_a, labels_b, mix_ratio,
                                            (begin_a, end_a, begin_b, end_b))
            print(len(input_ids_a), len(input_ids_b))
            instances.append(MixInstance(input_ids_a, input_ids_b, valid_a,
                                         valid_b, mix_ratio_fn(8), mixed_label,
                                         (begin_a, end_a, begin_b, end_b)))
            if len(instances) == batch_size:
                yield self._mixed_batch_fn(instances)
                instances = []

    @classmethod
    def _batch_fn(cls, instances):
        input_ids_batch = torch.tensor([ins.input_ids for ins in instances],
                                       dtype=torch.long)
        valid_batch = torch.tensor([ins.valid for ins in instances],
                                   dtype=torch.long)
        label_batch = torch.tensor([ins.labels for ins in instances],
                                   dtype=torch.float)
        return input_ids_batch, valid_batch, label_batch

    @classmethod
    def _mixed_batch_fn(cls, instances):
        input_ids_a_batch = torch.tensor([ins.input_ids_a for ins in instances],
                                         dtype=torch.long)
        input_ids_b_batch = torch.tensor([ins.input_ids_b for ins in instances],
                                         dtype=torch.long)
        valid_a_batch = torch.tensor([ins.valid_a for ins in instances],
                                     dtype=torch.long)
        valid_b_batch = torch.tensor([ins.valid_b for ins in instances],
                                     dtype=torch.long)
        mix_ratio_batch = torch.tensor([ins.mix_ratio for ins in instances],
                                       dtype=torch.float)
        label_batch = torch.tensor([ins.labels for ins in instances],
                                   dtype=torch.float)
        mixed_idxes = torch.tensor([ins.mix_idx for ins in instances],
                                   dtype=torch.long)
        return input_ids_a_batch, input_ids_b_batch, valid_a_batch, valid_b_batch, mix_ratio_batch, label_batch, mixed_idxes

    @classmethod
    def mix_sequence(cls, seq_a, seq_b, mix_ratio, mix_idx):
        begin_a, end_a, begin_b, end_b = mix_idx
        assert (end_a - begin_a == end_b - begin_b)
        mixed_seq = seq_a.copy()
        for idx in range(end_a - begin_a):
            idx_a = idx + begin_a
            idx_b = idx + begin_b
            if idx_a < len(seq_a) and idx_b < len(seq_b):
                mixed_seq[idx_a] = seq_a[idx_a] * mix_ratio + \
                                   seq_b[idx_b] * (1 - mix_ratio)
        return mixed_seq

    @classmethod
    def mix_embeddings(cls, embedding_a, embedding_b, mix_ratios, mix_idxes):
        print(embedding_a.shape)
        batch_size = embedding_a.shape[0]
        mixed_embeddings = embedding_a.clone()
        for bt in range(batch_size):
            if mix_ratios[bt] > 0:
                mix_ratio = mix_ratios[bt]
                assert (mix_idxes[bt][1] - mix_idxes[bt][0] == mix_idxes[bt][
                    3] - mix_idxes[bt][2])
                for idx in range(mix_idxes[bt][1] - mix_idxes[bt][0]):
                    idx_a = idx + mix_idxes[bt][0]
                    idx_b = idx + mix_idxes[bt][2]
                    if idx_a < len(embedding_a[bt]) and idx_b < len(
                            embedding_b[bt]):
                        mixed_embeddings[bt][idx_a] = \
                            embedding_a[bt][idx_a] * mix_ratio + \
                            embedding_b[bt][idx_b] * (1 - mix_ratio)
        return embedding_a
