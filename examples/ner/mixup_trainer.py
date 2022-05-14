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

import logging
import torch
import random
from torch import nn
import os
from typing import Dict, List, Optional, Tuple, Iterator
from collections import Counter
import numpy as np
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig, BertModel, \
    BertForTokenClassification

from forte.data.base_pack import PackType
from forte.data.data_pack import DataPack
from forte.data.data_pack_weight_sampler import DataPackWeightSampler
from forte.pipeline import Pipeline
from forte.data.mix_up_dataset import MixupIterator, mix_embeddings
from forte.data.readers.conll03_reader import CoNLL03Reader
from ft.onto.base_ontology import Sentence, EntityMention, Token

logger = logging.getLogger(__name__)
BERT_MODEL = "bert-base-cased"


class BertModelForNer(BertModel):
    def forward(self, input_ids_a=None, token_type_ids_a=None,
                attention_mask_a=None, position_ids_a=None, head_mask_a=None,
                input_ids_b=None, token_type_ids_b=None,
                attention_mask_b=None, position_ids_b=None, head_mask_b=None,
                mix_idxes=None, mix_ratios=None):

        embedding_output_a, extended_attention_mask_a, head_mask_a = \
            self._get_embedding(input_ids_a,
                                token_type_ids=token_type_ids_a,
                                attention_mask=attention_mask_a,
                                head_mask=head_mask_a,
                                position_ids=position_ids_a,
                                )
        embedding_output_b, extended_attention_mask_b, head_mask_b = \
            self._get_embedding(input_ids_b,
                                token_type_ids=token_type_ids_b,
                                attention_mask=attention_mask_b,
                                head_mask=head_mask_b,
                                position_ids=position_ids_b,
                                )
        # mix embeddings
        mixed_embedding = mix_embeddings(embedding_output_a, embedding_output_b,
                                         mix_ratios, mix_idxes)
        mixed_extended_attention_mask = mix_embeddings(
            extended_attention_mask_a,
            extended_attention_mask_b,
            mix_ratios, mix_idxes)
        mixed_head_mask = mix_embeddings(head_mask_a, head_mask_b,
                                         mix_ratios, mix_idxes)

        encoder_outputs = self.encoder(mixed_embedding,
                                       mixed_extended_attention_mask,
                                       head_mask=mixed_head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

    def _get_embedding(self, input_ids, token_type_ids=None,
                       attention_mask=None, head_mask=None,
                       position_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(
                    -1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1,
                                             -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(
                    self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        return self.embeddings(input_ids,
                               position_ids=position_ids,
                               token_type_ids=token_type_ids), \
               extended_attention_mask, head_mask


class BertForNer(BertForTokenClassification):
    def forward(self, input_ids_a=None, token_type_ids_a=None,
                attention_mask_a=None, position_ids_a=None, head_mask_a=None,
                input_ids_b=None, token_type_ids_b=None,
                attention_mask_b=None, position_ids_b=None, head_mask_b=None,
                mix_idxes=None, mix_ratios=None, labels=None):
        outputs = self.bert(input_ids_a=input_ids_a, token_type_ids_a=None,
                            attention_mask_a=attention_mask_a,
                            position_ids_a=position_ids_a,
                            head_mask_a=head_mask_a,
                            input_ids_b=input_ids_b,
                            token_type_ids_b=token_type_ids_b,
                            attention_mask_b=attention_mask_b,
                            position_ids_b=position_ids_b,
                            head_mask_b=head_mask_b,
                            mix_idxes=mix_idxes, mix_ratios=mix_ratios)

        sequence_output = outputs[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim,
                                   dtype=torch.float32, device='cuda')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if position_ids_a[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[
                              2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = softXEnt
            # Only keep active parts of the loss
            if head_mask_a is not None:
                active_loss = head_mask_a.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1, self.num_labels)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels),
                                labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


def softXEnt(input, target):
    logprobs = F.log_softmax(input, dim=1)
    return -(target * logprobs).sum() / input.shape[0]


# USE BERT FOR EXAMPLES
# Match with huggingface NER tutorial
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )

    def forward(self, input_ids_a=None, input_ids_b=None,
                mix_idxes=None, mix_ratios=None):
        embedding_output_a = self.embeddings(input_ids_a)
        embedding_output_b = self.embeddings(input_ids_b)
        return mix_embeddings(embedding_output_a, embedding_output_b,
                              mix_ratios, mix_idxes)


class Model(nn.Module):
    def __init__(
            self,
            config,
    ):
        super().__init__()
        self.embeddings = Embeddings(config)

    def forward(self, input_ids_a=None, input_ids_b=None, labels=None,
                masks_a=None, masks_b=None, token_type_ids_a=None,
                mix_idxes=None, mix_ratios=None):
        embeddings = self.embeddings(input_ids_a, input_ids_b,
                                     mix_idxes, mix_ratios)
        print(embeddings.shape)
        # add loss


class NerTokenizer():
    def __init__(self, word_alphabet):
        self.word_alphabet = word_alphabet

    def tokenize(self, text):
        word_ids = []
        for word in text.split(' '):
            word_ids.append(self.word_alphabet.get_index(word))
        return word_ids


class MixupTrainer():
    def __init__(self):
        # self.model = Model()
        self.get_data_iterator()

    def get_data_iterator(self):
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
        request = {
            Token: {"fields": ["ner"]},
            }
        skip_k = 0

        self.feature_schemes = {}

        train_pl: Pipeline = Pipeline()
        train_pl.set_reader(reader)
        train_pl.initialize()
        self.pack_iterator: Iterator[PackType] = train_pl.process_dataset(
            file_path)

        self.data_sampler: DataPackWeightSampler = DataPackWeightSampler(
            self.pack_iterator, data_pack_entity_weighting, context_type,
            request, skip_k
        )

        self.mixup_iterator = MixupIterator(
            self.pack_iterator,
            data_pack_entity_weighting,
            data_pack_random_entity,
            {"context_type": context_type,
             # "request": request,
             "skip_k": skip_k},
        )

    def train(self):
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        labels = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]
        label_map = {l: i for i, l in enumerate(labels)}
        for batch_data in self.mixup_iterator.get_data_batch(self.mixup_iterator, tokenizer,
                                        beta_mix_ratio, label_map, 2):
            print(batch_data)
        #     words_a, words_b, masks_a, masks_b, mix_ratios, mix_idxes = batch_data
            # loss = model(...)


# Abstract class with weighting / random entity
# DEFAULT WEIGHTING SCHEMES
# Assign weights based on number of entities contained in a sentence
def data_pack_entity_weighting(pack:DataPack, sentence: Sentence) -> float:
    entity_num = 0.
    for _ in pack.get(EntityMention, sentence):
        entity_num += 1
    return entity_num


# return a random entity id in a datapack
def data_pack_random_entity(pack: DataPack, sentence: Sentence, num_entity: int) -> int:
    rand_idx = random.randint(0, num_entity - 1)
    for idx, entity in enumerate(pack.get(EntityMention, sentence)):
        if rand_idx == idx:
            return entity.tid


def beta_mix_ratio(alpha):
    return np.random.beta(alpha, alpha)


def train():
    trainer = MixupTrainer()
    trainer.train()


if __name__ == '__main__':
    train()
