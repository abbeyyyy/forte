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
from tqdm import trange
import os
import numpy as np
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig, BertModel, \
    BertForTokenClassification
from seqeval.metrics import classification_report

from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.data.mix_up_dataset import MixUpIterator, MixupDataProcessor
from forte.data.readers.conll03_reader import CoNLL03Reader
from ft.onto.base_ontology import Sentence, EntityMention, Token

logger = logging.getLogger(__name__)
BERT_MODEL = "bert-base-cased"
TASK_NAME = "ner"


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
        if input_ids_b is not None:
            embedding_output_b, extended_attention_mask_b, head_mask_b = \
                self._get_embedding(input_ids_b,
                                    token_type_ids=token_type_ids_b,
                                    attention_mask=attention_mask_b,
                                    head_mask=head_mask_b,
                                    position_ids=position_ids_b,
                                    )
            # mix embeddings
            mixed_embedding = MixupDataProcessor.mix_embeddings(
                embedding_output_a,
                embedding_output_b,
                mix_ratios,
                mix_idxes)
        else:
            mixed_embedding = embedding_output_a
        mixed_extended_attention_mask = extended_attention_mask_a
        mixed_head_mask = head_mask_a

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


def one_hot_label(label, label_len):
    ohe = [0] * label_len
    ohe[label] = 1.0
    return np.array(ohe)


class MixupTrainer():
    def __init__(self, config):
        self._update_config(config)
        self.get_data_iterator()
        self.get_model()
        self.data_processor = MixupDataProcessor()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and self.config[
                "device"] == 'cuda'
            else 'cpu')
        print('DEVICE: ', self.device)

    def _update_config(self, config):
        self.config = self._default_config()
        for k, v in config.items():
            if k in self.config and v != self.config[k]:
                self.config[k] = v

    @classmethod
    def _default_config(cls):
        return {
            "train_batch_size": 32,
            "eval_batch_size": 16,
            "learning_rate": 5e-05,
            "max_len": 128,
            "num_pretrain_epochs": 5,
            "num_train_epochs": 20,
            "gradient_accumulation_steps": 1,
            "weight_decay": 0.01,
            "warmup_proportion": 1,
            "adam_epsilon": 1e-08,
            "max_grad_norm": 1e-08,
            "device": 'cuda',
            "num_initial": 200,
        }

    def get_data_iterator(self):
        root_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir,
                os.pardir,
            )
        )

        train_file_path: str = os.path.join(
            root_path, "data_samples/conll03/train"
        )
        eval_file_path: str = os.path.join(
            root_path, "data_samples/conll03/eval"
        )
        test_file_path: str = os.path.join(
            root_path, "data_samples/conll03/test"
        )
        reader = CoNLL03Reader()
        context_type = Sentence
        request = {
            Token: {"fields": ["ner"]},
        }
        skip_k = 0

        self.feature_schemes = {}

        num_initial = self.config["num_initial"]
        self.mixup_iterator = MixUpIterator(
            self.get_iterator(reader, train_file_path),
            data_pack_entity_weighting,
            data_pack_random_entity,
            {"context_type": context_type,
             # "request": request,
             "skip_k": skip_k},
            train_iterator=lambda: self.get_iterator(reader, train_file_path),
            eval_iterator=lambda: self.get_iterator(reader, eval_file_path),
            test_iterator=lambda: self.get_iterator(reader, test_file_path),
            num_initial=num_initial,
        )
        self.train_iterator = self.get_iterator(reader, train_file_path)
        self.eval_iterator = self.get_iterator(reader, eval_file_path)

    @classmethod
    def get_iterator(cls, reader, file_path):
        pl: Pipeline = Pipeline()
        pl.set_reader(reader)
        pl.initialize()
        return pl.process_dataset(file_path)

    def get_model(self):
        labels = ["[NULL]", "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG",
                  "I-ORG",
                  "B-LOC", "I-LOC", "[CLS]", "[SEP]"]
        self.label_map = {l: one_hot_label(i, len(labels)) for i, l in
                          enumerate(labels)}
        self.label_id_map = {i:l for i, l in
                          enumerate(labels)}
        bert_config = BertConfig.from_pretrained(BERT_MODEL,
                                                 num_labels=len(labels),
                                                 finetuning_task=TASK_NAME)
        self.model = BertForNer.from_pretrained(BERT_MODEL, config=bert_config)
        self.model.bert = BertModelForNer.from_pretrained(
            BERT_MODEL, config=bert_config
        )
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if
                        not any(nd in n for nd in no_decay)],
             'weight_decay': self.config["weight_decay"]},
            {'params': [p for n, p in param_optimizer if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        num_train_optimization_steps = int(
            200 / self.config["train_batch_size"] / self.config[
                "gradient_accumulation_steps"]) * (
                                           (self.config["num_train_epochs"] +
                                            self.config["num_pretrain_epochs"]))
        warmup_steps = int(
            self.config["warmup_proportion"] * num_train_optimization_steps)
        # self.optimizer = AdamW(optimizer_grouped_parameters,
        #                        lr=self.config["learning_rate"],
        #                        eps=self.config["adam_epsilon"])
        # self.scheduler = WarmupLinearSchedule(self.optimizer,
        #                                       warmup_steps=warmup_steps,
        #                                       t_total=num_train_optimization_steps)

    def _update_loss(self, loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       self.config["max_grad_norm"])
        # cur_loss += loss
        self.optimizer.step()
        self.scheduler.step()  # Update learning rate schedule
        self.model.zero_grad()

    def train(self):
        self.model.to(self.device)
        self.model.bert.to(self.device)
        pretrain_epochs = self.config["num_pretrain_epochs"]
        train_epochs = self.config["num_train_epochs"]
        for epoch in range(pretrain_epochs + train_epochs):
            for _ in trange(5, desc='Inner Epoch'):
                self.epoch_train(pretrain=epoch < pretrain_epochs)
            report = self.evaluate()
            print(report)

    def epoch_train(self, pretrain=False):
        i = 0
        for batch_data in self.data_processor.get_original_data_batch(
                self.mixup_iterator,
                self.tokenizer, self.label_map,
                self.config["train_batch_size"],
                self.config["max_len"],
        ):
            batch_data = tuple(d.to(self.device) for d in batch_data)
            input_ids, input_mask, valid, labels = batch_data
            i += 1
            print("b", i, len(input_ids))
            # loss, _ = self.model(input_ids_a=input_ids,
            #                      attention_mask_a=input_mask,
            #                      position_ids_a=valid,
            #                      labels=labels)
            # self._update_loss(loss)

        if not pretrain:
            for batch_data in self.data_processor.get_mixed_data_batch(
                    self.mixup_iterator,
                    self.tokenizer, beta_mix_ratio, self.label_map,
                    self.config["train_batch_size"],
                    self.config["max_len"],
            ):
                # print("b", batch_data)
                batch_data = tuple(d.to(self.device) for d in batch_data)
                input_ids_a, input_ids_b, input_mask_a, input_mask_b, valid_a, valid_b, mix_ratio, labels, mixed_idxes = batch_data
                loss, _ = self.model(input_ids_a=input_ids_a,
                                     attention_mask_a=input_mask_a,
                                     position_ids_a=valid_a,
                                     input_ids_b=input_ids_b,
                                     attention_mask_b=input_mask_b,
                                     position_ids_b=valid_b,
                                     mix_idxes=mixed_idxes,
                                     mix_ratios=mix_ratio,
                                     labels=labels)
                self._update_loss(loss)

    def evaluate(self):
        y_true = []
        y_pred = []
        for batch_data in self.data_processor.get_original_data_batch(
                self.mixup_iterator,
                self.tokenizer, self.label_map,
                self.config["train_batch_size"], mode='EVAL'):
            batch_data = tuple(d.to(self.device) for d in batch_data)
            input_ids, input_mask, valid, labels = batch_data
            with torch.no_grad():
                logits = self.model(input_ids_a=input_ids,
                                    attention_mask_a=input_mask,
                                    position_ids_a=valid)[0]
            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            logits = logits.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()

            for i, label in enumerate(labels):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif labels[i][j][-1] == 1.:
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_1.append(
                            self.label_id_map[self._get_label_ids(labels[i][j])])
                        try:
                            temp_2.append(self.label_id_map[logits[i][j]])
                        except:
                            temp_2.append('UKN')
        return classification_report(y_true, y_pred)

    @classmethod
    def _get_label_ids(cls, label):
        for i, l in enumerate(label):
            if l == 1:
                return i
        return -1


# Abstract class with weighting / random entity
# DEFAULT WEIGHTING SCHEMES
# Assign weights based on number of entities contained in a sentence
def data_pack_entity_weighting(pack: DataPack, sentence: Sentence) -> float:
    entity_num = 0.
    for _ in pack.get(EntityMention, sentence):
        entity_num += 1
    return entity_num


# return a random entity id in a datapack
def data_pack_random_entity(pack: DataPack, sentence: Sentence,
                            num_entity: int) -> int:
    rand_idx = random.randint(0, num_entity - 1)
    for idx, entity in enumerate(pack.get(EntityMention, sentence)):
        if rand_idx == idx:
            return entity.tid


def beta_mix_ratio(alpha):
    return np.random.beta(alpha, alpha)


def train():
    trainer = MixupTrainer({

    })

    trainer.train()


if __name__ == '__main__':
    train()
