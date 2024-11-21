from abc import ABC
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
from transformers import BertTokenizerFast, BertModel, BertConfig, BertForMaskedLM
from dataclasses import dataclass
from transformers import AutoModel, AutoConfig, BertForPreTraining, BertConfig
from triplet import EntityDict

from dict_hub import get_entity_dict, get_tokenizer
from triplet_mask import construct_mask

def build_model(args) -> nn.Module:
    return CustomBertModel(args)


@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor

class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]
        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.hr_bert.resize_token_embeddings(len(get_tokenizer()))
        self.tail_bert = deepcopy(self.hr_bert)

    def encode_neighbors(self, data):
        # 1. prepare data
        # 2. encode neighbors
        embeds_from_neighbors = []
        for i in range(len(data)):
            input_ids = data[i]['input_ids']  # (batch_size, seq_len)
            token_type_ids = data[i]['token_type_ids'] # (batch_size, seq_len)
            attention_mask = data[i]['attention_mask'] # (batch_size, seq_len)
            mask_pos = data[i]['mask_pos']  # (batch_size, 2)

            output =  self._encode(self.hr_bert,
                                   token_ids=input_ids,
                                   mask=attention_mask,time=None,type=1,
                                   token_type_ids=token_type_ids)
            embeds = torch.mean(output, dim=0)  # (batch_size, 768)
            embeds_from_neighbors.append(embeds)
        return  embeds_from_neighbors

    def _encode_embeds(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(inputs_embeds=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        return cls_output


    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        return cls_output

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,time,hr_neighbors,
                         
                only_ent_embedding=False, **kwargs) -> dict:

        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)
        hr_neighbor_embeds = self.encode_neighbors(hr_neighbors)
        if len(hr_neighbor_embeds) > 1:
            hr_neighbor_embeds = torch.stack(hr_neighbor_embeds, dim=0)
        inputs_embeds = self.hr_bert.embeddings.word_embeddings(hr_token_ids)
        unused_token_id = get_tokenizer().convert_tokens_to_ids("[neighbor_token]")
        pos = torch.eq(hr_token_ids, unused_token_id)
        head_pos = torch.nonzero(pos)
        inputs_embeds[head_pos[:, 0], head_pos[:, 1], :] = hr_neighbor_embeds

        hr_vector = self._encode_embeds(self.hr_bert,
                                 token_ids=inputs_embeds,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)
        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)

        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector
                }

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)
        hr_vector=hr_vector.to(tail_vector.dtype)
        logits = hr_vector.mm(tail_vector.t())
        
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()
        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids,**kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   time=None,type=1,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors':ent_vectors}


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
