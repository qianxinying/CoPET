import os
import json
import torch
import torch.utils.data.dataset
from datetime import datetime
from typing import Optional, List

from config import args
from triplet import reverse_triplet
from triplet_mask import construct_mask, construct_self_negative_mask, construct_negative_mask
from dict_hub import get_entity_dict, get_link_graph, get_tokenizer
from logger_config import logger

entity_dict = get_entity_dict()
if args.use_link_graph:
    # make the lazy data loading happen
    get_link_graph()



def _custom_tokenize(text: str,
                     text_pair: Optional[str] = None) -> dict:
    tokenizer = get_tokenizer()
    encoded_inputs = tokenizer(text=text,
                               text_pair=text_pair if text_pair else None,
                               add_special_tokens=True,
                               max_length=args.max_num_tokens,
                               return_token_type_ids=True,
                               truncation=True)
    return encoded_inputs


def _parse_entity_name(entity: str) -> str:
    if args.task.lower() == 'wn18rr':
        # family_alcidae_NN_1
        entity = ' '.join(entity.split('_')[:-2])
        return entity
    # a very small fraction of entities in wiki5m do not have name
    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity_desc is None:
        return entity
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity

def _concat_name_relation(entity: str,relation) -> str:
    return entity+" "+relation


def get_hr_neighbor_desc(head:str,head_desc:str,head_id: str,time:str,relation:str, relation_id:int,tail_id: str = None) -> str:
    neighbors = get_link_graph().get_relation_neighbor_ids(entity_id=head_id,relation_id=relation_id,time=time)

    neighbor_id = []
    neighbor_relation = []
    neighbor_relation_id = []
    neighbor_time = []
    for neighbor in neighbors:
        if neighbor[0] is tail_id and time is neighbor[3]:
            continue
        neighbor_id.append(neighbor[0])
        neighbor_relation.append(neighbor[1])
        neighbor_relation_id.append(neighbor[2])
        neighbor_time.append(neighbor[3])
    # avoid label leakage during training
    num = len(neighbor_relation_id)
    i = 0
    if num>3:
        neighbor_id = neighbor_id[:2]
        neighbor_relation = neighbor_relation[:2]
        neighbor_relation_id = neighbor_relation_id[:2]
        neighbor_time = neighbor_time[:2]
    result = []
    entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_id]
    entities = [_parse_entity_name(entity) for entity in entities]
    head_text = _concat_name_desc(head,head_desc)
    if num == 0:
        sep1, sep2,sep3 = f'[R_{relation_id}_SEP1]', f'[R_{relation_id}_SEP2]', f'[R_{relation_id}_SEP3]'
        time_sep = f'[Time_{time}_SEP]'
        template = f'{sep1} {head_text} {sep2}{relation}{sep3}{" in "}{time_sep}{time}'
        result.append(template)
        return result
    for e in entities:
        sep1, sep2,sep3,sep4= f'[R_{neighbor_relation_id[i]}_SEP1]', f'[R_{neighbor_relation_id[i]}_SEP2]',f'[R_{neighbor_relation_id[i]}_SEP3]',f'[R_{neighbor_relation_id[i]}_SEP4]'
        desc = entity_dict.get_entity_by_id(neighbor_id[i]).entity_desc
        entity_text = _concat_name_desc(e,desc)
        time_sep = f'[Time_{neighbor_time[i]}_SEP]'
        template =f'{sep1} {neighbor_relation[i]}{sep2}{e}{sep3}{" in "}{time_sep}{neighbor_time[i]} '
        # template =f'{sep1} {head_text}{sep2}{neighbor_relation[i]}{sep3}{entity_text}{sep4}{" in "}{time_sep}{neighbor_time[i]} '
    
        result.append(template)
        i = i+1
    return result



class Example:

    def __init__(self, head_id, relation,relation_id, tail_id,time,**kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation_id = relation_id
        self.relation = relation
        self.time=time

    @property
    def head_desc(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity

    def vectorize(self) -> dict:
        head_desc, tail_desc = self.head_desc, self.tail_desc
        if head_desc is None:
            head_desc = ""
        if tail_desc is None:
            tail_desc = ""      
 
        hr_neighbor =  get_hr_neighbor_desc(head = self.head,head_desc = head_desc,head_id=self.head_id, tail_id=self.tail_id,time = self.time,relation = self.relation,relation_id = self.relation_id)

        assert len(hr_neighbor)<4
        head_word = _parse_entity_name(self.head)
        head_text = _concat_name_desc(head_word, head_desc)
        time_sep = f'[Time_{self.time}_SEP1]'
        rel_sep = f'[R_{self.relation_id}_SEP1]'
        rel_sep2 = f'[R_{self.relation_id}_SEP2]'
        rel_sep3 = f'[R_{self.relation_id}_SEP3]'
        rel_sep4 = f'[R_{self.relation_id}_SEP4]'
        neighbor_token = "[neighbor_token]"
        head = f'{rel_sep}{head_text}{rel_sep2}{neighbor_token}{rel_sep3}{self.relation}{rel_sep4}{" in "}{time_sep}{self.time}'

        hr_encoded_inputs = _custom_tokenize(text=head)
        if len(hr_neighbor)>0:
            hr_neighbor = _custom_tokenize(hr_neighbor)

        head_encoded_inputs = _custom_tokenize(text=head_text)


        tail_word = _parse_entity_name(self.tail)
        tail_encoded_inputs = _custom_tokenize(text=_concat_name_desc(tail_word, tail_desc))
        return {'hr_token_ids': hr_encoded_inputs['input_ids'],
                'hr_token_type_ids': hr_encoded_inputs['token_type_ids'],
                'hr_neighbor':hr_neighbor,
                'tail_token_ids': tail_encoded_inputs['input_ids'],
                'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
                'head_token_ids': head_encoded_inputs['input_ids'],
                'head_token_type_ids': head_encoded_inputs['token_type_ids'],
                'negative_token_ids': None,
                'negative_token_type_ids': None,
                'obj': self}


class Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, path, task, examples=None):
        self.path_list = path.split(',')
        self.task = task
        assert all(os.path.exists(path) for path in self.path_list) or examples
        if examples:
            self.examples = examples
        else:
            self.examples = []
            for path in self.path_list:
                if not self.examples:
                    self.examples = load_data(path)
                else:
                    self.examples.extend(load_data(path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        result=self.examples[index].vectorize()
        result['examples']=self.examples[index]
        return result


def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[Example]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet
    logger.info('In test mode: {}'.format(args.is_test))
    print(path)
    data = json.load(open(path, 'r', encoding='ISO-8859-1'))
    logger.info('Load {} examples from {}'.format(len(data), path))

    cnt = len(data)
    examples = []
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(obj)))
        data[i] = None

    return examples


def collate(batch_data: List[dict]) -> dict:
    hr_token_ids, hr_mask = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    hr_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_type_ids']) for ex in batch_data],
        need_mask=False)

    tail_token_ids, tail_mask = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    tail_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_type_ids']) for ex in batch_data],
        need_mask=False)

    head_token_ids, head_mask = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    head_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_type_ids']) for ex in batch_data],
        need_mask=False)

    hr_text_neighbors = []
    hr_neighbor = [ex['hr_neighbor'] for ex in batch_data]
    if len(hr_neighbor) != 0 :
        for h in hr_neighbor:
            input_ids, attention_mask = to_indices_and_mask(
                [torch.LongTensor(h['input_ids'][i]) for i in range(len(h['input_ids']))],
                pad_token_id=get_tokenizer().pad_token_id)
            token_type_ids = to_indices_and_mask(
                [torch.LongTensor(h['token_type_ids'][i]) for i in range(len(h['input_ids']))],
                need_mask=False)
            mask_pos = torch.nonzero(torch.eq(input_ids, get_tokenizer().mask_token_id))
            result = {}
            result['input_ids'] = input_ids
            result['token_type_ids'] = token_type_ids
            result['attention_mask'] = attention_mask
            result['mask_pos'] = mask_pos
            hr_text_neighbors.append(result)

    examples = [ex['examples'] for ex in batch_data]
    time=[]
    neighbor_label = []
    for e in examples:
        neighbor_label.append(e.__dict__["head_id"])
        time_interval=e.__dict__["time"]
        if time_interval:
            time.append(int(time_interval))
    time=torch.tensor(time,dtype=torch.float)

    batch_exs = [ex['obj'] for ex in batch_data]
    batch_dict = {
        'neighbor_label':neighbor_label,
        'hr_neighbors':hr_text_neighbors,
        'hr_token_ids': hr_token_ids,
        'hr_mask': hr_mask,
        'hr_token_type_ids': hr_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids,
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'time': time,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
    }
    return batch_dict

def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(1)
    if need_mask:
        return indices, mask
    else:
        return indices
