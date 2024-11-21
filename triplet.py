import os
import json

from typing import List
from dataclasses import dataclass
from collections import deque
# from logger_config import logger

@dataclass
class EntityExample:
    entity_id: str
    entity: str
    entity_desc: str = ''



class TripletDict:

    def __init__(self, path_list: List[str],args):
        self.path_list = path_list
        # logger.info('Triplets path: {}'.format(self.path_list))
        self.relations = set()
        self.hrt2tails = {}
        self.triplet_cnt = 0
        for path in self.path_list:
            self._load(path)
        # logger.info('Triplet statistics: {} relations, {} triplets'.format(len(self.relations), self.triplet_cnt))

    def _load(self, path: str):
        examples = json.load(open(path, 'r',encoding='latin1'))
        examples += [reverse_triplet(obj) for obj in examples]
        for ex in examples:
            self.relations.add(ex['relation'])
            key = (ex['head_id'], ex['relation'],ex['time'])

            if key not in self.hrt2tails:
                self.hrt2tails[key] = set()
            self.hrt2tails[key].add(ex['tail_id'])

        self.triplet_cnt = len(examples)

    def get_neighbors(self, h: str, r: str,t:str) -> set:
        return self.hrt2tails.get((h, r, t), set())


class EntityDict:

    def __init__(self, entity_dict_dir: str, inductive_test_path: str = None):
        path = os.path.join(entity_dict_dir, 'entities.json')
        assert os.path.exists(path)
        self.entity_exs = [EntityExample(**obj) for obj in json.load(open(path, 'r',encoding='latin1'))]

        if inductive_test_path:
            examples = json.load(open(inductive_test_path, 'r'))
            valid_entity_ids = set()
            for ex in examples:
                valid_entity_ids.add(ex['head_id'])
                valid_entity_ids.add(ex['tail_id'])
            self.entity_exs = [ex for ex in self.entity_exs if ex.entity_id in valid_entity_ids]

        self.id2entity = {ex.entity_id: ex for ex in self.entity_exs}
        self.entity2idx = {ex.entity_id: i for i, ex in enumerate(self.entity_exs)}
        # logger.info('Load {} entities from {}'.format(len(self.id2entity), path))

    def entity_to_idx(self, entity_id: str) -> int:
        return self.entity2idx[entity_id]

    def get_entity_by_id(self, entity_id: str) -> EntityExample:
        return self.id2entity[entity_id]

    def get_entity_by_idx(self, idx: int) -> EntityExample:
        return self.entity_exs[idx]

    def __len__(self):
        return len(self.entity_exs)


class LinkGraph:

    def __init__(self, train_path: str):
        # logger.info('Start to build link graph from {}'.format(train_path))
        # id -> set(id)
        self.graph = {}
        self.relation_graph = {}
        examples = json.load(open(train_path, 'r',encoding='latin1'))

        for ex in examples:
            head_id, tail_id = ex['head_id'], ex['tail_id']
            relation = ex['relation']
            relation_id = ex['relation_id']
            time = ex['time']
            if head_id not in self.graph:
                self.graph[head_id] = set()
            self.graph[head_id].add((tail_id,relation,relation_id,time))
            if (head_id,relation_id) not in self.relation_graph:
                self.relation_graph[(head_id,relation_id)] = set()
            self.relation_graph[(head_id,relation_id)].add((tail_id,relation,relation_id,time))
            if tail_id not in self.graph:
                self.graph[tail_id] = set()
            self.graph[tail_id].add((head_id,'inverse {}'.format(relation),relation_id,time))
            if (tail_id,relation_id) not in self.graph:
                self.relation_graph[(tail_id,relation_id)] = set()
            self.relation_graph[(tail_id,relation_id)].add((head_id,'inverse {}'.format(relation),relation_id,time))
        # logger.info('Done build link graph with {} nodes'.format(len(self.graph)))

    def get_neighbor_ids(self, entity_id: str, time,max_to_keep=3) -> List[str]:
        # make sure different calls return the same results
        neighbor_ids = self.graph.get((entity_id), set())
        sorted_lst = sorted(neighbor_ids, key=lambda x: int(time) - int(x[3]) if int(x[3]) < int(time) else float('inf'))
        closest_n = [x for x in sorted_lst if x[3] < time][:max_to_keep]
        return closest_n
        # return sorted(list(neighbor_ids))[:max_to_keep]

    def get_relation_neighbor_ids(self, entity_id: str,time,relation_id,max_to_keep=3) -> List[str]:
        # make sure different calls return the same results
        neighbor_ids = self.relation_graph.get((entity_id,relation_id), set())
        sorted_lst = sorted(neighbor_ids, key=lambda x: int(time) - int(x[3]) if int(x[3]) < int(time) else float('inf'))
        closest_n = [x for x in sorted_lst if x[3] < time][:max_to_keep]
        return closest_n

    def get_n_hop_entity_indices2(self, entity_id: str,time:int,
                                 entity_dict: EntityDict,
                                 n_hop: int = 2,
                                 # return empty if exceeds this number
                                 max_nodes: int = 100000) -> set:
        if n_hop < 0:
            return set()

        seen_eids = set()
        seen_eids.add((entity_id,time))
        queue = deque([(entity_id,time)])
        for i in range(n_hop):
            len_q = len(queue)
            for _ in range(len_q):
                tp = queue.popleft()
                for node in self.graph.get(tp, set()):
                    if node not in seen_eids:
                        queue.append((node,time))
                        seen_eids.add((node,time))
                        if len(seen_eids) > max_nodes:
                            return set()
        return set([entity_dict.entity_to_idx(e_id[0]) for e_id in seen_eids])

    def get_n_hop_entity_indices(self, entity_id: str,
                                 entity_dict: EntityDict,
                                 n_hop: int = 2,
                                 # return empty if exceeds this number
                                 max_nodes: int = 100000) -> set:
        if n_hop < 0:
            return set()

        seen_eids = set()
        seen_eids.add(entity_id)
        queue = deque([entity_id])
        for i in range(n_hop):
            len_q = len(queue)
            for _ in range(len_q):
                tp = queue.popleft()
                for node in self.graph.get(tp, set()):
                    if node not in seen_eids:
                        queue.append(node)
                        seen_eids.add(node)
                        if len(seen_eids) > max_nodes:
                            return set()
        return set([entity_dict.entity_to_idx(e_id) for e_id in seen_eids])


def reverse_triplet(obj):

    return {
        'head_id': obj['tail_id'],
        'head': obj['tail'],
        'relation': 'inverse {}'.format(obj['relation']),
        'relation_id':'inverse_{}'.format(obj['relation_id']),
        'tail_id': obj['head_id'],
        'tail': obj['head'],
        'time': obj['time']
    }
