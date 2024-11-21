import os
import glob

from transformers import AutoTokenizer

from config import args
from triplet import TripletDict, EntityDict, LinkGraph
from logger_config import logger

train_triplet_dict: TripletDict = None
all_triplet_dict: TripletDict = None
link_graph: LinkGraph = None
entity_dict: EntityDict = None
tokenizer: AutoTokenizer = None

def _init_entity_dict():
    global entity_dict
    if not entity_dict:
        entity_dict = EntityDict(entity_dict_dir=os.path.dirname(args.valid_path))


def _init_train_triplet_dict():
    global train_triplet_dict
    if not train_triplet_dict:
        train_triplet_dict = TripletDict(path_list=[args.train_path],args = args)


def _init_all_triplet_dict():
    global all_triplet_dict
    if not all_triplet_dict:
        path_pattern = '{}/*.txt.json'.format(os.path.dirname(args.train_path))
        all_triplet_dict = TripletDict(path_list=glob.glob(path_pattern),args=args)


def _init_link_graph():
    global link_graph
    if not link_graph:
        link_graph = LinkGraph(train_path=args.train_path)


def get_entity_dict():
    _init_entity_dict()
    return entity_dict


def get_train_triplet_dict():
    _init_train_triplet_dict()
    return train_triplet_dict


def get_all_triplet_dict():
    _init_all_triplet_dict()
    return all_triplet_dict


def get_link_graph():
    _init_link_graph()
    return link_graph

def build_tokenizer(args):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        logger.info('Build tokenizer from {}'.format(args.pretrained_model))
        relation_names = []
        if args.task == "yago":
            entity_num = 10623
            rel_num = 10
            time_range = 189
        elif args.task == "wiki":
            entity_num = 12554
            rel_num = 24
            time_range = 232
        else:
            print("time_range")
        for i in range(rel_num):
            rel_sep = f'[R_{i}_SEP1]'
            rel_sep2 = f'[R_{i}_SEP2]'
            rel_sep3 = f'[R_{i}_SEP3]'
            rel_sep4 = f'[R_{i}_SEP4]'
            inverse_rel_sep = f'[R_inverse_{i}_SEP1]'
            inverse_rel_sep2 = f'[R_inverse_{i}_SEP2]'
            inverse_rel_sep3 = f'[R_inverse_{i}_SEP3]'
            inverse_rel_sep4 = f'[R_inverse_{i}_SEP4]'
            relation_names.append(rel_sep)
            relation_names.append(rel_sep2)
            relation_names.append(rel_sep3)
            relation_names.append(rel_sep4)
            relation_names.append(inverse_rel_sep)
            relation_names.append(inverse_rel_sep2)
            relation_names.append(inverse_rel_sep3)
            relation_names.append(inverse_rel_sep4)
        for i in  range(time_range):
            time_sep = f'[Time_{i}_SEP1]'
            relation_names.append(time_sep)
        relation_names.append("[neighbor_token]")
        num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': relation_names})



def get_tokenizer(entity_names=None):
    build_tokenizer(args)
    return tokenizer