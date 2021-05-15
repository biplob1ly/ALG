import numpy as np
import torch
import random
import logging
import os
from seqeval.metrics import f1_score, recall_score, precision_score

PAD = 'PAD'
UNK = 'UNK'
SOS = '<sos>'
EOS = '<eos>'

# MODEL_CLASSES = {
#     'joint_bert': (BertConfig, BertTokenizer, modeling_JointBert.JointBert),
#     'joint_AttnS2S': (modeling_JointRnn.RnnConfig, get_word_vocab, modeling_JointRnn.Joint_AttnSeq2Seq),
# }
# MODEL_PATH = {
#     'joint_bert': 'bert-base-uncased',
#     'joint_AttnS2S': './atis_model/joint_attnS2S'
# }


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_logger(args):
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    logging.basicConfig(
        filename=os.path.join(args.result_dir, 'app.log'),
        filemode='w',
        format='%(asctime)s - %(message)s',
        datefmt=r'%m/%d/%Y %H:%M:%S',
        level=logging.INFO)


def get_indexer(vocab_filepath):
    indexer = Indexer()
    with open(vocab_filepath, 'r') as fin:
        for line in fin:
            token = line.strip()
            indexer.add_and_get_index(token)
    return indexer


# count Model params
def count_model_params(model):
    total_params = sum([p.numel()
                        for p in model.parameters() if p.requires_grad])
    return total_params


def get_intent_accuracy(intent_preds, intent_labels):
    assert len(intent_preds) == len(intent_labels)
    corrects = (intent_preds == intent_labels)
    intent_accuracy = corrects.mean()
    return intent_accuracy


def get_slot_metrics(slot_preds, slot_labels):
    assert len(slot_preds) == len(slot_labels)
    # print('slot preds:')
    # print(slot_preds)
    # print('slot_labels')
    # print(slot_labels)
    slot_precision = precision_score(slot_labels, slot_preds)
    slot_recall = recall_score(slot_labels, slot_preds)
    slot_f1 = f1_score(slot_labels, slot_preds)
    return slot_precision, slot_recall, slot_f1


def get_semantic_frame_accuracy(intent_preds, intent_labels, slot_preds, slot_labels):
    correct_intents = (intent_preds == intent_labels)
    correct_slots = []
    for pred, label in zip(slot_preds, slot_labels):
        is_correct = True
        for p, l in zip(pred, label):
            if p != l:
                is_correct = False
                break
        correct_slots.append(is_correct)

    correct_slots = np.array(correct_slots)
    semantic_frame_accuracy = np.multiply(correct_intents, correct_slots).mean()
    return semantic_frame_accuracy


class Indexer():
    """
    Bijection between objects and integers starting at 0. Useful for mapping
    labels, features, etc. into coordinates of a vector space.

    Attributes:
        objs_to_ints
        ints_to_objs
    """
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        """
        :param index: integer index to look up
        :return: Returns the object corresponding to the particular index or None if not found
        """
        if index not in self.ints_to_objs:
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        """
        :param object: object to look up
        :return: Returns True if it is in the Indexer, False otherwise
        """
        return self.index_of(object) != -1

    def index_of(self, object):
        """
        :param object: object to look up
        :return: Returns -1 if the object isn't present, index otherwise
        """
        if object not in self.objs_to_ints:
            return -1
        else:
            return self.objs_to_ints[object]

    def add_and_get_index(self, object, add=True):
        """
        Adds the object to the index if it isn't present, always returns a nonnegative index
        :param object: object to look up or add
        :param add: True by default, False if we shouldn't add the object. If False, equivalent to index_of.
        :return: The index of the object
        """
        if not add:
            return self.index_of(object)
        if object not in self.objs_to_ints:
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]




