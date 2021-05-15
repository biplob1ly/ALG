# All utility methods goes here
import numpy as np
import torch
import random
import logging
import os
import regex
from nltk.translate.nist_score import corpus_nist
from nlgeval import NLGEval


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def init_logger(args):
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    logging.basicConfig(
        filename=os.path.join(args.result_dir, 'app.log'),
        filemode='w',
        format='%(asctime)s - %(message)s',
        datefmt=r'%m/%d/%Y %H:%M:%S',
        level=logging.INFO)


# count Model params
def count_model_params(model):
    total_params = sum([p.numel()
                        for p in model.parameters() if p.requires_grad])
    return total_params


def get_eval_metrics(references, hypotheses, result_dir):
    ref_path = os.path.join(result_dir, 'ref.txt')
    hyp_path = os.path.join(result_dir, 'hyp.txt')
    ref_sents = []
    hyp_sents = []
    with open(ref_path, 'w') as ref_out, open(hyp_path, 'w') as hyp_out:
        for ref_list, hyp in zip(references, hypotheses):
            ref_sent = ' '.join(ref_list[0])
            ref_sents.append(ref_sent)
            ref_out.write(ref_sent + '\n')

            hyp_sent = ' '.join(hyp)
            hyp_sents.append(hyp_sent)
            hyp_out.write(hyp_sent + '\n')

    eval_metrics = {}
    eval_metrics['nist_nltk'] = corpus_nist(references, hypotheses)  # tokenized

    nlgeval = NLGEval()  # loads the models
    metrics_dict = nlgeval.compute_metrics([ref_sents], hyp_sents)
    eval_metrics['nlgeval'] = metrics_dict

    return eval_metrics


def tokenize(text):
    """Tokenize the given text (i.e., insert spaces around all tokens)"""
    toks = ' ' + text + ' '  # for easier regexes

    # enforce space around all punct
    toks = regex.sub(r'(([^\p{IsAlnum}\s\.\,−\-])\2*)', r' \1 ', toks)  # all punct (except ,-.)
    toks = regex.sub(r'([^\p{N}])([,.])([^\p{N}])', r'\1 \2 \3', toks)  # ,. & no numbers
    toks = regex.sub(r'([^\p{N}])([,.])([\p{N}])', r'\1 \2 \3', toks)  # ,. preceding numbers
    toks = regex.sub(r'([\p{N}])([,.])([^\p{N}])', r'\1 \2 \3', toks)  # ,. following numbers
    toks = regex.sub(r'(–-)([^\p{N}])', r'\1 \2', toks)  # -/– & no number following
    # toks = regex.sub(r'(\p{N} *|[^ ])(-)', r'\1\2 ', toks)  # -/– & preceding number/no-space
    # toks = regex.sub(r'([-−])', r' \1', toks)  # -/– : always space before

    # keep apostrophes together with words in most common contractions
    toks = regex.sub(r'([\'’´]) (s|m|d|ll|re|ve)\s', r' \1\2 ', toks)  # I 'm, I 've etc.
    toks = regex.sub(r'(n [\'’´]) (t\s)', r' \1\2 ', toks)  # do n't

    # other contractions, as implemented in Treex
    toks = regex.sub(r' ([Cc])annot\s', r' \1an not ', toks)
    toks = regex.sub(r' ([Dd]) \' ye\s', r' \1\' ye ', toks)
    toks = regex.sub(r' ([Gg])imme\s', r' \1im me ', toks)
    toks = regex.sub(r' ([Gg])onna\s', r' \1on na ', toks)
    toks = regex.sub(r' ([Gg])otta\s', r' \1ot ta ', toks)
    toks = regex.sub(r' ([Ll])emme\s', r' \1em me ', toks)
    toks = regex.sub(r' ([Mm])ore\'n\s', r' \1ore \'n ', toks)
    toks = regex.sub(r' \' ([Tt])is\s', r' \'\1 is ', toks)
    toks = regex.sub(r' \' ([Tt])was\s', r' \'\1 was ', toks)
    toks = regex.sub(r' ([Ww])anna\s', r' \1an na ', toks)

    # clean extra space
    toks = regex.sub(r'\s+', ' ', toks)
    toks = toks.strip()
    return toks


def elapsed_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_indexer(vocab_filepath):
    indexer = Indexer()
    with open(vocab_filepath, 'r') as fin:
        for line in fin:
            token = line.strip()
            indexer.add_and_get_index(token)
    return indexer


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
