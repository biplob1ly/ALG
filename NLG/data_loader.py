import os
import torch
from utils import *


class DAI:
    def __init__(self, type, slot=None, value=None):
        self.type = type
        self.slot = slot
        self.value = value

    def __str__(self):
        if self.slot is None:
            return self.type + '()'
        if self.value is None:
            return self.type + '(' + self.slot + ')'
        quote = '\'' if (' ' in self.value or ':' in self.value) else ''
        return self.type + '(' + self.slot + '=' + quote + self.value + quote + ')'

    def __repr__(self):
        return 'DAI.parse("' + str(self) + '")'
    
    def __eq__(self, other):
        return (self.type == other.type and
                self.slot == other.slot and
                self.value == other.value)
    
    @staticmethod
    def parse_dai(dai_text):
        type, svp = dai_text[:-1].split('(', 1)

        if not svp:  # no slot + value (e.g. 'hello()')
            return DAI(type)

        if '=' not in svp:  # no value (e.g. 'request(to_stop)')
            return DAI(type, svp)

        slot, value = svp.split('=', 1)
        if value.endswith('"#'):  # remove special '#' characters in Bagel data (TODO treat right)
            value = value[:-1]
        if value[0] in ['"', '\'']:  # remove quotes
            value = value[1:-1]
        return DAI(type, slot, value)

    @staticmethod
    def parse_mr(src_text):
        """Parse a DA string into DAIs (DA types, slots, and values)."""
        dais = []
        for dai_text in src_text[:-1].split(')&'):
            dais.append(DAI.parse_dai(dai_text + ')'))
        return dais


class Featurizer:
    PAD = '<pad>'
    SOS = '<sos>'
    EOS = '<eos>'
    UNK = '<unk>'
    src_indexer = Indexer()
    trg_indexer = Indexer()

    def __init__(self, args):
        self.args = args


class DineFeaturizer(Featurizer):
    def __init__(self, args):
        super().__init__(args)
        self.init_indexers()

    def init_indexers(self):
        self.src_indexer.add_and_get_index(self.PAD)
        self.src_indexer.add_and_get_index(self.SOS)
        self.src_indexer.add_and_get_index(self.EOS)
        self.src_indexer.add_and_get_index(self.UNK)

        self.trg_indexer.add_and_get_index(self.PAD)
        self.trg_indexer.add_and_get_index(self.SOS)
        self.trg_indexer.add_and_get_index(self.EOS)
        self.trg_indexer.add_and_get_index(self.UNK)

        dataset_dir = os.path.join(self.args.data_dir, self.args.dataset_name)
        src_file = os.path.join(dataset_dir, 'train', 'query.txt')
        trg_file = os.path.join(dataset_dir, 'train', 'response.txt')

        with open(src_file, 'r') as fin:
            for line in fin:
                cleaned_text = tokenize(line.strip())
                tokens = list(filter(bool, cleaned_text.split()))
                for token in tokens:
                    self.src_indexer.add_and_get_index(token)

        with open(trg_file, 'r') as fin:
            for line in fin:
                cleaned_text = tokenize(line.strip())
                tokens = list(filter(bool, cleaned_text.split()))
                for token in tokens:
                    self.trg_indexer.add_and_get_index(token)

        logging.info(f'Size of vocabulary including UNK:'
                     f'Source Tokens: {len(self.src_indexer)}'
                     f'\nTarget Tokens: {len(self.trg_indexer)}\n')

    def featurize_src(self, src_line):
        src_seq = [self.src_indexer.index_of(self.PAD)] * self.args.max_src_len
        src = tokenize(src_line.strip())
        src_tokens = list(filter(bool, src.split()))
        src_token_count = self.args.max_src_len - 2 if len(src_tokens) > self.args.max_src_len - 2 else len(src_tokens)
        src_len = 0
        src_seq[src_len] = self.src_indexer.index_of(self.SOS)
        src_len += 1
        for i in range(src_token_count):
            token_idx = self.src_indexer.index_of(src_tokens[i])
            src_seq[src_len] = token_idx if token_idx >= 0 else self.src_indexer.index_of(self.UNK)
            src_len += 1
        src_seq[src_len] = self.src_indexer.index_of(self.EOS)
        src_len += 1
        return src_seq

    def featurize_trg(self, trg_line):
        trg_seq = [self.trg_indexer.index_of(self.PAD)] * self.args.max_trg_len
        trg = tokenize(trg_line.strip())
        trg_tokens = list(filter(bool, trg.split()))
        trg_token_count = self.args.max_trg_len - 2 if len(trg_tokens) > self.args.max_trg_len - 2 else len(trg_tokens)
        trg_len = 0
        trg_seq[trg_len] = self.trg_indexer.index_of(self.SOS)
        trg_len += 1
        for i in range(trg_token_count):
            token_idx = self.trg_indexer.index_of(trg_tokens[i])
            trg_seq[trg_len] = token_idx if token_idx >= 0 else self.trg_indexer.index_of(self.UNK)
            trg_len += 1
        trg_seq[trg_len] = self.trg_indexer.index_of(self.EOS)
        trg_len += 1
        return trg_seq

    def featurize_all(self, split):
        dataset_dir = os.path.join(self.args.data_dir, self.args.dataset_name)
        src_file = os.path.join(dataset_dir, split, 'query.txt')
        trg_file = os.path.join(dataset_dir, split, 'response.txt')

        src_seqs = []
        trg_seqs = []
        with open(src_file, 'r') as src_fin, open(trg_file, 'r') as trg_fin:
            for src_line, trg_line in zip(src_fin, trg_fin):
                src_seq = self.featurize_src(src_line)
                src_seqs.append(src_seq)
                trg_seq = self.featurize_trg(trg_line)
                trg_seqs.append(trg_seq)

        item_count = (len(src_seqs) // self.args.batch_size) * self.args.batch_size
        item_count = 4
        return src_seqs[:item_count], trg_seqs[:item_count]


class E2E_Featurizer(Featurizer):
    UNK_DA = '<unk_da>'
    UNK_SLOT = '<unk_slot>'
    UNK_VALUE = '<unk_vaue>'
    da_types = set()
    slots = set()
    values = set()

    def __init__(self, args):
        super().__init__(args)
        self.init_indexers()

    def init_indexers(self):
        self.src_indexer.add_and_get_index(self.PAD)
        self.src_indexer.add_and_get_index(self.SOS)
        self.src_indexer.add_and_get_index(self.EOS)
        self.src_indexer.add_and_get_index(self.UNK_DA)
        self.da_types.add(self.UNK_DA)
        self.src_indexer.add_and_get_index(self.UNK_SLOT)
        self.slots.add(self.UNK_SLOT)
        self.src_indexer.add_and_get_index(self.UNK_VALUE)
        self.values.add(self.UNK_VALUE)
        self.trg_indexer.add_and_get_index(self.PAD)
        self.trg_indexer.add_and_get_index(self.SOS)
        self.trg_indexer.add_and_get_index(self.EOS)
        self.trg_indexer.add_and_get_index(self.UNK)

        dataset_dir = os.path.join(self.args.data_dir, self.args.dataset_name)
        src_file = os.path.join(dataset_dir, 'train', 'query.txt')
        trg_file = os.path.join(dataset_dir, 'train', 'response.txt')

        max_mr_count = 0
        with open(src_file, 'r') as fin:
            for line in fin:
                dais = DAI.parse_mr(line.strip())
                if len(dais) > max_mr_count:
                    max_mr_count = len(dais)
                for dai in dais:
                    self.src_indexer.add_and_get_index(dai.type)
                    self.da_types.add(dai.type)
                    self.src_indexer.add_and_get_index(dai.slot)
                    self.slots.add(dai.slot)
                    self.src_indexer.add_and_get_index(dai.value)
                    self.values.add(dai.value)
        logging.info(f"Max mr count: {max_mr_count}")

        with open(trg_file, 'r') as fin:
            for line in fin:
                trg = tokenize(line.strip())
                trg_tokens = list(filter(bool, trg.split()))
                for trg_token in trg_tokens:
                    self.trg_indexer.add_and_get_index(trg_token)

        logging.info(f'Size of vocabulary including UNK:'
                     f'\nDialog Acts: {len(self.da_types)}'
                     f'\nSlots: {len(self.slots)}'
                     f'\nValues: {len(self.values)}'
                     f'\nReference Tokens: {len(self.trg_indexer)}\n')

    def featurize_together(self, split):
        dataset_dir = os.path.join(self.args.data_dir, self.args.dataset_name)
        src_file = os.path.join(dataset_dir, split, 'query.txt')
        trg_file = os.path.join(dataset_dir, split, 'response.txt')

        src_seqs = []
        src_lens = []
        with open(src_file, 'r') as fin:
            for line in fin:
                src_seq = [self.src_indexer.index_of(self.PAD)] * (self.args.max_mr_count * 3 + 2)
                dais = DAI.parse_mr(line.strip())
                mr_count = self.args.max_mr_count if len(dais) > self.args.max_mr_count else len(dais)
                src_len = 0
                src_seq[src_len] = self.src_indexer.index_of(self.SOS)
                src_len += 1
                for i in range(mr_count):
                    da_type_idx = self.src_indexer.index_of(dais[i].type)
                    slot_idx = self.src_indexer.index_of(dais[i].slot)
                    value_idx = self.src_indexer.index_of(dais[i].value)
                    src_seq[src_len] = da_type_idx if da_type_idx >= 0 else self.src_indexer.index_of(self.UNK_DA)
                    src_len += 1
                    src_seq[src_len] = slot_idx if slot_idx >= 0 else self.src_indexer.index_of(self.UNK_SLOT)
                    src_len += 1
                    src_seq[src_len] = value_idx if value_idx >= 0 else self.src_indexer.index_of(self.UNK_VALUE)
                    src_len += 1
                src_seq[src_len] = self.src_indexer.index_of(self.EOS)
                src_len += 1
                src_seqs.append(src_seq)
                src_lens.append(src_len)

        trg_seqs = []
        with open(trg_file, 'r') as fin:
            for line in fin:
                trg_seq = [self.trg_indexer.index_of(self.PAD)] * self.args.max_trg_len
                trg = tokenize(line.strip())
                trg_tokens = list(filter(bool, trg.split()))
                trg_token_count = self.args.max_trg_len-2 if len(trg_tokens) > self.args.max_trg_len-2 else len(trg_tokens)
                trg_len = 0
                trg_seq[trg_len] = self.trg_indexer.index_of(self.SOS)
                trg_len += 1
                for i in range(trg_token_count):
                    token_idx = self.trg_indexer.index_of(trg_tokens[i])
                    trg_seq[trg_len] = token_idx if token_idx >= 0 else self.trg_indexer.index_of(self.UNK)
                    trg_len += 1
                trg_seq[trg_len] = self.trg_indexer.index_of(self.EOS)
                trg_len += 1
                trg_seqs.append(trg_seq)

        item_count = (len(src_lens) // self.args.batch_size) * self.args.batch_size
        # item_count = 4
        return src_seqs[:item_count], src_lens[:item_count], trg_seqs[:item_count]

    def featurize_src(self, src_line):
        src_seq = []
        dais = DAI.parse_mr(src_line.strip())
        mr_count = self.args.max_mr_count if len(dais) > self.args.max_mr_count else len(dais)
        src_seq.append(self.src_indexer.index_of(self.SOS))
        for i in range(mr_count):
            da_type_idx = self.src_indexer.index_of(dais[i].type)
            slot_idx = self.src_indexer.index_of(dais[i].slot)
            value_idx = self.src_indexer.index_of(dais[i].value)
            src_seq.append(da_type_idx if da_type_idx >= 0 else self.src_indexer.index_of(self.UNK_DA))
            src_seq.append(slot_idx if slot_idx >= 0 else self.src_indexer.index_of(self.UNK_SLOT))
            src_seq.append(value_idx if value_idx >= 0 else self.src_indexer.index_of(self.UNK_VALUE))
        src_seq.append(self.src_indexer.index_of(self.EOS))
        return src_seq

    def featurize_trg(self, trg_line):
        trg_seq = [self.trg_indexer.index_of(self.PAD)] * self.args.max_trg_len
        trg = tokenize(trg_line.strip())
        trg_tokens = list(filter(bool, trg.split()))
        trg_token_count = self.args.max_trg_len - 2 if len(trg_tokens) > self.args.max_trg_len - 2 else len(trg_tokens)
        trg_len = 0
        trg_seq[trg_len] = self.trg_indexer.index_of(self.SOS)
        trg_len += 1
        for i in range(trg_token_count):
            token_idx = self.trg_indexer.index_of(trg_tokens[i])
            trg_seq[trg_len] = token_idx if token_idx >= 0 else self.trg_indexer.index_of(self.UNK)
            trg_len += 1
        trg_seq[trg_len] = self.trg_indexer.index_of(self.EOS)
        trg_len += 1
        return trg_seq

    def featurize_all(self, split):
        dataset_dir = os.path.join(self.args.data_dir, self.args.dataset_name)
        src_file = os.path.join(dataset_dir, split, 'query.txt')
        trg_file = os.path.join(dataset_dir, split, 'response.txt')

        src_seqs = []
        trg_seqs = []
        with open(src_file, 'r') as src_fin, open(trg_file, 'r') as trg_fin:
            for src_line, trg_line in zip(src_fin, trg_fin):
                src_seq = self.featurize_src(src_line)
                src_seqs.append(src_seq)
                trg_seq = self.featurize_trg(trg_line)
                trg_seqs.append(trg_seq)

        item_count = (len(src_seqs) // self.args.batch_size) * self.args.batch_size
        item_count = 4
        return src_seqs[:item_count], trg_seqs[:item_count]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, src_seqs, trg_seqs):
        self.src_seqs = src_seqs
        self.trg_seqs = trg_seqs

    def __len__(self):
        return len(self.src_seqs)

    def __getitem__(self, index):
        src_seq = torch.tensor(self.src_seqs[index], dtype=torch.long)
        trg_seq = torch.tensor(self.trg_seqs[index], dtype=torch.long)
        return src_seq, trg_seq


# Prepares pytorch dataset
def prepare_dataset(featurizer, split):
    src_seqs, trg_seqs = featurizer.featurize_all(split)
    dataset = Dataset(src_seqs, trg_seqs)
    return dataset

