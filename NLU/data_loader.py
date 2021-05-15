import os
import torch
import logging
from utils import *


class Dataset(torch.utils.data.Dataset):
    def __init__(self, src_seqs, trg_seqs, intent_ids, lens):
        self.src_seqs = src_seqs
        self.trg_seqs = trg_seqs
        self.intent_ids = intent_ids
        self.lens = lens

    def __len__(self):
        return len(self.intent_ids)

    def __getitem__(self, index):
        src_seq = torch.tensor(self.src_seqs[index], dtype=torch.long)
        trg_seq = torch.tensor(self.trg_seqs[index], dtype=torch.long)
        intent_id = torch.tensor(self.intent_ids[index], dtype=torch.long)
        length = torch.tensor(self.lens[index], dtype=torch.long)
        return {'src_seq': src_seq, 'trg_seq': trg_seq, 'intent_id': intent_id, 'length': length}


def featurize(args, src_indexer, trg_indexer, intent_indexer, split):
    dataset_dir = os.path.join(args.data_dir, args.dataset_name)
    input_text_file = os.path.join(dataset_dir, split, 'seq.in')
    slot_labels_file = os.path.join(dataset_dir, split, 'seq.out')
    intent_label_file = os.path.join(dataset_dir, split, 'label')
    src_seqs = []
    trg_seqs = []
    intent_ids = []
    lens = []
    with open(input_text_file, 'r') as text_fin, open(slot_labels_file, 'r') as slot_fin, open(intent_label_file, 'r') as intent_fin:
        for text, slot, intent in zip(text_fin, slot_fin, intent_fin):
            text_tokens = text.strip().split()
            src_seqLen = args.max_seqLen if len(text_tokens) > args.max_seqLen else len(text_tokens)
            slot_tokens = slot.strip().split()
            intent = intent.strip()
            if len(text_tokens) == len(slot_tokens):
                src_seq = [src_indexer.index_of(PAD)] * args.max_seqLen
                trg_seq = [trg_indexer.index_of(PAD)] * args.max_seqLen

                for i in range(src_seqLen):
                    if src_indexer.index_of(text_tokens[i]) >= 0:
                        src_seq[i] = src_indexer.index_of(text_tokens[i])
                    else:
                        src_seq[i] = src_indexer.index_of(UNK)

                for i in range(src_seqLen):
                    if trg_indexer.index_of(slot_tokens[i]) >= 0:
                        trg_seq[i] = trg_indexer.index_of(slot_tokens[i])
                    else:
                        trg_seq[i] = trg_indexer.index_of(UNK)

                if intent_indexer.index_of(intent) >= 0:
                    intent_id = intent_indexer.index_of(intent)
                else:
                    intent_id = intent_indexer.index_of(UNK)

                src_seqs.append(src_seq)
                trg_seqs.append(trg_seq)
                intent_ids.append(intent_id)
                lens.append(src_seqLen)
    item_count = (len(lens) // args.batch_size) * args.batch_size
    # item_count = 2
    return src_seqs[:item_count], trg_seqs[:item_count], intent_ids[:item_count], lens[:item_count]


def get_indexers(args):
    src_indexer = get_indexer(os.path.join(args.data_dir, args.dataset_name, args.word_vocab_file))
    trg_indexer = get_indexer(os.path.join(args.data_dir, args.dataset_name, args.slot_label_file))
    intent_indexer = get_indexer(os.path.join(args.data_dir, args.dataset_name, args.intent_label_file))

    logging.info(f'Size of vocabulary including PAD, UNK:'
                 f'\nSource text: {len(src_indexer)} (Including PAD, UNK, <sos>, <eos>)'
                 f'\nTarget slot: {len(trg_indexer)} (Including PAD, UNK, <sos>, <eos>)'
                 f'\nIntent: {len(intent_indexer)} (Including UNK)\n')
    return src_indexer, trg_indexer, intent_indexer


def prepare_dataset(args, src_indexer, trg_indexer, intent_indexer, split):
    src_seqs, trg_seqs, intent_ids, lens = featurize(args, src_indexer, trg_indexer, intent_indexer, split)
    dataset = Dataset(src_seqs, trg_seqs, intent_ids, lens)
    return dataset
