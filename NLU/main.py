import torch
import logging
from utils import *
import argparse
from collections import OrderedDict, namedtuple
from itertools import product
from data_loader import get_indexers, prepare_dataset
from trainer import Trainer


def get_hyper_params_combinations(args):
    params = OrderedDict(
        lr=args.lr,
        num_epochs=args.num_epochs
    )

    HyperParams = namedtuple('HyperParams', params.keys())
    hyper_params_list = []
    for v in product(*params.values()):
        hyper_params_list.append(HyperParams(*v))
    return hyper_params_list


def run(args):
    src_indexer, trg_indexer, intent_indexer = get_indexers(args)
    train_set = prepare_dataset(args, src_indexer, trg_indexer, intent_indexer, 'train')
    dev_set = prepare_dataset(args, src_indexer, trg_indexer, intent_indexer, 'dev')
    test_set = prepare_dataset(args, src_indexer, trg_indexer, intent_indexer, 'test')
    logging.info(f'Size of dataset:'
                 f'\nTrain: {len(train_set)}'
                 f'\nDev: {len(dev_set)}'
                 f'\nTest: {len(test_set)}\n')

    model_config = {
        'vocab_size': len(src_indexer),
        'embed_size': args.embed_size,
        'hidden_size': args.hidden_size,
        'intent_size': len(intent_indexer),
        'slot_size': len(trg_indexer),
        'SOS_idx': trg_indexer.index_of(SOS)
    }
    # trainer = Trainer(train_set, dev_set, test_set, args, model_config)
    # for hyper_params in get_hyper_params_combinations(args):
    #     logging.info(f'Training with: {hyper_params}')
    #     trainer.train(hyper_params)
    #     trainer.evaluate('dev')

    if args.do_eval:
        logging.info('\n')
        if args.model_type == 'RNN':
            loaded_trainer = Trainer.reload_model(train_set, dev_set, test_set, args.model_dir)
            loaded_trainer.evaluate('test')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data', type=str,
                        help='Root dir path for save data.')
    parser.add_argument('--model_dir', default='./models', type=str,
                        help='Path to save training model.')
    parser.add_argument('--result_dir', default='./results', type=str,
                        help='Path to save results.')
    parser.add_argument('--dataset_name', default='atis', choices=['snips', 'atis'],
                        type=str, help='Select train Model task:[atis,snips].Required Argument.')
    parser.add_argument('--intent_label_file', default='intent_label.txt',
                        type=str, help='File path for loading intent_label vocab')
    parser.add_argument('--slot_label_file', default='slot_label.txt',
                        type=str, help='File path for loading slot_label vocab ')
    parser.add_argument('--word_vocab_file', default='word_vocab.txt',
                        type=str, help='File path for loading word vocab ')

    parser.add_argument('--model_type', default='RNN', type=str)
    parser.add_argument('--random_seed', type=int,
                        default=1234, help='set random seed')
    parser.add_argument('--max_seqLen', type=int, default=50,
                        help='Set max sequence len After tokenize text.Default is 50')
    parser.add_argument('--embed_size', type=int, default=128,
                        help='Embedding dimension for text token')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Embedding dimension for text token')

    parser.add_argument('--batch_size', type=int, default=2,
                        help='Train model Batch size.Default is 64.')
    parser.add_argument('--lr', type=float, default=[5e-5], nargs='+',
                        help='Learning rate for AdamW.Default is 5e-5')
    parser.add_argument('--num_epochs', type=int, default=[20], nargs='+',
                        help='If>0:set number of train model epochs.Default is 10')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0, help='L2 weight regularization for AdamW.Default is 0')
    parser.add_argument('--adam_epsilon', type=float,
                        default=1e-8, help='Epsilon for Adam optimizer.')

    parser.add_argument('--max_steps', type=int, default=0,
                        help='If >0:set total_number of train step.Override num_train_epochsDefault is 0')
    parser.add_argument('--warm_steps', type=int,
                        default=0, help='Linear Warm up steps.Default is 0')
    parser.add_argument('--grad_accumulate_steps', type=int,
                        default=1, help='Number of update gradient to accumulate before update model.Default is 1.')
    parser.add_argument('--logging_steps', type=int, default=200,
                        help='Every X train step to logging model info.Default is 200.')
    parser.add_argument('--save_steps', type=int, default=200,
                        help='Every X train step to save Model.Default is 200.')

    parser.add_argument('--max_norm', type=float,
                        default=1.0, help='Max norm to avoid gradinet exploding.Default is 1')
    parser.add_argument('--dropout', type=float,
                        default=0.1, help='Dropout rate.Default is 0.1')
    parser.add_argument('--slot_loss_coef', type=float,
                        default=1.0, help='Slot loss coefficient.Default is 1')
    parser.add_argument('--use_crf', action='store_true',
                        help='Whether to using CRF layer for slot pred')

    parser.add_argument('--do_train', action='store_false',
                        help='Whether to run training')
    parser.add_argument('--do_eval', action='store_false',
                        help='Whether to run evaluate')

    parser.add_argument('--slot_pad_label', type=str, default='PAD',
                        help='Pad token for slot label(Noe contribute loss)')
    parser.add_argument('--ignore_index', type=int, default=0,
                        help='Specifies a target value that not contribute loss and gradient.Default is 0')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    init_logger(args)
    logging.info(f'{args}\n')
    set_random_seed(args.random_seed)

    run(args)