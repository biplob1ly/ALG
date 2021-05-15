import logging
from utils import *
import argparse
from collections import OrderedDict, namedtuple
from itertools import product
from data_loader import E2E_Featurizer, DineFeaturizer, prepare_dataset
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
    if args.dataset_name == 'e2e':
        featurizer = E2E_Featurizer(args)
    elif args.dataset_name == 'dineEquity':
        featurizer = DineFeaturizer(args)
    else:
        raise Exception('Unknown Dataset Name!')

    train_set = prepare_dataset(featurizer, 'train')
    dev_set = prepare_dataset(featurizer, 'dev')
    test_set = prepare_dataset(featurizer, 'test')
    logging.info(f'Size of dataset:'
                 f'\nTrain: {len(train_set)}'
                 f'\nDev: {len(dev_set)}'
                 f'\nTest: {len(test_set)}\n')

    model_config = {
        'src_vocab_size': len(featurizer.src_indexer),
        'trg_vocab_size': len(featurizer.trg_indexer),
        'embed_size': args.embed_size,
        'hidden_size': args.hidden_size
    }

    if args.do_train:
        print('\nTraining....\n')
        trainer = Trainer(train_set, dev_set, test_set, args, featurizer, model_config)
        for hyper_params in get_hyper_params_combinations(args):
            logging.info(f'Training with: {hyper_params}')
            trainer.train(hyper_params)

    if args.do_eval:
        print('\nEvaluating....\n')
        logging.info('\n')
        loaded_trainer = Trainer.reload_model(train_set, dev_set, test_set, featurizer, args.model_dir)
        loaded_trainer.evaluate('test')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data', type=str,
                        help='Root dir path for save data.')
    parser.add_argument('--model_dir', default='./models', type=str,
                        help='Path to save training model.')
    parser.add_argument('--result_dir', default='./results', type=str,
                        help='Path to save results.')
    parser.add_argument('--dataset_name', default='e2e', choices=['e2e', 'dineEquity'],
                        type=str, help='Select train Model task:[e2e,dstc8].Required Argument.')

    parser.add_argument('--model_type', default='RNN', type=str)
    parser.add_argument('--random_seed', type=int,
                        default=1234, help='set random seed')
    parser.add_argument('--max_mr_count', type=int, default=10,
                        help='Set max mr count (for e2e) After tokenize text. Default is 10')
    parser.add_argument('--max_src_len', type=int, default=120,
                        help='Set max source len (for dineEquity) After tokenize text.Default is 120')
    parser.add_argument('--max_trg_len', type=int, default=120,
                        help='Set max target len After tokenize text.Default is 120')
    parser.add_argument('--embed_size', type=int, default=128,
                        help='Embedding dimension for text token')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Embedding dimension for text token')

    parser.add_argument('--batch_size', type=int, default=2,
                        help='Train model Batch size.Default is 32.')
    parser.add_argument('--lr', type=float, default=[5e-3], nargs='+',
                        help='Learning rate for AdamW.Default is 5e-5')
    parser.add_argument('--num_epochs', type=int, default=[2], nargs='+',
                        help='If>0:set number of train model epochs.Default is 10')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0, help='L2 weight regularization for AdamW.Default is 0')
    parser.add_argument('--adam_epsilon', type=float,
                        default=1e-8, help='Epsilon for Adam optimizer.')

    parser.add_argument('--warm_steps', type=int,
                        default=0, help='Linear Warm up steps.Default is 0')

    parser.add_argument('--do_train', action='store_false',
                        help='Whether to run training')
    parser.add_argument('--do_eval', action='store_false',
                        help='Whether to run evaluate')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    init_logger(args)
    logging.info(f'{args}\n')
    set_random_seed(args.random_seed)

    run(args)
