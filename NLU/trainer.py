import torch
import logging
from torch import nn
from torch.utils.data import DataLoader
from run_manager import RunManager
import torch.nn.functional as F
from models.rnn_model import JointEncoderDecoder
from utils import *
import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
import os
import json


class Trainer:
    def __init__(self, train_set, dev_set, test_set, args,
                 model_config, pretrained_path=None):
        self.train_set = train_set
        self.dev_set = dev_set
        self.test_set = test_set
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trg_indexer = get_indexer(os.path.join(args.data_dir, args.dataset_name, args.slot_label_file))
        self.PAD_idx = self.trg_indexer.index_of(PAD)

        self.model_config = model_config
        self.vocab_size = model_config['vocab_size']
        self.intent_size = model_config['intent_size']
        self.slot_size = model_config['slot_size']
        self.embed_size = model_config['embed_size']
        self.hidden_size = model_config['hidden_size']
        self.SOS_idx = model_config['SOS_idx']

        if args.model_type == 'RNN':
            self.model = JointEncoderDecoder(self.vocab_size, self.embed_size, self.hidden_size,
                                             self.intent_size, self.slot_size, self.SOS_idx)
            if not pretrained_path:
                self.model.init_weight()
            else:
                # loading pretrained weights
                params_path = os.path.join(pretrained_path, 'pretrain_model.pt')
                model_params = torch.load(params_path)['model_state_dict']
                self.model.load_state_dict(model_params)
        self.model.to(self.device)
        self.run_manager = RunManager()

    def train(self, hyper_params):
        train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True, num_workers=1)
        criterion = nn.CrossEntropyLoss(ignore_index=self.PAD_idx)

        no_decay = ['LayerNorm', 'bias']
        param_gropus = [
            {'params': [p for n, p in self.model.named_parameters() if not any([nd in n for nd in no_decay])],
             'weight_decay':self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any([nd in n for nd in no_decay])],
             'weight_decay':0.0},
        ]

        optimizer = AdamW(param_gropus, lr=hyper_params.lr,
                          eps=self.args.adam_epsilon)

        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_loader) * hyper_params.num_epochs
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, self.args.warm_steps, total_steps)

        self.run_manager.begin_run(hyper_params, self.model, train_loader)
        self.model.train()
        for epoch in tqdm.trange(hyper_params.num_epochs, desc='Epoch'):
            self.run_manager.begin_epoch(epoch + 1)
            for batch in train_loader:
                src_seqs = batch['src_seq']
                trg_seqs = batch['trg_seq']
                intent_ids = batch['intent_id']
                lens = batch['length']
                if self.args.model_type == 'RNN':
                    lens, idxs = lens.sort(0, descending=True)
                    src_seqs = src_seqs[idxs].to(self.device)
                    trg_seqs = trg_seqs[idxs].to(self.device)
                    lens = lens.to(self.device)
                    intent_ids = intent_ids[idxs].to(self.device)

                intent_logits, slot_logits = self.model(src_seqs, trg_seqs, lens)
                intent_loss = criterion(intent_logits, intent_ids)
                slot_loss = criterion(slot_logits.reshape(-1, self.slot_size), trg_seqs.reshape(-1))
                loss = intent_loss + slot_loss

                optimizer.zero_grad()
                loss.backward()
                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                self.run_manager.track_loss(loss)

            self.run_manager.end_epoch()
        self.run_manager.end_run()
        self.run_manager.save(os.path.join(self.args.result_dir, 'train_results'))
        logging.info("Training finished.\n")
        self.save_model(optimizer, lr_scheduler)

    def save_model(self, optimizer, lr_scheduler):
        if not os.path.isdir(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        args_path = os.path.join(self.args.model_dir, 'train_args.bin')
        model_path = os.path.join(self.args.model_dir, 'pretrain_model.pt')
        config_path = os.path.join(self.args.model_dir, 'config.json')

        optim_params = {
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }

        try:
            with open(config_path, 'w', encoding='utf-8') as fin:
                json.dump(self.model_config, fin)

            torch.save(optim_params, model_path)
            torch.save(self.args, args_path)
            logging.info(f'Successfully saved model to {self.args.model_dir} !\n')
        except:
            logging.info(f'Model saving failed')

    @classmethod
    def reload_model(cls, train_set, dev_set, test_set, model_dir_path):
        if os.path.exists(model_dir_path):
            args_path = os.path.join(model_dir_path, 'train_args.bin')
            config_path = os.path.join(model_dir_path, 'config.json')
            args = torch.load(args_path)
            logging.info('Reloading model ...\n')
            with open(config_path, 'r', encoding='utf-8') as fin:
                model_config = json.load(fin)

            trainer = cls(train_set, dev_set, test_set, args, model_config,  pretrained_path=model_dir_path)
            return trainer
        else:
            raise FileNotFoundError('Model dir not found!')

    def evaluate(self, mode):
        logging.info('Evaluating ...')
        data_set = self.dev_set if mode == 'dev' else self.test_set
        data_loader = DataLoader(data_set, batch_size=self.args.batch_size, shuffle=True, num_workers=1)
        criterion = nn.CrossEntropyLoss(ignore_index=self.PAD_idx)

        all_intent_logits = None
        all_intent_label_ids = None
        all_slot_logits = None
        all_slot_label_ids = None
        total_loss = 0

        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                src_seqs = batch['src_seq']
                trg_seqs = batch['trg_seq']
                intent_ids = batch['intent_id']
                lens = batch['length']
                if self.args.model_type == 'RNN':
                    lens, idxs = lens.sort(0, descending=True)
                    src_seqs = src_seqs[idxs].to(self.device)
                    trg_seqs = trg_seqs[idxs].to(self.device)
                    lens = lens.to(self.device)
                    intent_ids = intent_ids[idxs].to(self.device)

                # intent_logits: B x I, slot_logits: B x S x T
                intent_logits, slot_logits = self.model(src_seqs, trg_seqs, lens, teacher_forcing_ratio=0.0)
                intent_loss = criterion(intent_logits, intent_ids)
                slot_loss = criterion(slot_logits.reshape(-1, self.slot_size), trg_seqs.reshape(-1))
                loss = intent_loss + slot_loss
                total_loss += loss

                cur_intent_label_ids = intent_ids.detach()
                cur_intent_logits = intent_logits.detach()
                if all_intent_label_ids is None:
                    all_intent_label_ids = cur_intent_label_ids
                    all_intent_logits = cur_intent_logits
                else:
                    all_intent_label_ids = torch.cat((all_intent_label_ids, cur_intent_label_ids), dim=0)
                    all_intent_logits = torch.cat((all_intent_logits, cur_intent_logits), dim=0)

                cur_slot_label_ids = trg_seqs.detach()
                cur_slot_logits = slot_logits.detach()
                if all_slot_label_ids is None:
                    all_slot_label_ids = cur_slot_label_ids
                    all_slot_logits = cur_slot_logits
                else:
                    all_slot_label_ids = torch.cat((all_slot_label_ids, cur_slot_label_ids), dim=0)
                    all_slot_logits = torch.cat((all_slot_logits, cur_slot_logits), dim=0)

        eval_loss = total_loss / len(data_loader)
        eval_metrics = {'eval_loss': eval_loss}

        # Shape: Data_size
        all_intent_label_ids = all_intent_label_ids.cpu().numpy()
        all_intent_pred_ids = torch.argmax(F.softmax(all_intent_logits, dim=1), dim=1).cpu().numpy()

        # Shape: Data_size x Example_length
        all_slot_label_ids = all_slot_label_ids.cpu().numpy()
        all_slot_pred_ids = torch.argmax(F.softmax(all_slot_logits, dim=2), dim=2).cpu().numpy()

        # To Convert from id to tag (in python list)
        all_slot_labels = [[]]*all_slot_label_ids.shape[0]
        all_slot_preds = [[]]*all_slot_label_ids.shape[0]

        for seq_no in range(all_slot_label_ids.shape[0]):
            for slot_no in range(all_slot_label_ids.shape[1]):
                if all_slot_label_ids[seq_no, slot_no] != self.PAD_idx:
                    all_slot_labels[seq_no].append(self.trg_indexer.get_object(all_slot_label_ids[seq_no, slot_no]))
                    all_slot_preds[seq_no].append(self.trg_indexer.get_object(all_slot_pred_ids[seq_no, slot_no]))
                else:
                    break

        eval_metrics['intent_accuracy'] = get_intent_accuracy(all_intent_pred_ids, all_intent_label_ids)
        slot_precision, slot_recall, slot_f1 = get_slot_metrics(all_slot_preds, all_slot_labels)
        eval_metrics['slot_precision'] = slot_precision
        eval_metrics['slot_recall'] = slot_recall
        eval_metrics['slot_f1'] = slot_f1
        eval_metrics['semantic_frame_accuracy'] = get_semantic_frame_accuracy(all_intent_pred_ids, all_intent_label_ids,
                                                                             all_slot_preds, all_slot_labels)

        for metric, score in eval_metrics.items():
            logging.info(f'{metric}: {score}')
        return eval_metrics


