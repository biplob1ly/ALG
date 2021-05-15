import torch
import logging
from torch import nn
from torch.utils.data import DataLoader
from run_manager import RunManager
import torch.nn.functional as F
from models.rnn_model import EncoderDecoder
from utils import *
import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
import os
import json


def pad_collate(batch):
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    src_seqs = [x[0] for x in sorted_batch]
    src_lens = torch.tensor([len(x) for x in src_seqs], dtype=torch.long)
    padded_src_seqs = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True)
    trg_seqs = torch.cat([x[1].unsqueeze(0) for x in sorted_batch], dim=0)
    return {'src_seq': padded_src_seqs, 'src_len': src_lens, 'trg_seq': trg_seqs}


class Trainer:
    def __init__(self, train_set, dev_set, test_set, args, featurizer,
                 model_config, pretrained_path=None):
        self.train_set = train_set
        self.dev_set = dev_set
        self.test_set = test_set
        self.args = args
        self.featurizer = featurizer
        self.src_pad_idx = featurizer.src_indexer.index_of(featurizer.PAD)
        self.trg_sos_idx = featurizer.trg_indexer.index_of(featurizer.SOS)
        self.trg_eos_idx = featurizer.trg_indexer.index_of(featurizer.EOS)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.model_config = model_config
        self.src_vocab_size = model_config['src_vocab_size']
        self.trg_vocab_size = model_config['trg_vocab_size']
        self.embed_size = model_config['embed_size']
        self.hidden_size = model_config['hidden_size']

        if args.model_type == 'RNN':
            self.model = EncoderDecoder(self.src_vocab_size, self.embed_size, self.hidden_size,
                                             self.trg_vocab_size, self.src_pad_idx, self.args.max_trg_len)
            if not pretrained_path:
                self.model.init_weight()
            else:
                # loading pretrained weights
                params_path = os.path.join(pretrained_path, 'pretrain_model.pt')
                model_params = torch.load(params_path)['model_state_dict']
                self.model.load_state_dict(model_params)
        self.model.to(self.device)
        self.run_manager = RunManager(self.args.result_dir)

    def train(self, hyper_params):
        train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True, collate_fn=pad_collate, num_workers=1)
        dev_loader = DataLoader(self.dev_set, batch_size=self.args.batch_size, shuffle=True, collate_fn=pad_collate, num_workers=1)
        criterion = nn.CrossEntropyLoss(ignore_index=self.src_pad_idx)

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
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, self.args.warm_steps, total_steps)

        best_dev_loss = float('inf')
        self.run_manager.begin_run(hyper_params, self.model, train_loader)
        for epoch in tqdm.trange(hyper_params.num_epochs, desc='Epoch'):
            self.run_manager.begin_epoch()
            train_loss = self.learn(train_loader, criterion, optimizer, lr_scheduler)
            self.run_manager.end_epoch()
            dev_loss = self.validate(dev_loader, criterion)
            self.run_manager.track_loss(train_loss, dev_loss)
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                self.save_model(optimizer, lr_scheduler)

        self.run_manager.end_run()
        logging.info("Training finished.\n")

    def learn(self, data_loader, criterion, optimizer, lr_scheduler):
        epoch_loss = 0
        self.model.train()
        for batch in data_loader:
            src_seqs = batch['src_seq'].to(self.device)
            src_lens = batch['src_len'].to(self.device)
            trg_seqs = batch['trg_seq'].to(self.device)

            trg_logits = self.model(src_seqs, src_lens, trg_seqs)
            loss = criterion(trg_logits[1:].reshape(-1, self.trg_vocab_size), trg_seqs[1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(data_loader)
        return train_loss

    def validate(self, data_loader, criterion):
        epoch_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                src_seqs = batch['src_seq'].to(self.device)
                src_lens = batch['src_len'].to(self.device)
                trg_seqs = batch['trg_seq'].to(self.device)

                # trg_logits: B x S x T
                trg_logits = self.model(src_seqs, src_lens, trg_seqs)
                loss = criterion(trg_logits[1:].reshape(-1, self.trg_vocab_size), trg_seqs[1:].reshape(-1))
                epoch_loss += loss.item()

        dev_loss = epoch_loss / len(data_loader)
        return dev_loss

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
    def reload_model(cls, train_set, dev_set, test_set, featurizer, model_dir_path):
        if os.path.exists(model_dir_path):
            args_path = os.path.join(model_dir_path, 'train_args.bin')
            config_path = os.path.join(model_dir_path, 'config.json')
            args = torch.load(args_path)
            logging.info('Reloading model ...\n')
            with open(config_path, 'r', encoding='utf-8') as fin:
                model_config = json.load(fin)

            trainer = cls(train_set, dev_set, test_set, args, featurizer, model_config,  pretrained_path=model_dir_path)
            return trainer
        else:
            raise FileNotFoundError('Model dir not found!')

    def decode(self, src_sent):
        src_idxs = self.featurizer.featurize_src(src_sent)
        src_seqs = torch.tensor([src_idxs], dtype=torch.long).to(self.device)
        src_lens = torch.tensor([len(src_idxs)], dtype=torch.long).to(self.device)
        # attn_mask: B x S
        attn_mask = (src_seqs != self.model.src_pad_idx)

        self.model.eval()
        with torch.no_grad():
            # encoder_outputs: B x S x 2H, encoder_hc: 1 x B x H
            encoder_outputs, encoder_hc = self.model.encoder(src_seqs, src_lens)

        batch_size = src_seqs.size(0)
        decoder_context = torch.zeros(batch_size, 1, 2 * self.model.hidden_size).to(src_seqs.device)
        decoder_hc = encoder_hc

        trg_idxs = [self.trg_sos_idx]
        attentions = torch.zeros(self.model.max_trg_len, 1, len(src_idxs)).to(self.device)
        for di in range(1, self.model.max_trg_len):
            trg_seqs = torch.tensor([[trg_idxs[-1]]], dtype=torch.long).to(self.device)
            with torch.no_grad():
                # decoder_output: B x T
                decoder_output, decoder_context, decoder_hc, decoder_attention = self.model.decoder(trg_seqs,
                                                                                          decoder_context,
                                                                                          decoder_hc,
                                                                                          encoder_outputs,
                                                                                          attn_mask)
            attentions[di] = decoder_attention
            # pred_idxs: (B,)
            pred_idxs = decoder_output.argmax(dim=1)
            trg_idxs.append(pred_idxs.item())
            if pred_idxs == self.trg_eos_idx:
                break
        trg_tokens = [self.featurizer.trg_indexer.get_object(trg_idx) for trg_idx in trg_idxs]
        return trg_tokens[1:], attentions[:len(trg_tokens)-1]

    def evaluate(self, split):
        if split == 'test':
            featurized_dataset = self.test_set
        elif split == 'dev':
            featurized_dataset = self.dev_set
        else:
            featurized_dataset = self.train_set
        eval_loader = DataLoader(featurized_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=pad_collate, num_workers=1)
        criterion = nn.CrossEntropyLoss(ignore_index=self.src_pad_idx)
        eval_loss = self.validate(eval_loader, criterion)
        eval_metrics = {'eval_loss': eval_loss}

        trgs = []
        preds = []
        dataset_dir = os.path.join(self.args.data_dir, self.args.dataset_name)
        src_file = os.path.join(dataset_dir, split, 'query.txt')
        trg_file = os.path.join(dataset_dir, split, 'response.txt')

        with open(src_file, 'r') as src_fin, open(trg_file, 'r') as trg_fin:
            for src_line, trg_line in zip(src_fin, trg_fin):
                trg_sent = tokenize(trg_line.strip())
                trg_tokens = list(filter(bool, trg_sent.split()))
                pred_trg_tokens, _ = self.decode(src_line)

                # cut off <eos> token
                pred_trg_tokens = pred_trg_tokens[:-1]

                trgs.append([trg_tokens])
                preds.append(pred_trg_tokens)
                break

        eval_metrics.update(get_eval_metrics(trgs, preds, self.args.result_dir))
        logging.info(f'Evaluations scores for {split} dataset')
        for metric, score in eval_metrics.items():
            logging.info(f'{metric}: {score}')
        return eval_metrics
