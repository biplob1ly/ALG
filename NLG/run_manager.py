import time
# import torchvision
import pandas as pd
import json
import torch
# from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
# from IPython.display import display, clear_output
from utils import count_model_params, elapsed_time
import logging
import os
import math


class RunManager:
    def __init__(self, result_dir):
        self.result_dir = result_dir
        self.epoch_count = 0
        self.train_loss = None
        self.dev_loss = None
        self.epoch_start_time = None
        self.epoch_end_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.model = None
        self.loader = None

    def begin_run(self, hyper_params, model, loader):
        self.run_start_time = time.time()
        self.run_params = hyper_params
        self.run_count += 1

        self.model = model
        self.loader = loader
        logging.info("Training starting")
        total_param_count = count_model_params(self.model)
        logging.info(f'Total trainable params: {total_param_count}')

    def end_run(self):
        self.epoch_count = 0
        self.save(os.path.join(self.result_dir, 'train_results'))

    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.train_loss = 0
        self.dev_loss = 0

    def end_epoch(self):
        self.epoch_end_time = time.time()

    def track_loss(self, train_loss, dev_loss):
        self.train_loss = train_loss
        self.dev_loss = dev_loss
        epoch_mins, epoch_secs = elapsed_time(self.epoch_start_time, self.epoch_end_time)
        run_mins, run_secs = elapsed_time(self.run_start_time, self.epoch_end_time)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results["train_loss"] = train_loss
        results["dev_loss"] = dev_loss
        results["train_ppl"] = math.exp(train_loss)
        results["dev_ppl"] = math.exp(dev_loss)
        results["epoch_duration"] = f'{epoch_mins}m {epoch_secs}s'
        results["run duration"] = f'{run_mins}m {run_secs}s'

        for k, v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)

    def save(self, fileName):
        pd.DataFrame.from_dict(self.run_data, orient='columns', ).to_csv(f'{fileName}.csv')
        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)
