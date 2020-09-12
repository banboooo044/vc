import torch
import numpy as np
import sys
import os
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
from inference import Inferencer
from model import AE, EarlyStopping
from data_utils import get_data_loader
from data_utils import PickleDataset
from utils import *
from functools import reduce
from collections import defaultdict
from tqdm import tqdm
from fastdtw import fastdtw

class Evaluator(object):
    def __init__(self, config, args):
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config
        print(config)

        # args store other information
        self.args = args

        self.inferencer = Inferencer(config, args)

        # get dataloader
        self.get_data_loaders()

        # init the model with config
        self.build_model()
        self.load_model()

    # データをロードする.
    def get_data_loaders(self):
        data_dir = self.args.data_dir

        self.test_dataset = PickleDataset(
            os.path.join(data_dir, f'{self.args.test_set}.pkl'),
            os.path.join(data_dir, self.args.test_index_file),
            segment_size=self.config['data_loader']['segment_size']
        )

        self.test_loader = get_data_loader(self.test_dataset,
                frame_size=self.config['data_loader']['frame_size'],
                batch_size=self.config['data_loader']['batch_size'],
                shuffle=False,
                num_workers=4, drop_last=False
        )

    def load_model(self):
        print(f'Load model from {self.args.load_model_path}')
        self.model.load_state_dict(torch.load(f'{self.args.load_model_path}/{self.args.tag}.ckpt', map_location='cpu'))
        return

    def build_model(self):
        self.model = cc(AE(self.config))
        print(self.model)
        return

    def denormalize(self, x):
        m, s = self.attr['mean'], self.attr['std']
        ret = x * s + m
        return ret

    def ae_step(self, data):
        x = cc(data)
        with torch.set_grad_enabled(False):
            mu, log_sigma, emb, dec = self.model(x)
            criterion = nn.L1Loss()
            loss_rec = criterion(dec, x)
            loss_kl = 0.5 * torch.mean(torch.exp(log_sigma) + mu ** 2 - 1 - log_sigma)

            dec = dec.transpose(1, 2).squeeze(0)
            dec = dec.detach().cpu().numpy()
            dec = self.denormalize(dec)

            meta = {
                'loss_rec': loss_rec.item(),
                'loss_kl': loss_kl.item() }
        return meta

    def eval_rec(self, data):
        self.model.eval()
        loss_rec, loss_kl = 0.0, 0.0, 0.0
        num = 0
        for data in tqdm(self.test_loader):
            meta = self.ae_step(data)
            loss_rec += meta['loss_rec']
            loss_kl += meta['loss_kl']
            num += len(data)
        # add to logger
        loss_rec /= num
        loss_kl /= num
        meta = {'loss_rec': loss_rec, 'loss_kl' : loss_kl }
        print(f'Test Evaluation : loss_rec={loss_rec:.4f}, loss_kl={loss_kl:.4f}')
        return

    def eval_MCD(self, target_path_list):
        conv_mel = inference_form_path_multi_target()
