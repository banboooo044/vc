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

class Evaluator(object):
    def __init__(self, config, args):
        # config store the value of hyperparameters, turn to attr by AttrDict
        inferencer = Inferencer(config=config, args=args)

        self.config = config
        print(config)

        # args store other information
        self.args = args

        # get dataloader
        self.get_data_loaders()

        # init the model with config
        self.build_model()
        self.load_model()

    def eval(self):
        dec = self.model.inference(x, x_cond)
        dec = dec.transpose(1, 2).squeeze(0)
        dec = dec.detach().cpu().numpy()
        dec = self.denormalize(dec)

        loss_rec, loss_kl = 0.0, 0.0, 0.0
        num = 0
        for data, label in tqdm(self.test_loader):
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

