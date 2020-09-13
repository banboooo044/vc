import torch
import numpy as np
import sys
import os
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
from model import AE, EarlyStopping
from data_utils import get_data_loader
from data_utils import PickleDataset
from utils import *
from functools import reduce
from collections import defaultdict

class Solver(object):
    def __init__(self, config, args):
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config
        print(config)

        # args store other information
        self.args = args

        # logger to use tensorboard
        os.makedirs(self.args.logdir, exist_ok=True)
        self.logger = Logger(f'{self.args.logdir}/{self.args.tag}')

        # get dataloader
        self.get_data_loaders()

        os.makedirs(self.args.store_model_path, exist_ok=True)
        # init the model with config
        self.build_model()
        self.save_config()

        if args.load_model:
            self.load_model()

        self.EarlyStopping = EarlyStopping(patient=self.config['early_stopping']['patient'], min_delta=self.config['early_stopping']['min_delta'])

    def save_model(self, iteration):
        # save model and discriminator and their optimizer
        if isinstance(self.model, DataParallel):
            torch.save(self.model.module.state_dict(), f'{self.args.store_model_path}/{self.args.tag}.ckpt')
        else:
            torch.save(self.model.state_dict(), f'{self.args.store_model_path}/{self.args.tag}.ckpt')
        torch.save(self.opt.state_dict(), f'{self.args.store_model_path}/{self.args.tag}.opt')

    def save_config(self):
        with open(f'{self.args.store_model_path}/{self.args.tag}.config.yaml', 'w') as f:
            yaml.dump(self.config, f)
        with open(f'{self.args.store_model_path}/{self.args.tag}.args.yaml', 'w') as f:
            yaml.dump(vars(self.args), f)
        return

    def load_model(self):
        print(f'Load model from {self.args.load_model_path}')
        self.model.load_state_dict(torch.load(f'{self.args.load_model_path}/{self.args.tag}.ckpt'))
        self.opt.load_state_dict(torch.load(f'{self.args.load_model_path}/{self.args.tag}.opt'))
        return

    # データをロードする.
    def get_data_loaders(self):
        data_dir = self.args.data_dir
        self.gpu_num = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.train_dataset = PickleDataset(os.path.join(data_dir, f'{self.args.train_set}.pkl'), 
                os.path.join(data_dir, self.args.train_index_file), 
                segment_size=self.config['data_loader']['segment_size'])
        self.train_loader = get_data_loader(self.train_dataset,
                frame_size=self.config['data_loader']['frame_size'],
                batch_size=self.config['data_loader']['batch_size']*self.gpu_num, 
                shuffle=self.config['data_loader']['shuffle'], 
                drop_last=False)
        self.train_iter = infinite_iter(self.train_loader)

        if self.args.use_eval_set:
            self.eval_dataset = PickleDataset(
                os.path.join(data_dir, f'{self.args.eval_set}.pkl'),
                os.path.join(data_dir, self.args.eval_index_file),
                segment_size=self.config['data_loader']['segment_size'])

            self.eval_loader = get_data_loader(self.eval_dataset,
                frame_size=self.config['data_loader']['frame_size'],
                batch_size=self.config['data_loader']['batch_size']*self.gpu_num, 
                shuffle=self.config['data_loader']['shuffle'],
                drop_last=False)
            self.eval_iter = infinite_iter(self.eval_loader)

        if self.args.use_test_set:
            self.test_dataset = PickleDataset(
                os.path.join(data_dir, f'{self.args.test_set}.pkl'),
                os.path.join(data_dir, self.args.test_index_file),
                segment_size=self.config['data_loader']['segment_size']
            )

            self.test_loader = get_data_loader(self.test_dataset,
                frame_size=self.config['data_loader']['frame_size'],
                batch_size=self.config['data_loader']['batch_size'],
                shuffle=False,
                drop_last=False
            )
            self.test_iter = infinite_iter(self.test_loader)

        return

    def build_model(self):
        # create model, discriminator, optimizers
        self.model = cc_model(AE(self.config))
        print(self.model)
        optimizer = self.config['optimizer']
        self.opt = torch.optim.Adam(self.model.parameters(), 
                lr=optimizer['lr'], betas=(optimizer['beta1'], optimizer['beta2']), 
                amsgrad=optimizer['amsgrad'], weight_decay=optimizer['weight_decay'])
        print(self.opt)
        return

    def ae_step(self, data, lambda_kl, phase):
        x = cc_data(data)
        self.opt.zero_grad()
        with torch.set_grad_enabled(phase=='train'):
            mu, log_sigma, emb, dec = self.model(x)
            criterion = nn.L1Loss()
            loss_rec = criterion(dec, x)
            loss_kl = 0.5 * torch.mean(torch.exp(log_sigma) + mu ** 2 - 1 - log_sigma)
            loss = self.config['lambda']['lambda_rec'] * loss_rec + \
                lambda_kl * loss_kl
            if phase == 'train':
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                        max_norm=self.config['optimizer']['grad_norm'])
                self.opt.step()
                meta = {'loss_rec': loss_rec.item(),
                        'loss_kl': loss_kl.item(),
                        'grad_norm': grad_norm}
            else:
                meta = {'loss_rec': loss_rec.item(),
                        'loss_kl': loss_kl.item() }

        return meta

    def train(self, n_iterations):
        loss_eval = 0.0
        epoch = 1
        phases = ['train']
        if self.args.use_eval_set:
            phases.append('eval')
        if self.args.use_test_set:
            phases.append('test')
        try:
            for iteration in range(n_iterations):
                if iteration >= (self.config['annealing_iters'] / self.gpu_num):
                    lambda_kl = self.config['lambda']['lambda_kl']
                else:
                    lambda_kl = self.config['lambda']['lambda_kl'] * (iteration + 1) * self.gpu_num / self.config['annealing_iters']

                for phase in phases:
                    if phase == 'train':
                        self.model.train()
                        data, _ = next(self.train_iter)
                    elif phase == 'eval':
                        self.model.eval()
                        data, flg = next(self.eval_iter)
                        if iteration > 0 and flg:
                            print(f"eval epoch[{epoch}] : eval loss : {loss_eval:.4f}")
                            print()
                            loss_eval = 0.0
                            epoch+=1
                            flg = self.EarlyStopping.is_stop(loss_eval)
                            if flg:
                                self.save_model(iteration=iteration)
                                return
                    elif phase == 'test':
                        self.model.eval()
                        data, _ = next(self.test_iter)

                    meta = self.ae_step(data, lambda_kl, phase)
                    # add to logger
                    loss_rec = meta['loss_rec']
                    loss_kl = meta['loss_kl']

                    if phase == 'eval':
                        loss_eval += (loss_rec + loss_kl)

                    if iteration % self.args.summary_steps == 0:
                        print(f'{format(phase, ">5")} :: AE:[{iteration + 1}/{n_iterations}], loss_rec={loss_rec:.2f}, '
                            f'loss_kl={loss_kl:.2f}, lambda={lambda_kl:.1e}     ')
                        self.logger.scalars_summary(f'{self.args.tag}/ae_{phase}', meta, iteration)

                    print(f'{format(phase, ">5")} :: AE:[{iteration + 1}/{n_iterations}], loss_rec={loss_rec:.2f}, '
                            f'loss_kl={loss_kl:.2f}, lambda={lambda_kl:.1e}     ', end='\r')

                    if phase=='train' and ((iteration + 1) % self.args.save_steps == 0 or iteration + 1 == n_iterations):
                        self.save_model(iteration=iteration)

        except KeyboardInterrupt:
            self.save_model(iteration=iteration)
            self.logger.scalars_summary(f'{self.args.tag}/ae_{phase}', meta, iteration)
        return

