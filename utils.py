import torch 
import numpy as np
from tensorboardX import SummaryWriter
import editdistance
import torch.nn as nn
import torch.nn.init as init

def cc_data(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    return data

def cc_model(net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        device_ids = list(range(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net, device_ids=device_ids) # make parallel
        torch.backends.cudnn.benchmark = True
    net = net.to(device)
    return net


class Logger(object):
    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def scalars_summary(self, tag, dictionary, step):
        self.writer.add_scalars(tag, dictionary, step)

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)

    def audio_summary(self, tag, value, step, sr):
        writer.add_audio(tag, value, step, sample_rate=sr)

def infinite_iter(iterable):
    it = iter(iterable)
    reset_flg = True
    while True:
        try:
            ret = next(it)
            yield ret, reset_flg
            reset_flg = False
        except StopIteration:
            it = iter(iterable)
            reset_flg = True

