import torch 
import numpy as np
from tensorboardX import SummaryWriter
import editdistance
import torch.nn as nn
import torch.nn.init as init

def cc(net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    print(device)
    if device == 'cuda':
        device_ids = list(range(torch.cuda.device_count()))
        print(device_ids)
        net = torch.nn.DataParallel(net, device_ids=device_ids) # make parallel
        cudnn.benchmark = True
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
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(iterable)
