import torch
from torch.utils.data import Dataset
import os 
import pickle 
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

class CollateFn(object):
    def __init__(self, frame_size):
        self.frame_size = frame_size

    def make_frames(self, tensor):
        out = tensor.view(tensor.size(0), tensor.size(1) // self.frame_size, self.frame_size * tensor.size(2))
        out = out.transpose(1, 2)
        return out 

    def __call__(self, l):
        data_tensor = torch.from_numpy(np.array(l))
        segment = self.make_frames(data_tensor)
        return segment

def get_data_loader(dataset, batch_size, frame_size, shuffle=True, num_workers=4, drop_last=False):
    _collate_fn = CollateFn(frame_size=frame_size) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
            num_workers=num_workers, collate_fn=_collate_fn, pin_memory=True)
    return dataloader

class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.utt_ids = list(self.data.keys())

    def __getitem__(self, ind):
        utt_id = self.utt_ids[ind]
        ret = self.data[utt_id].transpose()
        return ret

    def __len__(self):
        return len(self.utt_ids)

class PickleDataset(Dataset):
    def __init__(self, pickle_path, sample_index_path, segment_size):
        with open(pickle_path, 'rb') as f:
            self.data = pickle.load(f)
        with open(sample_index_path, 'r') as f:
            self.indexes = json.load(f)
        self.segment_size = segment_size

    def __getitem__(self, ind):
        utt_id, t = self.indexes[ind]
        segment = self.data[utt_id][t:t + self.segment_size]
        return segment

    def __len__(self):
        return len(self.indexes)

class PickleDatasetParallel(Dataset):
    def __init__(self, source_pickle_path, sample_index_path, segment_size):
        with open(source_pickle_path, 'rb') as f:
            self.data = pickle.load(f)
        with open(sample_index_path, 'r') as f:
            self.indexes = json.load(f)
        self.segment_size = segment_size

    def _read_parallel_data(self):
        pass

def _sqrt(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.sqrt(x) if isnumpy else math.sqrt(x) if isscalar else x.sqrt()


def _exp(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.exp(x) if isnumpy else math.exp(x) if isscalar else x.exp()


def _sum(x):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return np.sum(x)
    return float(x.sum())

def melcd(X, Y, lengths=None):
    """Mel-cepstrum distortion (MCD).

    The function computes MCD for time-aligned mel-cepstrum sequences.

    Args:
        X (ndarray): Input mel-cepstrum, shape can be either of
          (``D``,), (``T x D``) or (``B x T x D``). Both Numpy and torch arrays
          are supported.
        Y (ndarray): Target mel-cepstrum, shape can be either of
          (``D``,), (``T x D``) or (``B x T x D``). Both Numpy and torch arrays
          are supported.
        lengths (list): Lengths of padded inputs. This should only be specified
          if you give mini-batch inputs.

    Returns:
        float: Mean mel-cepstrum distortion in dB.

    .. note::

        The function doesn't check if inputs are actually mel-cepstrum.
    """
    # summing against feature axis, and then take mean against time axis
    # Eq. (1a)
    # https://www.cs.cmu.edu/~awb/papers/sltu2008/kominek_black.sltu_2008.pdf
    logdb_const = 10.0 / np.log(10.0) * np.sqrt(2.0)
    if lengths is None:
        z = X - Y
        r = _sqrt((z * z).sum(-1))
        if not np.isscalar(r):
            r = r.mean()
        return logdb_const * float(r)

    # Case for 1-dim features.
    if len(X.shape) == 2:
        # Add feature axis
        X, Y = X[:, :, None], Y[:, :, None]

    s = 0.0
    T = _sum(lengths)
    for x, y, length in zip(X, Y, lengths):
        x, y = x[:length], y[:length]
        z = x - y
        s += _sqrt((z * z).sum(-1)).sum()

    return logdb_const * float(s) / float(T)