import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import re

import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
import torch


def fc2vector(fc, offset=-1):
    index = torch.tril_indices(fc.shape[-2], fc.shape[-1], offset=offset)
    fc_vector = fc[..., index[0], index[1]]
    return fc_vector


def corrcoef(X):
    avg = X.mean(-1)
    X = X - avg[..., None]
    X_T = X.swapaxes(-2, -1)
    c = X @ X_T
    d = c.diagonal(0, -2, -1)
    stddev = np.sqrt(d)
    c /= stddev[..., None]
    c /= stddev[..., None, :]
    np.clip(c, -1, 1, out=c)
    return c


def sliding_window_corrcoef(X, window, padding=True):
    if padding:
        left = (window - 1) // 2
        right = window - 1 - left
        X = np.concatenate((X[..., :left], X, X[..., -right:]), axis=-1)
    X_window = sliding_window_view(X, (X.shape[-2], window), (-2, -1)).squeeze()
    return corrcoef(X_window)


def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def cluster_score(labels_true, labels_pred):
    purity = purity_score(labels_true, labels_pred)
    acc = accuracy_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    return purity, acc, nmi


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def plot_epochs(fname, X, epochs, xlabel, ylabel, legends, max=True):
    plt.figure()
    for i, x in enumerate(X):
        val = np.max(x) if max else np.min(x)
        idx = np.argmax(x) + 1 if max else np.argmin(x) + 1
        plt.plot(epochs, x, label=legends[i])
        plt.plot(idx, val, 'ko')
        plt.annotate(f'({idx},{val:.4f})', xy=(idx, val), xytext=(idx, val))

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname)
    # plt.show()
    plt.close()
