import torch
from spdnet.utils import *
from spdnet import StiefelParameter, SPDParameter
from spd.parallel_transport import expm


class StiefelMetaOptimizer(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.state = {}

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.grad[torch.isnan(p.grad)] = 0.0
                if isinstance(p, StiefelParameter):
                    trans = orthogonal_projection(p.grad, p)
                    p.grad.fill_(0).add_(trans)
                elif isinstance(p, SPDParameter):
                    riem = p @ ((p.grad + p.grad.transpose(-2, -1)) / 2) @ p
                    self.state[p] = p.clone()
                    p.fill_(0)
                    p.grad.fill_(0).add_(riem)

        loss = self.optimizer.step(closure)

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if isinstance(p, StiefelParameter):
                    trans = retraction(p)
                    p.fill_(0).add_(trans)
                elif isinstance(p, SPDParameter):
                    trans = expm(self.state[p], p)
                    p.fill_(0).add_(trans)

        return loss
