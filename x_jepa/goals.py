from __future__ import annotations

import math

import torch
from torch import nn, cat
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange

from x_mlps_pytorch import MLP
from torch_einops_utils import pad_right_ndim_to, temp_eval

from x_jepa.flow_matching import FlowMatching

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# pos emb

class RandomSinusoidalPosEmb(Module):
    def __init__(self, dim, omega = 1.):
        super().__init__()
        half_dim = dim // 2
        self.register_buffer('weights', torch.randn(half_dim) * omega)

    def forward(self, x):
        x = rearrange(x, '... -> ... 1')
        freqs = x * self.weights * 2 * math.pi
        return cat((freqs.sin(), freqs.cos()), dim = -1)

# goal generator

class GoalGenerator(Module):
    def __init__(
        self,
        dim,
        dim_hidden = None,
        returns_omega = 1.,
        returns_norm_momentum = 0.01,
        net: Module | None = None
    ):
        super().__init__()
        dim_hidden = default(dim_hidden, dim * 4)

        self.norm_returns = nn.BatchNorm1d(1, affine = False, momentum = returns_norm_momentum)

        self.time_emb = RandomSinusoidalPosEmb(dim)
        self.returns_emb = RandomSinusoidalPosEmb(dim, omega = returns_omega)
        self.net = default(net, MLP(dim * 3, dim_hidden, dim_hidden, dim))

    @property
    def freeze_return_stats(self):
        return temp_eval(self.norm_returns)

    def reset_return_stats(self):
        self.norm_returns.reset_running_stats()

    def forward(
        self,
        state,
        time,
        returns
    ):
        returns = rearrange(returns, 'b -> b 1')
        returns = self.norm_returns(returns)
        returns = rearrange(returns, 'b 1 -> b')

        time_embed = self.time_emb(time)
        returns_embed = self.returns_emb(returns)

        return self.net(cat((state, time_embed, returns_embed), dim = -1))
