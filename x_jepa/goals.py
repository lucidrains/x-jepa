from __future__ import annotations

import math

import torch
from torch import nn, cat
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, reduce

from x_mlps_pytorch import MLP
from torch_einops_utils import pad_right_ndim_to, temp_eval

from x_jepa.flow_matching import FlowMatching

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

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

# metric residual network
# https://arxiv.org/abs/2208.08133

class MetricResidualNetwork(Module):
    def __init__(
        self,
        *,
        sym_network: Module,
        asym_network: Module,
        distance_groups = 8,
        margin = 10.
    ):
        super().__init__()

        # the two network backbones, producing inputs for symmetric and asymmetric half of quasimetric distance

        self.sym_network = sym_network
        self.asym_network = asym_network

        # distance related

        self.distance_groups = distance_groups
        self.margin = margin
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    @staticmethod
    def quasimetric_distance(x, y, asym_x, asym_y):
        assert x.shape[-1] == y.shape[-1] == asym_x.shape[-1] == asym_y.shape[-1]

        sym = (x - y).norm(p = 2, dim = -1)
        asym = (asym_x - asym_y).relu().amax(dim = -1)

        return sym + asym

    # loss from "Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning"
    # https://arxiv.org/abs/2304.01203

    def time_contrastive_loss(
        self,
        state_left,
        state_right,
        t1,
        t2
    ):
        dists = self(state_left, state_right)

        dt = t2 - t1
        is_forward = dt >= 0
        is_backward = ~is_forward

        # forward transitions (t2 >= t1): distance should match exact time elapsed
        forward_loss = F.mse_loss(dists[is_forward], dt[is_forward].float()) if is_forward.any() else self.zero

        # backward transitions (t2 < t1): distance should be large
        backward_loss = F.relu(self.margin - dists[is_backward]).mean() if is_backward.any() else self.zero

        # negative pairs from different trajectories
        neg_dists = self(state_left, torch.roll(state_right, 1, dims = 0))
        neg_loss = F.relu(self.margin - neg_dists).mean()

        return forward_loss + backward_loss + neg_loss

    def forward(
        self,
        encoded_left,
        encoded_right,
        reduce_groups = True
    ):
        encoded = [encoded_left, encoded_right]

        sym_x, sym_y = [self.sym_network(t) for t in encoded]
        asym_x, asym_y = [self.asym_network(t) for t in encoded]

        dim_embed = sym_x.shape[-1]
        assert divisible_by(dim_embed, self.distance_groups)

        sym_x, sym_y, asym_x, asym_y = (rearrange(t, '... (g d) -> ... g d', g = self.distance_groups) for t in (sym_x, sym_y, asym_x, asym_y))

        distance = self.quasimetric_distance(sym_x, sym_y, asym_x, asym_y)

        if not reduce_groups:
            return distance

        return reduce(distance, '... g -> ...', 'mean')
