from __future__ import annotations

import math

import torch
from torch import nn, cat
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange

from x_mlps_pytorch import MLP
from torch_einops_utils import pad_right_ndim_to, temp_eval

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

# flow matching

class FlowMatching(Module):
    def __init__(
        self,
        model: Module,
        data_shape = None,
        loss_fn = F.mse_loss,
        noise_std = 1.
    ):
        super().__init__()
        self.model = model
        self.data_shape = data_shape

        self.loss_fn = loss_fn
        self.noise_std = noise_std

    @torch.no_grad()
    def sample(
        self,
        steps = 16,
        batch_size = 1,
        data_shape = None,
        return_noise = False,
        noise = None,
        **kwargs
    ):

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'shape of the data must be passed in, or set at init or during training'
        device = next(self.model.parameters()).device

        init = default(noise, torch.randn((batch_size, *data_shape), device = device) * self.noise_std)
        state = init

        times = torch.linspace(0., 1., steps + 1, device = device)
        times = times[:-1]

        delta = 1. / steps

        for time in times:
            time = time.expand(batch_size)
            pred_flow = self.model(state, time = time, **kwargs)
            state = state + delta * pred_flow

        out = state

        if not return_noise:
            return out

        return out, init

    def forward(
        self,
        data,
        noise = None,
        times = None,
        loss_reduction = 'mean',
        **kwargs
    ):
        shape, ndim = data.shape, data.ndim
        self.data_shape = default(self.data_shape, shape[1:])
        batch, device = shape[0], data.device

        times = default(times, torch.rand(batch, device = device))

        noise = default(noise, torch.randn_like(data) * self.noise_std)
        flow = data - noise

        padded_times = pad_right_ndim_to(times, ndim)
        noised_data = noise.lerp(data, padded_times)

        pred_flow = self.model(noised_data, time = times, **kwargs)

        return self.loss_fn(flow, pred_flow, reduction = loss_reduction)

# goal generator

class GoalGenerator(Module):
    def __init__(
        self,
        dim,
        dim_hidden = None,
        returns_omega = 1.,
        net: Module | None = None
    ):
        super().__init__()
        dim_hidden = default(dim_hidden, dim * 4)

        self.norm_returns = nn.BatchNorm1d(1, affine = False)

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
