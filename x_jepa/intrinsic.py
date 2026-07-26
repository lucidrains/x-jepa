from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.nn import Module

from einops import reduce
from torch_einops_utils import safe_cat

def exists(val):
    return val is not None

# Sam Lobel et al. https://arxiv.org/abs/2306.03186

class CoinFlipNetwork(Module):
    def __init__(
        self,
        net: Module,
        dim: int,
        use_prior = True,
        prior_decay = 0.99,
        accepts_actions = False,
        eps = 1e-8
    ):
        super().__init__()
        self.net = net
        self.dim = dim
        self.use_prior = use_prior
        self.prior_decay = prior_decay
        self.accepts_actions = accepts_actions
        self.eps = eps

        self.prior_net = None

        if use_prior:
            self.prior_net = deepcopy(net)
            self.prior_net.requires_grad_(False)

            # running stats for normalizing the prior
            self.register_buffer('prior_mean', torch.zeros(dim))
            self.register_buffer('prior_var', torch.ones(dim))

    def reset(self):
        if not self.use_prior:
            return

        self.prior_mean.zero_()
        self.prior_var.fill_(1.)

    def update_prior_stats(self, prior_out):
        if not self.use_prior or not self.training:
            return

        batch_mean = prior_out.mean(dim = 0)
        batch_var = prior_out.var(dim = 0, unbiased = False)

        self.prior_mean.lerp_(batch_mean, 1 - self.prior_decay)
        self.prior_var.lerp_(batch_var, 1 - self.prior_decay)

    def get_normalized_prior(self, x):
        batch, device = x.shape[0], x.device

        if not self.use_prior:
            return torch.zeros(batch, self.dim, device = device)

        prior_out = self.prior_net(x).detach()
        self.update_prior_stats(prior_out)

        # section 3.5.2: optimistic initialization of bonus
        normalized_prior = (prior_out - self.prior_mean) / self.prior_var.clamp(min = self.eps).sqrt()
        return normalized_prior

    def compute_bonus(self, states, actions = None):
        f_phi = self(states, actions)

        # equation 5
        return reduce(f_phi ** 2, '... d -> ...', 'mean').clamp(min = self.eps).sqrt()

    def compute_loss(self, states, actions = None, coin_flips = None):
        f_phi = self(states, actions)

        if not exists(coin_flips):
            batch, device = f_phi.shape[0], f_phi.device
            coin_flips = self.sample_coin_flips(batch, device = device)

        return F.mse_loss(f_phi, coin_flips)

    @torch.no_grad()
    def sample_coin_flips(self, batch_size, device = None):
        # section 3.1: rademacher distribution (coin flips)
        coin_flips = torch.randint(0, 2, (batch_size, self.dim), device = device)
        return (coin_flips * 2 - 1).float()

    def forward(self, states, actions = None):
        actions = actions if self.accepts_actions else None
        x = safe_cat((states, actions), dim = -1)
        pred = self.net(x)

        if not self.use_prior:
            return pred

        prior = self.get_normalized_prior(x)
        return pred + prior
