"""universal successor features over the world model's final contextual state

reference:
  Zero-Shot Control with Gaussian Representations and Test-Time Adjustment.
  Anonymous ICLR 2027 submission. https://openreview.net/forum?id=WeUNn9rlIi

  psi(z, a) = z + gamma * E[psi(z', a')]   # phi is the identity on z
  Q(z, a, w) = psi(z, a) . w               # UVFA / successor-feature readout
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Module
import einx
from ema_pytorch import EMA
from einops import rearrange, reduce, repeat
from torch_einops_utils import pad_right_at_dim_to
from x_mlps_pytorch import MLP

from x_jepa.utils import exists, default, masked_mean
from x_jepa.value_networks import resolve_discount, broadcast_cond


class UniversalSuccessorFeatures(Module):
    def __init__(
        self,
        dim_state_latent,
        dim_action,
        hidden_dim = None,
        ema_beta = 0.99,
        discount_factor = 0.99,
        frac_gradient = 0.,
        reg: Module | None = None,
        reg_weight = 0.
    ):
        super().__init__()

        hidden_dim = default(hidden_dim, dim_state_latent * 2)

        self.dim_state_latent = dim_state_latent
        self.dim_action = dim_action
        self.discount_factor = discount_factor
        self.frac_gradient = frac_gradient
        self.reg = reg
        self.reg_weight = reg_weight

        self.net = MLP(
            dim_state_latent + dim_action,
            hidden_dim,
            hidden_dim,
            dim_state_latent
        )
        self.ema_net = EMA(self.net, beta = ema_beta, update_after_step = 0, update_every = 1)

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    @property
    def feature_dim(self):
        return self.dim_state_latent

    @property
    def device(self):
        return self.zero.device

    def _state_for_grad(self, state_latents):
        if self.frac_gradient <= 0.:
            return state_latents.detach()

        if self.frac_gradient >= 1.:
            return state_latents

        return state_latents.detach().lerp(state_latents, self.frac_gradient)

    def successor_features(self, state_latents, actions, ema = False):
        net = self.ema_net.ema_model if ema else self.net
        return net((state_latents, actions))

    def forward(self, state_latents, actions):
        return self.successor_features(state_latents, actions)

    def q_from_psi(self, psi, task):
        if task.ndim == 1:
            weighted = einx.multiply('b ... d, d -> b ... d', psi, task)
        elif task.ndim == 2:
            weighted = einx.multiply('b ... d, b d -> b ... d', psi, task)
        else:
            weighted = einx.multiply('b ... d, b ... d -> b ... d', psi, task)

        return reduce(weighted, 'b ... d -> b ...', 'sum')

    def q(self, state_latents, actions, task, ema = False):
        psi = self.successor_features(state_latents, actions, ema = ema)
        return self.q_from_psi(psi, task)

    def goal_q(self, state_latents, actions, goals, ema = False):
        # w is the task/goal vector; psi stays goal-agnostic (one behavior policy)
        return self.q(state_latents, actions, goals, ema = ema)

    def score_actions(self, state_latent, actions, task, ema = False):
        # actions: (b, p, d) or (b, p, h, d) - score the first action of each candidate
        if state_latent.ndim == 1:
            state_latent = rearrange(state_latent, 'd -> 1 d')

        if actions.ndim == 4:
            actions = actions[:, :, 0]

        state_latents = repeat(state_latent, 'b d -> b p d', p = actions.shape[1])
        return self.q(state_latents, actions, task, ema = ema)

    def compute_loss(
        self,
        state_latents,
        actions,
        next_state_latents = None,
        next_actions = None,
        discount = None,
        mask = None,
        dones = None
    ):
        if state_latents.shape[1] < 2 or actions.shape[1] < 1:
            return self.zero

        m = min(state_latents.shape[1] - 1, actions.shape[1])
        if m < 1:
            return self.zero

        states = state_latents[:, :m]
        next_states = default(next_state_latents, state_latents[:, 1:m + 1])[:, :m]
        actions = actions[:, :m]

        if exists(next_actions):
            next_actions = pad_right_at_dim_to(next_actions[:, :m], m, dim = 1)
        else:
            next_actions = pad_right_at_dim_to(actions[:, 1:m + 1], m, dim = 1)

        if exists(mask):
            mask = pad_right_at_dim_to(mask.bool(), m + 1, dim = 1)
            valid = mask[:, :m] & mask[:, 1:m + 1]
        else:
            valid = torch.ones((states.shape[0], m), dtype = torch.bool, device = states.device)

        if exists(dones):
            dones = pad_right_at_dim_to(dones.bool(), m, dim = 1)
        else:
            dones = None

        bootstrap = valid if not exists(dones) else valid & ~dones

        has_next_action = torch.arange(m, device = states.device) < (m - 1)
        has_next_action = repeat(has_next_action, 'm -> b m', b = states.shape[0])

        pred = self.successor_features(self._state_for_grad(states), actions)

        with torch.no_grad():
            next_feature = einx.where(
                'b m, b m d, b m d -> b m d',
                has_next_action,
                self.successor_features(next_states, next_actions, ema = True),
                next_states
            )

            gamma = resolve_discount(
                default(discount, self.discount_factor),
                batch = states.shape[0],
                device = states.device,
                dtype = states.dtype,
                sample = True
            )

            gamma = torch.as_tensor(gamma, device = states.device, dtype = states.dtype)

            if gamma.ndim == 2:
                gamma = gamma[:, :m]

            gamma = broadcast_cond(gamma, pred)

            future = einx.multiply('b m d, b m, b m -> b m d', next_feature, gamma, bootstrap.float())
            target = einx.add('b m d, b m d -> b m d', states, future)

        # scale by (1 - gamma) so the loss matches the latent-feature scale, not 1 / (1 - gamma)

        loss_scale = (1. - gamma).clamp_min(1e-3)
        raw_loss = F.smooth_l1_loss(pred, target, reduction = 'none')
        raw_loss = einx.multiply('b m d, b m -> b m d', raw_loss, loss_scale)
        loss = masked_mean(raw_loss, valid, average_mode = 'timestep')

        if exists(self.reg) and self.reg_weight > 0. and bool(valid.any()):
            loss = loss + self.reg_weight * self.reg(self._state_for_grad(states)[valid])

        return loss

    def loss(self, *args, **kwargs):
        return self.compute_loss(*args, **kwargs)

    def update(self):
        self.ema_net.update()
