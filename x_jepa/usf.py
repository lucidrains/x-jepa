# universal successor features over the world model's final contextual state
#
# reference:
#   Zero-Shot Control with Gaussian Representations and Test-Time Adjustment.
#   Anonymous ICLR 2027 submission. https://openreview.net/forum?id=WeUNn9rlIi
#
#   psi(z, a, gamma) = z + gamma * E[psi(z', a', gamma)]
#   Q(z, a, w, gamma) = psi(z, a, gamma) . w

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Module

import einx
from ema_pytorch import EMA
from einops import rearrange, repeat
from torch_einops_utils import pad_right_at_dim_to
from x_mlps_pytorch import MLP

from x_jepa.utils import exists, default, masked_mean
from x_jepa.value_networks import DiscountEmbedder, resolve_discount, broadcast_cond

# classes

class UniversalSuccessorFeatures(Module):
    def __init__(
        self,
        dim_state_latent,
        dim_action,
        hidden_dim = None,
        ema_beta = 0.99,
        discount_factor = 0.99,
        discount_embedder: Module | None = None,
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

        self.discount_embedder = default(discount_embedder, DiscountEmbedder(dim_state_latent) if exists(discount_factor) else None)
        self.ema_discount_embedder = EMA(self.discount_embedder, beta = ema_beta, update_after_step = 0, update_every = 1) if exists(self.discount_embedder) else None

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

    def successor_features(self, state_latents, actions, discount = None, ema = False):
        net = self.ema_net.ema_model if ema else self.net
        discount = default(discount, self.discount_factor)

        if exists(self.discount_embedder) and exists(discount):
            embedder = self.ema_discount_embedder.ema_model if ema else self.discount_embedder
            state_latents = state_latents + embedder(broadcast_cond(discount, state_latents))

        return net((state_latents, actions))

    forward = successor_features

    def q(self, state_latents, actions, task, discount = None, ema = False):
        psi = self.successor_features(state_latents, actions, discount = discount, ema = ema)

        if task.ndim == 2 and task.shape[0] != psi.shape[0]:
            return einx.dot('b ... d, k d -> b ... k', psi, task)

        task_einops = 'd' if task.ndim == 1 else ('b d' if task.ndim == 2 else 'b ... d')
        return einx.dot(f'b ... d, {task_einops} -> b ...', psi, task)

    goal_q = q

    def score_actions(self, state_latent, actions, task, discount = None, ema = False):
        if state_latent.ndim == 1:
            state_latent = rearrange(state_latent, 'd -> 1 d')

        if actions.ndim == 4:
            actions = actions[:, :, 0]

        state_latents = repeat(state_latent, 'b d -> b p d', p = actions.shape[1])
        return self.q(state_latents, actions, task, discount = discount, ema = ema)

    @torch.no_grad()
    def infer_task(
        self,
        state_latents,
        rewards,
        actions = None,
        discount = None,
        mask = None,
        ridge_lambda = 1e-2,
        prior_task = None,
        ema = True
    ):
        features = self.successor_features(state_latents, actions, discount = discount, ema = ema) if exists(actions) else state_latents

        has_single_reward = rewards.ndim < features.ndim
        if has_single_reward:
            rewards = rearrange(rewards, '... -> ... 1')

        if exists(mask):
            features = features[mask]
            rewards = rewards[mask]
        else:
            features = rearrange(features, '... d -> (...) d')
            rewards = rearrange(rewards, '... k -> (...) k')

        num_samples, dim = features.shape

        if not exists(ridge_lambda):
            task = einx.dot('n d, n k -> d k', features, rewards) / max(num_samples, 1)
        else:
            cov = einx.dot('n d1, n d2 -> d1 d2', features, features) / max(num_samples, 1)
            rhs = einx.dot('n d, n k -> d k', features, rewards) / max(num_samples, 1)

            if exists(prior_task):
                prior = rearrange(prior_task, 'd -> d 1') if prior_task.ndim == 1 else rearrange(prior_task, 'k d -> d k')
                rhs = rhs + ridge_lambda * prior

            eye = torch.eye(dim, device = features.device, dtype = features.dtype)
            task = torch.linalg.solve(cov + ridge_lambda * eye, rhs)

        return rearrange(task, 'd 1 -> d') if has_single_reward else rearrange(task, 'd k -> k d')

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
        m = min(state_latents.shape[1] - 1, actions.shape[1])
        if m < 1:
            return self.zero

        states = state_latents[:, :m]
        next_states = default(next_state_latents, state_latents[:, 1:m + 1])[:, :m]

        next_actions = default(next_actions, actions[:, 1:m + 1])[:, :m]
        has_next_action = torch.arange(m, device = states.device) < next_actions.shape[1]
        next_actions = pad_right_at_dim_to(next_actions, m, dim = 1)
        actions = actions[:, :m]

        if exists(mask):
            mask = pad_right_at_dim_to(mask.bool(), m + 1, dim = 1)
            valid = mask[:, :m] & mask[:, 1:m + 1]
        else:
            valid = torch.ones((states.shape[0], m), dtype = torch.bool, device = states.device)

        bootstrap = valid
        if exists(dones):
            dones = pad_right_at_dim_to(dones[:, :m].bool(), m, dim = 1)
            bootstrap = bootstrap & ~dones

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

        gamma = broadcast_cond(gamma, states)

        states_for_grad = states.detach().lerp(states, self.frac_gradient)
        pred = self.successor_features(states_for_grad, actions, discount = gamma)

        with torch.no_grad():
            next_feature = einx.where(
                'm, b m d, b m d -> b m d',
                has_next_action,
                self.successor_features(next_states, next_actions, discount = gamma, ema = True),
                next_states
            )

            target = states + einx.multiply('b m d, b m, b m -> b m d', next_feature, gamma, bootstrap.float())

        # scale by (1 - gamma) so the loss matches the latent-feature scale, not 1 / (1 - gamma)

        loss = F.smooth_l1_loss(pred, target, reduction = 'none')
        loss = einx.multiply('b m d, b m -> b m d', loss, (1. - gamma).clamp_min(1e-3))
        loss = masked_mean(loss, valid, average_mode = 'timestep')

        if exists(self.reg) and self.reg_weight > 0. and bool(valid.any()):
            loss = loss + self.reg_weight * self.reg(states_for_grad[valid])

        return loss

    loss = compute_loss

    def update(self):
        self.ema_net.update()

        if exists(self.ema_discount_embedder):
            self.ema_discount_embedder.update()
