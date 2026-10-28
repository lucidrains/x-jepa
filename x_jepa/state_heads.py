"""state readout heads for latent world models - decoder + V(s), not an SSM"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, is_tensor, tensor
from torch.nn import Module
from einops import rearrange, repeat

from ema_pytorch import EMA
from torch_einops_utils import lens_to_mask
from x_mlps_pytorch import MLP

from x_jepa.utils import exists, default
from x_jepa.value_networks import DiscountEmbedder, resolve_discount, broadcast_cond

# helpers

def pad_and_unfold(t, seq_len, window, pad_value = 0):
    t = t[:, :seq_len]
    t = F.pad(t, (0, max(seq_len - t.shape[1], 0) + window - 1), value = pad_value)
    return t.unfold(1, window, 1)

def n_step_td_target(
    returns,
    boot_values,
    discounts = 0.99,
    n_step = 4,
    valid_len = None,
    dones = None
):
    batch, seq_len, device, dtype = *returns.shape[:2], returns.device, returns.dtype
    is_3d = returns.ndim == 3

    t = torch.arange(seq_len, device = device)[None, :]
    valid_len = default(valid_len, torch.full((batch, 1), seq_len, device = device, dtype = torch.long))
    last_step = (valid_len - 1).clamp(min = 0, max = seq_len - 1)
    boot_step = (t + n_step).clamp(max = last_step)

    can_boot = (t + n_step) < valid_len

    if exists(dones):
        dones_win = pad_and_unfold(dones.to(device).bool(), seq_len, n_step, pad_value = False)
        can_boot = can_boot & (~dones_win.any(dim = -1))

    gamma = torch.as_tensor(default(discounts, 0.99), dtype = dtype, device = device)

    if gamma.ndim == 2:
        gamma_n = pad_and_unfold(gamma, seq_len, n_step, pad_value = 1.).prod(dim = -1)
    else:
        gamma = rearrange(gamma, 'b -> b 1') if gamma.ndim == 1 else gamma
        gamma_n = gamma ** n_step

    gather_idx = repeat(boot_step, 'b n -> b n k', k = returns.shape[-1]) if is_3d else boot_step
    boot_next = boot_values.gather(1, gather_idx)
    ret_boot = returns.gather(1, gather_idx)

    if is_3d:
        gamma_n = rearrange(gamma_n, '... -> ... 1') if gamma_n.ndim == 2 else gamma_n
        can_boot = rearrange(can_boot, 'b n -> b n 1')

    td_returns = torch.where(can_boot, returns + gamma_n * (boot_next - ret_boot), returns)
    valid = (t < valid_len) if not is_3d else rearrange(t < valid_len, 'b n -> b n 1')
    return td_returns.masked_fill(~valid, 0.)

@torch.no_grad()
def n_step_returns(
    states,
    returns = None,
    value_fn = None,
    discounts = 0.99,
    lens = None,
    n_step = 4,
    td_weight = 1.,
    rewards = None,
    dones = None,
    value_scale = 1.,
    clamp_range: tuple[float, float] | None = None
):
    batch, seq_len, device, dtype = *states.shape[:2], states.device, states.dtype

    t = torch.arange(seq_len, device = device)[None, :]
    valid_len = lens.to(device).clamp(min = 0, max = seq_len)[:, None] if exists(lens) else torch.full((batch, 1), seq_len, device = device, dtype = torch.long)
    last_step = (valid_len - 1).clamp(min = 0, max = seq_len - 1)
    boot_step = (t + n_step).clamp(max = last_step)
    valid = t < valid_len

    for mod in (getattr(value_fn, 'ema_model', None), value_fn):
        if hasattr(mod, 'record_discount_training'):
            mod.record_discount_training(discounts)
            break

    boot_values = value_fn(states, discount = discounts) * value_scale

    if exists(clamp_range):
        boot_values = boot_values.clamp(*clamp_range)

    boot_values = boot_values.masked_fill(~valid, 0.)

    if not exists(rewards):
        assert exists(returns), 'either returns or rewards must be provided to n_step_returns'
        mc_returns = returns[:, :seq_len].to(device = device, dtype = dtype).masked_fill(~valid, 0.)
        td_returns = n_step_td_target(mc_returns, boot_values, discounts = discounts, n_step = n_step, valid_len = valid_len, dones = dones)
        return ((1. - td_weight) * mc_returns + td_weight * td_returns).masked_fill(~valid, 0.)

    # direct n-step rollout from rewards and dones

    k = torch.arange(n_step, device = device)
    valid_win = (t[:, :, None] + k) < valid_len[:, :, None]

    rewards_win = pad_and_unfold(rewards.to(device = device, dtype = dtype), seq_len, n_step, pad_value = 0.)
    dones_win = pad_and_unfold(dones.to(device).bool(), seq_len, n_step, pad_value = False) if exists(dones) else torch.zeros_like(valid_win)
    dones_win = dones_win & valid_win

    survived = (~dones_win).float().cumprod(dim = -1)
    alive_before = F.pad(survived[..., :-1], (1, 0), value = 1.) * valid_win.float()
    alive_after = survived[..., -1]
    terminated_in_window = dones_win.any(dim = -1)

    gamma = torch.as_tensor(discounts, dtype = dtype, device = device)
    is_per_step = gamma.ndim == 2

    if is_per_step:
        g_win = pad_and_unfold(gamma, seq_len, n_step, pad_value = 1.)
        discount_pows = F.pad(g_win.cumprod(dim = -1)[..., :-1], (1, 0), value = 1.)
        gamma_n = g_win.prod(dim = -1)
    else:
        gamma = rearrange(gamma, 'b -> b 1') if gamma.ndim == 1 else gamma
        discount_pows = gamma[..., None] ** k
        gamma_n = gamma ** n_step

    n_step_rew = (rewards_win * discount_pows * alive_before).sum(dim = -1)
    boot_next = boot_values.gather(1, boot_step)
    can_boot = (t + n_step) < valid_len
    td_boot = n_step_rew + gamma_n * alive_after * boot_next

    if exists(returns):
        mc_returns = returns[:, :seq_len].to(device = device, dtype = dtype).masked_fill(~valid, 0.)
        tail_returns = torch.where(terminated_in_window, n_step_rew, mc_returns)
    else:
        tail_disc = (gamma ** (last_step - t + 1).clamp_min(1)) if not is_per_step else gamma_n
        tail_boot = n_step_rew + tail_disc * alive_after * boot_values.gather(1, last_step)
        tail_returns = torch.where(terminated_in_window, n_step_rew, tail_boot)
        mc_returns = torch.where(can_boot, td_boot, tail_returns)

    td_returns = torch.where(can_boot, td_boot, tail_returns)
    return ((1. - td_weight) * mc_returns + td_weight * td_returns).masked_fill(~valid, 0.)

# heads

class StateDecoder(Module):
    def __init__(self, latent_dim, state_dim, hidden = 256):
        super().__init__()
        self.net = MLP(latent_dim, hidden, hidden, state_dim, activation = nn.SiLU())

    def forward(self, latents):
        return self.net(latents)

class StateValue(Module):
    def __init__(self, state_dim, hidden = 256, eps = 1e-20):
        super().__init__()
        self.to_hidden = nn.Linear(state_dim, hidden)
        self.discount_embedder = DiscountEmbedder(hidden, eps = eps)
        self.net = MLP(hidden, hidden, 1, activation = nn.SiLU())

        self.register_buffer('trained_with_discount', tensor(0.))
        self.register_buffer('trained_without_discount', tensor(0.))
        self.register_buffer('min_trained_discount', tensor(1.))
        self.register_buffer('max_trained_discount', tensor(0.))

    @property
    def has_trained_with_discount(self) -> bool:
        return bool(self.trained_with_discount.item() > 0.)

    @property
    def has_trained_without_discount(self) -> bool:
        return bool(self.trained_without_discount.item() > 0.)

    @property
    def is_uvfa_trained(self) -> bool:
        return self.has_trained_with_discount and bool((self.max_trained_discount - self.min_trained_discount).item() > 1e-6)

    def record_discount_training(self, discount):
        if not exists(discount):
            self.trained_without_discount.fill_(1.)
            self.trained_with_discount.fill_(0.)
            self.min_trained_discount.fill_(1.)
            self.max_trained_discount.fill_(0.)
            return

        d = torch.as_tensor(discount, dtype = torch.float32, device = self.min_trained_discount.device)
        assert bool((d >= 0.).all() and (d <= 1.).all()), 'discount factor must be in [0, 1]'

        if not self.has_trained_with_discount:
            self.min_trained_discount.fill_(1.)
            self.max_trained_discount.fill_(0.)

        self.trained_with_discount.fill_(1.)
        self.trained_without_discount.fill_(0.)
        self.min_trained_discount.copy_(torch.minimum(self.min_trained_discount, d.min()))
        self.max_trained_discount.copy_(torch.maximum(self.max_trained_discount, d.max()))

    def validate_discount(self, discount):
        if not exists(discount):
            assert not self.has_trained_with_discount, 'StateValue was trained with discount factor conditioning and requires a discount factor'
            return

        assert self.has_trained_with_discount, 'StateValue was not trained with discount factor conditioning'
        d = torch.as_tensor(discount, dtype = torch.float32, device = self.min_trained_discount.device)
        assert bool((d >= 0.).all() and (d <= 1.).all()), 'discount factor must be in [0, 1]'

        if not self.is_uvfa_trained:
            assert torch.allclose(d, self.min_trained_discount, atol = 1e-5), (
                f'StateValue was only trained with a single fixed discount factor ({self.min_trained_discount.item():.4f}) '
                'and not UVFA discount factor conditioning; cannot vary discount_factor at evaluation / planning time'
            )

    def embed_discount(self, states, discount):
        if not exists(discount):
            return 0.

        return self.discount_embedder(broadcast_cond(discount, states))

    def forward(self, states, discount = None):
        self.validate_discount(discount)
        hidden = F.silu(self.to_hidden(states) + self.embed_discount(states, discount))
        return self.net(hidden).squeeze(-1)

    def loss(self, states, returns, discounts = None, mask = None, value_scale = 1.):
        if isinstance(returns, (tuple, list)):
            tuple_discounts, returns = returns
            discounts = default(discounts, tuple_discounts)

        self.record_discount_training(discounts)
        pred = self.forward(states, discount = discounts)
        tgt = returns / value_scale
        loss = F.smooth_l1_loss(pred, tgt, reduction = 'none')

        if exists(mask):
            m = mask[:, :loss.shape[1]]
            return (loss * m).sum() / m.sum().clamp_min(1.)

        return loss.mean()

class StateSpaceHeads(Module):
    """decoder + V(s), aligned to the planner's own latent rollout"""

    def __init__(self, latent_dim, state_dim, hidden = 256, value_scale = 100., ema_beta = 0.99):
        super().__init__()
        self.decoder = StateDecoder(latent_dim, state_dim, hidden)
        self.value = StateValue(state_dim, hidden)
        self.ema_value = EMA(self.value, beta = ema_beta, update_after_step = 0, update_every = 1)
        self.value_scale = value_scale

    def update(self):
        self.ema_value.update()

    def decode(self, latents):
        return self.decoder(latents)

    def state_value(self, states, discount = None):
        return self.value(states, discount = discount)

    @torch.no_grad()
    def n_step_returns(
        self,
        states,
        returns = None,
        discounts = 0.99,
        lens = None,
        n_step = 4,
        td_weight = 1.,
        rewards = None,
        dones = None,
        clamp_range: tuple[float, float] | None = None
    ):
        return n_step_returns(
            states,
            returns,
            self.ema_value,
            discounts = discounts,
            lens = lens,
            n_step = n_step,
            td_weight = td_weight,
            rewards = rewards,
            dones = dones,
            value_scale = self.value_scale,
            clamp_range = clamp_range
        )

    def losses(
        self,
        latents,                 # (b n d)
        states,                  # (b n s)
        actions = None,          # (b n a)
        returns = None,          # (b n) | (discounts, (b n))
        rewards = None,          # (b n)
        dones = None,            # (b n)
        discounts = None,        # float | (b,) | (b n) | (min_discount, max_discount)
        n_step: int | None = None,
        lens = None,             # (b,)
        world_model = None,
        mask = None              # (b n)
    ):
        if isinstance(returns, (tuple, list)):
            tuple_discounts, returns = returns
            discounts = default(discounts, tuple_discounts)

        discounts = resolve_discount(discounts, batch = states.shape[0], device = states.device, dtype = states.dtype, sample = True)

        if exists(lens) and not exists(mask):
            mask = lens_to_mask(lens, states.shape[1]).float()

        if (exists(returns) or exists(rewards)) and exists(n_step):
            assert exists(discounts), 'discounts must be provided when computing n_step returns'
            returns = self.n_step_returns(states, returns, discounts = discounts, lens = lens, n_step = n_step, rewards = rewards, dones = dones)

        n = latents.shape[1]
        states = states[:, :n]
        tgt = returns[:, :n] / self.value_scale if exists(returns) else None
        disc_n = discounts[:, :n] if (is_tensor(discounts) and discounts.ndim == 2) else discounts

        decoded = self.decoder(latents)                                  # (b n s)
        out = dict(decoder = F.smooth_l1_loss(decoded, states, reduction = 'none').mean(-1))

        if exists(tgt):
            self.value.record_discount_training(disc_n)
            self.ema_value.ema_model.record_discount_training(disc_n)
            out['value'] = F.smooth_l1_loss(self.value(states, discount = disc_n), tgt, reduction = 'none')
            out['value_decoded'] = F.smooth_l1_loss(self.value(decoded.detach(), discount = disc_n), tgt, reduction = 'none')

        # align the heads to the latents the planner actually rolls out

        if exists(world_model) and exists(actions) and n > 1:
            m = min(n - 1, actions.shape[1])

            if getattr(world_model, 'transition_is_prefix', False) and getattr(world_model.state_transition_for_planning, 'lookahead', 1) == 1:
                pred_h, _ = world_model.state_transition_for_planning(latents[:, :m].detach(), actions[:, :m])
                k = pred_h.shape[1] * pred_h.shape[2]
                next_latents = world_model.apply_transition_residual(latents[:, :k:pred_h.shape[2], None].detach(), pred_h).reshape(latents.shape[0], k, -1)
            else:
                anchor = latents[:, :m, None].detach()
                residual = world_model.state_transition_for_planning.predict(anchor, actions[:, :m, None])
                next_latents = world_model.apply_transition_residual(anchor, residual).squeeze(2)  # (b m d)
                k = m

            decoded_next = self.decoder(next_latents)                    # (b k s)
            out['decoder_next'] = F.smooth_l1_loss(decoded_next, states[:, 1:k + 1], reduction = 'none').mean(-1)

            if exists(tgt):
                disc_k = discounts[:, 1:k + 1] if (is_tensor(discounts) and discounts.ndim == 2) else discounts
                out['value_next'] = F.smooth_l1_loss(self.value(decoded_next.detach(), discount = disc_k), tgt[:, 1:k + 1], reduction = 'none')

            if exists(getattr(world_model, 'to_reward_pred', None)) and exists(rewards):
                mr = min(n, actions.shape[1], rewards.shape[1])
                pred_r = world_model.to_reward_pred((latents[:, :mr], actions[:, :mr])).squeeze(-1)
                out['reward'] = F.smooth_l1_loss(pred_r, rewards[:, :mr], reduction = 'none')

            if exists(getattr(world_model, 'to_discount_pred', None)) and exists(dones):
                md = min(n, actions.shape[1], dones.shape[1])
                pred_d = world_model.to_discount_pred((latents[:, :md], actions[:, :md])).squeeze(-1)
                out['discount'] = F.binary_cross_entropy_with_logits(pred_d, 1. - dones[:, :md].float(), reduction = 'none')

        if exists(mask):
            next_keys = {'decoder_next', 'value_next'} if (exists(actions) and actions.shape[1] >= n) else {'decoder_next', 'value_next', 'reward', 'discount'}
            out = {
                k: ((v * (m := mask[:, 1:v.shape[1] + 1] if k in next_keys else mask[:, :v.shape[1]])).sum(-1) / m.sum(-1).clamp_min(1.)).mean()
                for k, v in out.items()
            }
        else:
            out = {k: v.mean() for k, v in out.items()}

        return out

    def loss(self, *args, **kwargs):
        losses = self.losses(*args, **kwargs)
        return sum(losses.values()), losses

# fitness

def state_space_fitness(
    decode,
    value,
    goal,
    scale = 1.,
    device = 'cpu',
    gamma = .99,
    value_scale = 100.,
    value_weight = 1.,
    value_best_weight = .5,
    goal_weight = 1.,
    bonus_fn = None,             # (states, alive) -> (b p)
):
    goal = torch.as_tensor(goal, dtype = torch.float32, device = device)
    scale = torch.as_tensor(scale, dtype = torch.float32, device = device)

    def fitness(pred_state_latents, pred_rewards, pred_discounts, discount_factor = None):
        gamma_ = default(discount_factor, gamma)
        states = decode(pred_state_latents)                             # (b p h s)
        alive = pred_discounts.cumprod(dim = -1)                        # (b p h)
        alive_prev = F.pad(alive[..., :-1], (1, 0), value = 1.)

        horizon = states.shape[-2]
        gamma_t = broadcast_cond(gamma_, states[..., 0, :]).to(device = states.device, dtype = states.dtype)

        steps = torch.arange(horizon, dtype = states.dtype, device = states.device)
        discounts = gamma_t.unsqueeze(-1) ** steps                      # (b p h)
        ret = (pred_rewards * discounts * alive_prev).sum(-1)

        v = value(states, discount = gamma_) * value_scale
        v_final = v[..., -1] * alive[..., -1] * (gamma_t ** horizon)    # gamma^H V(s_H)
        v_best = (v * alive_prev * (1. - pred_discounts) * discounts * gamma_t.unsqueeze(-1)).max(-1).values

        score = ret + value_weight * ((1. - value_best_weight) * v_final + value_best_weight * v_best)
        goal_ = goal.to(device = states.device, dtype = states.dtype)
        scale_ = scale.to(device = states.device, dtype = states.dtype)
        dist = (((states[..., :goal_.shape[-1]] - goal_) / scale_) ** 2).sum(-1)
        score = score - goal_weight * dist.min(-1).values

        return score + bonus_fn(states, alive_prev) if exists(bonus_fn) else score

    return fitness
