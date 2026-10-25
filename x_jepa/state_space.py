"""state-space heads for latent world models"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module

from x_mlps_pytorch import MLP

from x_jepa.utils import exists


# heads

class StateDecoder(Module):
    def __init__(self, latent_dim, state_dim, hidden = 256):
        super().__init__()
        self.net = MLP(latent_dim, hidden, hidden, state_dim, activation = nn.SiLU())

    def forward(self, latents):
        return self.net(latents)


class StateValue(Module):
    def __init__(self, state_dim, hidden = 256):
        super().__init__()
        self.net = MLP(state_dim, hidden, hidden, 1, activation = nn.SiLU())

    def forward(self, states):
        return self.net(states).squeeze(-1)


class StateSpaceHeads(Module):
    """decoder + V(s), aligned to the planner's own latent rollout"""

    def __init__(self, latent_dim, state_dim, hidden = 256, value_scale = 100.):
        super().__init__()
        self.decoder = StateDecoder(latent_dim, state_dim, hidden)
        self.value = StateValue(state_dim, hidden)
        self.value_scale = value_scale

    def decode(self, latents):
        return self.decoder(latents)

    def state_value(self, states):
        return self.value(states)

    def losses(
        self,
        latents,                 # (b n d)
        states,                  # (b n s)
        actions = None,          # (b n a)
        returns = None,          # (b n)
        rewards = None,          # (b n)
        dones = None,            # (b n)
        world_model = None,
        mask = None              # (b n)
    ):
        n = latents.shape[1]
        states = states[:, :n]
        tgt = returns[:, :n] / self.value_scale if exists(returns) else None

        decoded = self.decoder(latents)                                  # (b n s)
        out = dict(decoder = F.smooth_l1_loss(decoded, states, reduction = 'none').mean(-1))

        if exists(tgt):
            out['value'] = F.smooth_l1_loss(self.value(states), tgt, reduction = 'none')
            out['value_decoded'] = F.smooth_l1_loss(self.value(decoded.detach()), tgt, reduction = 'none')

        # align the heads to the latents the planner actually rolls out
        if exists(world_model) and exists(actions) and n > 1:
            m = min(n - 1, actions.shape[1])

            with torch.no_grad():
                residual = world_model.state_transition_for_planning.predict(latents[:, :m, None], actions[:, :m, None])
                next_latents = world_model.apply_transition_residual(latents[:, :m, None], residual).squeeze(2)  # (b m d)

            decoded_next = self.decoder(next_latents)                    # (b m s)
            out['decoder_next'] = F.smooth_l1_loss(decoded_next, states[:, 1:m + 1], reduction = 'none').mean(-1)

            if exists(tgt):
                out['value_next'] = F.smooth_l1_loss(self.value(decoded_next.detach()), tgt[:, 1:m + 1], reduction = 'none')

            if exists(getattr(world_model, 'to_reward_pred', None)) and exists(rewards):
                pred_r = world_model.to_reward_pred((latents[:, :m], actions[:, :m])).squeeze(-1)
                out['reward'] = F.smooth_l1_loss(pred_r, rewards[:, :m], reduction = 'none')

            if exists(getattr(world_model, 'to_discount_pred', None)) and exists(dones):
                pred_d = world_model.to_discount_pred((latents[:, :m], actions[:, :m])).squeeze(-1)
                out['discount'] = F.binary_cross_entropy_with_logits(pred_d, 1. - dones[:, :m].float(), reduction = 'none')

        if exists(mask):
            next_keys = {'decoder_next', 'value_next', 'reward', 'discount'}
            masked = {}
            for k, v in out.items():
                m = mask[:, 1:v.shape[1] + 1] if k in next_keys else mask[:, :v.shape[1]]
                masked[k] = (v * m).sum() / m.sum().clamp_min(1.)
            out = masked
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
        gamma_ = gamma if discount_factor is None else discount_factor
        states = decode(pred_state_latents)                             # (b p h s)
        alive = torch.cumprod(pred_discounts, dim = -1)                 # (b p h)
        alive_prev = F.pad(alive[..., :-1], (1, 0), value = 1.)

        horizon = states.shape[-2]
        discounts = gamma_ ** torch.arange(horizon, device = device)               # (h,)
        ret = (pred_rewards * discounts * alive_prev).sum(-1)

        v = value(states) * value_scale
        v_final = v[..., -1] * alive[..., -1] * gamma_ ** horizon                  # gamma^H V(s_H)
        v_best = (v * alive_prev * (1. - pred_discounts) * discounts * gamma_).max(-1).values

        score = ret + value_weight * ((1. - value_best_weight) * v_final + value_best_weight * v_best)
        dist = (((states[..., :goal.shape[-1]] - goal) / scale) ** 2).sum(-1)
        score = score - goal_weight * dist.min(-1).values

        return score + bonus_fn(states, alive_prev) if exists(bonus_fn) else score

    return fitness
