from typing import NamedTuple, Any
import inspect

import torch
from torch import is_tensor, Tensor

from torch_einops_utils import masked_mean as torch_einops_masked_mean

# constants

class Experience(NamedTuple):
    states: Tensor
    actions: Tensor
    actor_log_probs: Tensor | None = None
    rewards: Tensor | None = None
    terminated: Tensor | None = None
    truncated: Tensor | None = None
    infos: Any | None = None
    episode_len: Tensor | None = None
    cumulative_rewards: Tensor | None = None
    returns: Tensor | tuple[Tensor, Tensor] | None = None
    state_latents: Tensor | None = None
    skill_ids: Tensor | None = None
    discount_factor: float | Tensor | None = None
    risk_factor: float | Tensor | None = None
    base_reward: Tensor | None = None
    risk_reward: Tensor | None = None
    skill_reward: Tensor | None = None

# fn inspection helpers

def accepts_kwarg(fn, kwarg: str) -> bool:
    if not exists(fn):
        return False
    return kwarg in inspect.signature(fn).parameters

def filter_kwargs_for_fn(fn, **kwargs):
    if not exists(fn):
        return dict()
    params = inspect.signature(fn).parameters
    return {k: v for k, v in kwargs.items() if k in params}

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def masked_mean(tensor, mask = None, dim = None, average_mode = 'timestep'):
    assert average_mode in {'timestep', 'trajectory'}, f'invalid average_mode {average_mode}'

    if average_mode == 'timestep':
        return torch_einops_masked_mean(tensor, mask = mask, dim = dim)

    seq_dim = dim if exists(dim) else 1
    trajectory_means = torch_einops_masked_mean(tensor, mask = mask, dim = seq_dim)
    return trajectory_means.mean()

# replay buffer

from memmap_replay_buffer import ReplayBuffer

def tensor_to_field_spec(t):
    dtype = 'float' if t.is_floating_point() else 'int'
    return (dtype, t.shape[2:]) if t.ndim > 2 else dtype

def store_experience_in_replay_buffer(
    experience: Experience,
    max_episodes: int,
    max_timesteps: int,
    folder = './replay_buffer',
    buffer = None,
    overwrite = True
):
    states, actions = experience.states, experience.actions
    batch_size = states.shape[0]

    returns = experience.returns
    if isinstance(returns, (tuple, list)):
        returns = returns[1]

    if not exists(buffer):
        fields = dict(
            states = tensor_to_field_spec(states),
            actions = tensor_to_field_spec(actions)
        )
        if exists(experience.actor_log_probs):
            fields['actor_log_probs'] = 'float'
        if exists(experience.rewards):
            fields['rewards'] = 'float'

        if exists(returns) and is_tensor(returns):
            fields['returns'] = 'float'
        if exists(experience.state_latents):
            fields['state_latents'] = tensor_to_field_spec(experience.state_latents)

        meta_fields = dict()
        if exists(experience.cumulative_rewards):
            meta_fields['cumulative_rewards'] = 'float'
        if exists(experience.episode_len):
            meta_fields['episode_len'] = 'int'
        if exists(experience.skill_ids):
            meta_fields['skill_ids'] = 'int' if not experience.skill_ids.is_floating_point() else 'float'

        buffer = ReplayBuffer(
            folder = folder,
            max_episodes = max_episodes,
            max_timesteps = max_timesteps,
            fields = fields,
            meta_fields = meta_fields,
            circular = True,
            overwrite = overwrite
        )

    # validate buffer schema against experience fields once up front

    candidate_time_fields = dict(
        states = states,
        actions = actions,
        actor_log_probs = experience.actor_log_probs,
        rewards = experience.rewards,
        returns = returns if (exists(returns) and is_tensor(returns)) else None,
        state_latents = experience.state_latents
    )

    candidate_meta_fields = dict(
        cumulative_rewards = experience.cumulative_rewards,
        episode_len = experience.episode_len,
        skill_ids = experience.skill_ids
    )

    active_time_fields = {
        name: tensor for name, tensor in candidate_time_fields.items()
        if exists(tensor) and name in buffer.fieldnames
    }

    active_meta_fields = {
        name: tensor for name, tensor in candidate_meta_fields.items()
        if exists(tensor) and name in buffer.meta_fieldnames
    }

    for i in range(batch_size):
        ep_data = {name: tensor[i] for name, tensor in active_time_fields.items()}

        ep_data.update({
            name: tensor[i].flatten()[0].item() if is_tensor(tensor[i]) else tensor[i]
            for name, tensor in active_meta_fields.items()
        })

        buffer.store_episode(**ep_data)

    return buffer
