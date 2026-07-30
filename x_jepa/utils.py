from typing import NamedTuple, Any
import inspect

import numpy as np

import torch
from torch import tensor, is_tensor, Tensor
from torch.utils._pytree import tree_map

from einops import rearrange
from torch_einops_utils import tree_map_tensor, lens_to_mask, masked_mean as torch_einops_masked_mean

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

def is_vectorized(env):
    if getattr(env, 'is_vector', False):
        return True

    curr = env
    while exists(curr):
        if isinstance(curr, EnvWrapper):
            return True
        curr = getattr(curr, 'env', None)

    if hasattr(env, 'num_envs') and getattr(env, 'num_envs', 0) > 0:
        return True

    try:
        from gymnasium.vector import VectorEnv
        return isinstance(getattr(env, 'unwrapped', env), VectorEnv)
    except ImportError:
        return False

def to_torch_and_batch(x, is_vector, device = None, cast_double_to_float = True):
    def transform(t):
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t)
        elif isinstance(t, (int, float, bool, np.number, np.bool_)):
            t = tensor(t)

        if not is_tensor(t):
            return t

        if cast_double_to_float and t.dtype == torch.float64:
            t = t.float()

        if not is_vector:
            # single non-vector env: scalars (rewards/dones) -> 1D (1,), multi-element (obs) -> prepended batch (1, ...)
            if t.numel() == 1:
                t = t.reshape(1)
            else:
                t = rearrange(t, '... -> 1 ...')

        if exists(device):
            t = t.to(device)

        return t

    return tree_map(transform, x)

def get_first_tensor_device(x):
    devices = set()
    tree_map_tensor(lambda t: devices.add(t.device), x)
    return next(iter(devices), None)

def to_numpy_and_unbatch(x, is_vector):
    def transform(t):
        if not is_vector:
            t = rearrange(t, '1 ... -> ...')

        return t.detach().cpu().numpy()

    return tree_map_tensor(transform, x)

# classes

class EnvWrapper:
    def __init__(
        self,
        env,
        return_cpu = False,
        cast_double_to_float = True
    ):
        assert not isinstance(env, EnvWrapper), 'EnvWrapper should only be applied once'

        self.env = env
        self.is_vector = is_vectorized(env)
        self.return_cpu = return_cpu
        self.cast_double_to_float = cast_double_to_float

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return to_torch_and_batch(obs, self.is_vector, cast_double_to_float = self.cast_double_to_float), info

    def step(self, action):

        # automatically unbatch and convert to numpy for the underlying environment

        action_device = get_first_tensor_device(action) if not self.return_cpu else None
        action = to_numpy_and_unbatch(action, self.is_vector)

        out = self.env.step(action)

        # auto-detect whether environment returns 4 or 5 items

        if len(out) == 4:
            obs, reward, terminated, info = out
            truncated = np.zeros_like(terminated) if isinstance(terminated, np.ndarray) else False
        elif len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            raise ValueError(f"expected env.step to return 4 or 5 items, got {len(out)}")

        # automatically batch and cast back to tensor on correct device

        obs, reward, terminated, truncated = to_torch_and_batch(
            (obs, reward, terminated, truncated),
            self.is_vector,
            action_device,
            cast_double_to_float = self.cast_double_to_float
        )

        return obs, reward, terminated, truncated, info

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
