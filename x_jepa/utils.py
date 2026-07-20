from collections import namedtuple

import numpy as np

import torch
from torch import tensor, is_tensor
from torch.utils._pytree import tree_map

from einops import rearrange
from torch_einops_utils import tree_map_tensor

# constants

Experience = namedtuple('Experience', [
    'states',
    'actions',
    'actor_log_probs',
    'rewards',
    'terminated',
    'truncated',
    'infos',
    'episode_len',
    'cumulative_rewards',
    'returns'
])

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def is_vectorized(env):
    num_envs = getattr(env, 'num_envs', 0)
    is_vector_env = getattr(env, 'is_vector_env', False)
    return num_envs > 0 or is_vector_env

def to_torch_and_batch(x, is_vector, device = None):
    def transform(t):
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t)
        elif isinstance(t, (int, float, bool, np.number, np.bool_)):
            t = tensor(t)

        if not is_tensor(t):
            return t

        if not is_vector:
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
    def __init__(self, env, return_cpu = False):
        assert not isinstance(env, EnvWrapper), 'EnvWrapper should only be applied once'

        self.env = env
        self.is_vector = is_vectorized(env)
        self.return_cpu = return_cpu

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return to_torch_and_batch(obs, self.is_vector), info

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
            action_device
        )

        return obs, reward, terminated, truncated, info
