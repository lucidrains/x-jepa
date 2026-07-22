from typing import NamedTuple, Any

import numpy as np

import torch
from torch import tensor, is_tensor, Tensor
from torch.utils._pytree import tree_map

from einops import rearrange
from torch_einops_utils import tree_map_tensor

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

        buffer = ReplayBuffer(
            folder = folder,
            max_episodes = max_episodes,
            max_timesteps = max_timesteps,
            fields = fields,
            meta_fields = meta_fields,
            overwrite = overwrite
        )

    for i in range(batch_size):
        ep_data = dict(states = states[i], actions = actions[i])

        if exists(experience.actor_log_probs):
            ep_data['actor_log_probs'] = experience.actor_log_probs[i]
        if exists(experience.rewards):
            ep_data['rewards'] = experience.rewards[i]

        if exists(returns) and is_tensor(returns):
            ep_data['returns'] = returns[i]
        if exists(experience.state_latents):
            ep_data['state_latents'] = experience.state_latents[i]

        if exists(experience.cumulative_rewards):
            v = experience.cumulative_rewards[i]
            ep_data['cumulative_rewards'] = v.item() if is_tensor(v) else v
        if exists(experience.episode_len):
            v = experience.episode_len[i]
            ep_data['episode_len'] = v.item() if is_tensor(v) else v

        buffer.store_episode(**ep_data)

    return buffer

def experience_from_replay_buffer(
    buffer: ReplayBuffer,
    device = None
) -> Experience:

    data = buffer.get_all_data(device = device)
    states = data['states']
    actions = data['actions']
    actor_log_probs = data.get('actor_log_probs')
    rewards = data.get('rewards')
    returns = data.get('returns')
    cumulative_rewards = data.get('cumulative_rewards')
    episode_len = data.get('episode_len', data.get('episode_lens'))

    state_latents = data.get('state_latents')

    return Experience(
        states = states,
        actions = actions,
        actor_log_probs = actor_log_probs,
        rewards = rewards,
        terminated = None,
        truncated = None,
        infos = None,
        episode_len = episode_len,
        cumulative_rewards = cumulative_rewards,
        returns = returns,
        state_latents = state_latents
    )
