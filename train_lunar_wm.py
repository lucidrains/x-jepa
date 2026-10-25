# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate",
#     "einops>=0.8.0",
#     "fire",
#     "gymnasium[box2d]",
#     "memmap-replay-buffer",
#     "torch>=2.4",
#     "torch-einops-utils",
#     "x-mlps-pytorch",
# ]
# ///


import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

from functools import partial

import fire
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import Adam
from torch_einops_utils import lens_to_mask, tree_map_tensor_to_device
from x_mlps_pytorch import MLP

from x_jepa import Agent, StateSpaceHeads, state_space_fitness
from x_jepa.x_jepa import Transformer
from x_jepa.utils import divisible_by, store_experience_in_replay_buffer as store

# accelerate

accelerator = Accelerator()

# constants

STATE_DIM, ACTION_DIM = 8, 2

# landing goal: x y vx vy angle angvel

GOAL = torch.zeros(6)
SCALE = torch.tensor([.25, .20, .50, .50, .20, .50])

# helpers

def is_strict(obs, reward, terminated):
    if not terminated or reward < 50.:
        return False
    x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = map(float, obs)
    return (abs(x) < .2 and abs(y) < .15 and abs(vx) < .5 and abs(vy) < .5
            and abs(angle) < .2 and abs(angular_velocity) < .5 and left_leg > .5 and right_leg > .5)

def landing_bonus(states, alive, weight = 50.):
    x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = states.unbind(-1)
    sigmoid = torch.sigmoid
    region = (
        sigmoid((.2 - x.abs()) / .03) * sigmoid((.15 - y.abs()) / .03)
        * sigmoid((.5 - vx.abs()) / .05) * sigmoid((.5 - vy.abs()) / .05)
        * sigmoid((.2 - angle.abs()) / .03) * sigmoid((.5 - angular_velocity.abs()) / .05)
        * sigmoid((left_leg - .5) / .1) * sigmoid((right_leg - .5) / .1)
    )
    return weight * (region * alive).max(-1).values

# world model

def build_wm(device = 'cpu', dim = 64, hidden = 128, depth = 3, ema_beta = .995, gamma = .99):
    wm = Agent(
        state_encoder = MLP(STATE_DIM, hidden, dim), action_encoder = MLP(ACTION_DIM, hidden, dim),
        model = Transformer(dim = dim, depth = depth, causal = True), dim_action = ACTION_DIM,
        continuous_actions = True, transition_action_space = 'raw',
        state_linear_rnn_depth = 1, action_linear_rnn_depth = 1,
        value_loss_weight = 1.0, discount_factor = gamma, ema_beta = ema_beta, add_reflexive_actor = True,
    ).to(device)
    return wm, StateSpaceHeads(dim, STATE_DIM, hidden).to(device)

def pad_sequence(sequence, length):
    padding = length - sequence.shape[1]
    return sequence[:, :length] if padding <= 0 else F.pad(sequence, (0, padding) if sequence.dim() == 2 else (0, 0, 0, padding))

def episode_stream(buffer, count):
    while True:
        for batch in buffer.dataloader(batch_size = min(count, max(buffer.num_episodes, 1)), shuffle = True):
            yield batch

def collate_episodes(parts, device, length):
    episode_lens = torch.cat([p['episode_len'].reshape(p['states'].shape[0], -1)[:, 0] for p in parts]).clamp(max = length)
    fields = ('states', 'actions', 'rewards', 'dones', 'returns')
    batch = {field: torch.cat([pad_sequence(p[field], length - (field == 'actions')) for p in parts]) for field in fields}
    return tree_map_tensor_to_device(dict(**batch, episode_lens = episode_lens), device)

# replay

class Replay:
    def __init__(self, folder, max_episodes, max_steps, recent_capacity = 20):
        self.folder = folder
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.recent_capacity = recent_capacity
        self.all_episodes = self.recent = self.success = None

    def _add_episode(self, experience, name, buffer, capacity):
        return store(
            experience, max_episodes = capacity, max_timesteps = self.max_steps + 1,
            folder = f'{self.folder}/{name}', buffer = buffer, overwrite = buffer is None,
        )

    def add_episode(self, experience, strict):
        self.all_episodes = self._add_episode(experience, 'all', self.all_episodes, self.max_episodes)
        self.recent = self._add_episode(experience, 'recent', self.recent, self.recent_capacity)

        if strict:
            self.success = self._add_episode(experience, 'success', self.success, self.max_episodes)


# training

def train(wm, heads, opt, replay, batch_size, success_frac, recent_frac, device, steps, seq_len):
    success_batch = int(batch_size * success_frac) if replay.success is not None else 0
    recent_batch = int(batch_size * recent_frac)
    buffers = (replay.all_episodes, replay.recent, replay.success)
    batch_sizes = (batch_size - success_batch - recent_batch, recent_batch, success_batch)
    streams = [episode_stream(buffer, size) if size > 0 else None for buffer, size in zip(buffers, batch_sizes)]

    last_loss = 0.
    for _ in range(steps):
        batch = collate_episodes([next(stream) for stream in streams if stream is not None], device, seq_len)

        loss, _, outputs = wm(
            batch['states'], batch['actions'], lens = batch['episode_lens'],
            returns = None, rewards = batch['rewards'], dones = batch['dones'],
            behavior_clone = False, return_outputs = True,
        )
        aux_loss, _ = heads.loss(
            outputs['state_latents_full'], batch['states'], batch['actions'],
            returns = batch['returns'], rewards = batch['rewards'], dones = batch['dones'],
            world_model = wm.get_world_model(0),
            mask = lens_to_mask(batch['episode_lens'], batch['states'].shape[1]).float(),
        )
        accelerator.backward(loss + aux_loss)
        opt.step(); opt.zero_grad(set_to_none = True); wm.update()
        last_loss = float((loss + aux_loss).detach())
    return last_loss


def apply_strict_reward(experience, gamma):
    """anything that is not a clean pad landing is a failure"""
    episode_len = int(experience.episode_len.reshape(-1)[0])
    experience.rewards[0, episode_len - 1] = -100.

    returns = experience.returns[1] if isinstance(experience.returns, tuple) else experience.returns
    running_return = 0.
    for step in reversed(range(episode_len)):
        running_return = float(experience.rewards[0, step]) + gamma * running_return
        returns[0, step] = running_return


# main

def main(
    episodes = 250,
    random_eps = 100,
    pretrain_steps = 2000,
    batch_size = 8,
    train_steps = 300,
    train_every = 10,
    horizon = 5,
    commit_k_steps = 1,
    population = 32,
    elites = 6,
    generations = 2,
    max_steps = 500,
    lr = 3e-4,
    dim = 64, hidden = 128, depth = 3, ema_beta = .995, seq_len = 128, gamma = .99,
    value_scale = 100., value_weight = 1., value_best_weight = 0., goal_weight = 20., bonus_weight = 50.,
    success_frac = .4, recent_frac = .2, recent_capacity = 20, print_every = 10, save_every = 25,
    buffer_folder = 'lunar-wm-memories',
    checkpoint = 'lunar-wm.pt',
    seed = 0,
):
    torch.manual_seed(seed)
    device = accelerator.device
    print(f'device {device}', flush = True)

    env = gym.make('LunarLanderContinuous-v3')
    wm, heads = build_wm(device, dim, hidden, depth, ema_beta, gamma)
    opt = Adam(list(wm.parameters()) + list(heads.parameters()), lr = lr)
    replay = Replay(buffer_folder, episodes, max_steps, recent_capacity)

    fitness = state_space_fitness(
        heads.decode, heads.state_value, GOAL, SCALE, device,
        gamma = gamma, value_scale = value_scale, value_weight = value_weight,
        value_best_weight = value_best_weight, goal_weight = goal_weight,
        bonus_fn = partial(landing_bonus, weight = bonus_weight),
    )

    episode_returns = []
    for episode in range(1, episodes + 1):
        random_episode = episode <= random_eps
        experience = wm.interact_with_environment(
            env, max_steps = max_steps,
            fitness_fn = None if random_episode else fitness,
            action_fn = (lambda env: env.action_space.sample()) if random_episode else None,
            horizon = horizon, pop_size = population, generations = generations,
            elite_frac = elites / population, commit_k_steps = commit_k_steps,
            warm_start = True, use_ema_for_planning = False,
        )
        episode_returns.append(experience.cumulative_rewards.item())

        episode_len = int(experience.episode_len.reshape(-1)[0])
        strict = is_strict(
            experience.states[0, episode_len - 1].cpu().numpy(),
            float(experience.rewards[0, episode_len - 1]),
            bool(experience.terminated[0, episode_len - 1]),
        )
        if not strict:
            apply_strict_reward(experience, gamma)

        replay.add_episode(experience, strict)

        if episode == random_eps:
            loss = train(wm, heads, opt, replay, batch_size, 0., 0., device, pretrain_steps, seq_len)
            print(f'=== pretrain done | loss {loss:.4f} ===', flush = True)

        if episode > random_eps and divisible_by(episode, train_every):
            train(wm, heads, opt, replay, batch_size, success_frac, recent_frac, device, train_steps, seq_len)

        if divisible_by(episode, print_every):
            print(f'episode {episode:4d} | reward {episode_returns[-1]:8.2f} | avg100 {np.mean(episode_returns[-100:]):8.2f}', flush = True)

        if divisible_by(episode, save_every) or episode == episodes:
            torch.save(dict(wm = wm.state_dict(), heads = heads.state_dict()), checkpoint)

    print(f'saved {checkpoint}', flush = True)

if __name__ == '__main__':
    fire.Fire(main)
