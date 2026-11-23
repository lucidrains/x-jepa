# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate",
#     "ale-py",
#     "einops>=0.8.0",
#     "fire",
#     "gymnasium>=1.0",
#     "imageio[ffmpeg]",
#     "memmap-replay-buffer",
#     "Pillow",
#     "torch>=2.4",
#     "wandb>=0.25.1"
# ]
# ///
#
# Montezuma's Revenge with x-JEPA world model + CEM planning + LearnableNovelty (arXiv:2607.18433)
#
#   python train_montezuma.py                       # default run (+LearnableNovelty with LOO)
#   python train_montezuma.py --smoke True          # quick smoke test
#   python train_montezuma.py --use_loo False       # vanilla LearnableNovelty
#   python train_montezuma.py --use_intrinsic False # baseline without novelty bonus

from __future__ import annotations

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import csv
from collections import deque

import fire
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch import cat
from torch.optim import Adam

from accelerate import Accelerator
import gymnasium as gym
import ale_py # noqa: F401 - registers ALE envs with gymnasium

from einops import rearrange, pack, unpack
from torch_einops_utils import tree_map_tensor_to_device

from x_jepa import Agent, LearnableNovelty, LearnableNoveltyReward, rollout_return
from x_jepa.x_jepa import Transformer, calc_returns
from x_jepa.utils import store_experience_in_replay_buffer, divisible_by, exists

# environment

class MontezumaEnv:
    def __init__(self, seed, frame_size = 42, stack_frames = 4, max_episode_steps = 400, record = False):
        self.frame_size = frame_size
        self.stack_frames = stack_frames
        self.max_episode_steps = max_episode_steps
        self.record = record
        self.seed = seed

        self.env = gym.make('ALE/MontezumaRevenge-v5', frameskip = 4, full_action_space = False)
        self.action_space = gym.spaces.Discrete(self.env.action_space.n)
        self.observation_space = gym.spaces.Box(0., 1., (stack_frames, frame_size, frame_size), dtype = np.float32)

        self.room = 1
        self.max_room = 1
        self.rooms_seen = {1}
        self.steps = 0
        self._frames = None
        self._video = []

    def _preprocess(self, rgb):
        img = Image.fromarray(rgb).convert('L').resize((self.frame_size, self.frame_size), Image.BILINEAR)
        return np.asarray(img, dtype = np.float32) / 255.

    def _read_room(self):
        return int(self.env.unwrapped.ale.getRAM()[3]) + 1

    def _obs(self):
        return np.stack(self._frames, axis = 0)

    def reset(self, seed = None, options = None):
        if exists(seed):
            self.seed = seed

        obs, _ = self.env.reset(seed = self.seed)
        self.steps = 0
        self.room = self._read_room()
        self.max_room = self.room
        self.rooms_seen = {self.room}
        self._video = [obs.copy()] if self.record else []
        self._frames = deque([self._preprocess(obs)] * self.stack_frames, maxlen = self.stack_frames)
        return self._obs(), {}

    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(int(np.asarray(action).item()))
        self.steps += 1
        self.room = self._read_room()
        self.max_room = max(self.max_room, self.room)
        self.rooms_seen.add(self.room)

        if self.record:
            self._video.append(obs.copy())

        self._frames.append(self._preprocess(obs))
        truncated = truncated or self.steps >= self.max_episode_steps
        return self._obs(), float(reward), bool(terminated), bool(truncated), {}

    def close(self):
        self.env.close()

class BatchedEnv:
    """No auto-reset: done envs freeze so each env yields one episode per rollout."""

    def __init__(self, seeds, max_episode_steps = 400, frame_size = 42, stack_frames = 4, record = False):
        self.envs = [MontezumaEnv(s, frame_size, stack_frames, max_episode_steps, record) for s in seeds]
        self.num_envs = len(self.envs)
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self._done = [False] * self.num_envs
        self._last_obs = [None] * self.num_envs

    def reset(self, seed = None, options = None):
        self._done = [False] * self.num_envs
        obs = [env.reset(seed = seed)[0] for env in self.envs]
        self._last_obs = list(obs)
        return np.stack(obs), {}

    def step(self, actions):
        actions = np.asarray(actions).reshape(self.num_envs)
        obs, rewards, terminateds, truncateds = [], [], [], []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if self._done[i]:
                obs.append(self._last_obs[i])
                rewards.append(0.)
                terminateds.append(True)
                truncateds.append(False)
                continue

            o, r, term, trunc, _ = env.step(action)
            self._last_obs[i] = o
            self._done[i] = term or trunc
            obs.append(o)
            rewards.append(r)
            terminateds.append(term)
            truncateds.append(trunc)

        return (
            np.stack(obs),
            np.array(rewards, dtype = np.float32),
            np.array(terminateds, dtype = bool),
            np.array(truncateds, dtype = bool),
            {}
        )

    def close(self):
        for env in self.envs:
            env.close()

# encoder

class CNNEncoder(nn.Module):
    def __init__(self, frame_size = 42, stack_frames = 4, dim = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(stack_frames, 16, 5, stride = 2, padding = 2),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride = 2, padding = 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride = 2, padding = 1),
            nn.SiLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            feat_dim = self.conv(torch.zeros(1, stack_frames, frame_size, frame_size)).shape[-1]

        self.to_latent = nn.Linear(feat_dim, dim)

    def forward(self, x):
        x, ps = pack([x], '* c h w')
        out = self.to_latent(self.conv(x))
        return unpack(out, ps, '* d')[0]

# planning fitness

def make_fitness(gamma = 0.99, novelty_reward = None, imagined_novelty_weight = 0.25, history_len = 32):
    latent_history = deque(maxlen = history_len)

    def reset_history():
        latent_history.clear()

    def fitness_fn(pred_state_latents, state_latents, pred_rewards, pred_discounts, pred_values):
        score = rollout_return(
            pred_rewards = pred_rewards,
            pred_discounts = pred_discounts,
            pred_values = pred_values,
            discount_factor = gamma,
            termination_threshold = 0.5
        )

        if exists(novelty_reward) and imagined_novelty_weight > 0.:
            curr = state_latents[:, 0].detach()
            if len(latent_history) == 0 or not torch.equal(latent_history[-1], curr):
                latent_history.append(curr)

            if len(latent_history) >= 2:
                history = torch.stack(list(latent_history), dim = 1)
                score = score + imagined_novelty_weight * novelty_reward.continuation_reward(history, pred_state_latents)

        return score

    return fitness_fn, reset_history

# coverage tracking

def count_unique_frames(states, episode_lens, unique_seen):
    episode_unique = 0
    for i, length in enumerate(episode_lens):
        hashes = {np.round(states[i, t, -1] * 15).astype(np.uint8).tobytes() for t in range(int(length))}
        episode_unique += len(hashes)
        unique_seen.update(hashes)
    return episode_unique, len(unique_seen)

# world model training

def train_wm_epoch(wm, buffer, optimizer, device, batch_size, epochs_per_train, grad_accum_steps, train_window, batches_per_epoch):
    dl = buffer.dataloader(
        batch_size = batch_size,
        shuffle = True,
        n_steps = train_window,
        return_indices = True,
        sequence_fields = ('states', 'actions', 'returns', 'rewards', 'dones', 'base_rewards'),
        fieldname_map = {
            'seq_states': 'states',
            'seq_actions': 'actions',
            'seq_returns': 'returns',
            'seq_rewards': 'rewards',
            'seq_dones': 'dones',
            'seq_base_rewards': 'base_rewards'
        }
    )
    dl_iter = iter(dl)
    wm.train()

    loss_val = reward_loss_val = discount_loss_val = 0.
    total_batches = epochs_per_train * batches_per_epoch

    for step in range(total_batches):
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)

        batch = tree_map_tensor_to_device(batch, device)
        states, actions, returns, base_rewards, dones = (
            batch['states'], batch['actions'], batch['returns'], batch['base_rewards'], batch['dones']
        )
        lens = batch.get('n_step_lens', batch.get('episode_len', batch.get('episode_lens')))

        if '_indices' in batch and 'episode_len' in batch:
            lens = torch.minimum(lens, (batch['episode_len'] - batch['_indices'][:, 1]).clamp(min = 0))

        loss, breakdown = wm(
            states, actions,
            lens = lens,
            returns = returns,
            rewards = base_rewards,
            dones = dones,
            behavior_clone = True
        )
        (loss / grad_accum_steps).backward()

        if divisible_by(step + 1, grad_accum_steps) or step + 1 == total_batches:
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)
            wm.update()
            loss_val = loss.item()
            reward_loss_val = breakdown.reward_pred.item()
            discount_loss_val = breakdown.discount_pred.item()

    return loss_val, reward_loss_val, discount_loss_val

# main

def main(
    dim = 32,
    depth = 2,
    batch_size = 32,
    grad_accum_steps = 2,
    lr = 3e-3,
    num_episodes = 40,
    num_envs = 6,
    max_timesteps = 400,
    train_every_eps = 1,
    epochs_per_train = 3,
    batches_per_epoch = 6,
    train_window = 64,
    max_episodes_buffer = 40,
    cem_num_samples = 128,
    cem_num_iters = 3,
    cem_elite_frac = 0.1,
    cem_horizon = 8,
    commit_k_steps = 4,
    random_action_eps = 10,
    gamma = 0.99,
    use_intrinsic = True,
    use_loo = True,
    window_horizon = 1,
    reservoir_dim = 32,
    reservoir_layers = 4,
    ridge_penalty = 0.03,
    target_scale = 0.02,
    eta = 32.,
    novelty_scale = 10.,
    increment_stride = 4,
    intrinsic_reward_scale = 1.,
    imagined_novelty_weight = 0.25,
    history_len = 32,
    existence_penalty = -0.01,
    death_penalty = -6.,
    seed_with_actor_after = 25,
    frame_size = 42,
    stack_frames = 4,
    seed = 0,
    tag = '',
    record_video = False,
    video_fps = 15,
    project_name = 'x-jepa-montezuma',
    use_wandb = False,
    cpu = False,
    smoke = False
):
    if smoke:
        num_envs = 2
        max_timesteps = 64
        num_episodes = 3
        random_action_eps = 1
        cem_num_samples = 16
        cem_num_iters = 1
        train_window = 32
        batch_size = 4
        epochs_per_train = 1
        batches_per_epoch = 2

    device = Accelerator(cpu = cpu).device
    condition = ('ln-loo' if use_loo else 'ln') if use_intrinsic else 'baseline'
    suffix = f'-{tag}' if tag else f'-{condition}-s{seed}'
    print(f'device: {device} | condition: {condition} | seed: {seed}', flush = True)

    wandb_mod = None
    if use_wandb:
        import wandb as wandb_mod
        wandb_mod.init(project = project_name, config = dict(condition = condition, seed = seed, use_loo = use_loo))

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = BatchedEnv(
        [seed * 1000 + i for i in range(num_envs)],
        max_episode_steps = max_timesteps,
        frame_size = frame_size,
        stack_frames = stack_frames,
        record = record_video
    )
    action_dim = int(env.single_action_space.n)

    wm = Agent(
        state_encoder = CNNEncoder(frame_size = frame_size, stack_frames = stack_frames, dim = dim),
        action_encoder = nn.Embedding(action_dim, dim),
        action_decoder = nn.Linear(dim, action_dim),
        model = Transformer(dim = dim, depth = depth, causal = True),
        dim_action = action_dim,
        continuous_actions = False,
        transition_action_space = 'local',
        state_linear_rnn_depth = 1,
        action_linear_rnn_depth = 1,
        value_loss_weight = 1.,
        discount_factor = gamma,
        add_reflexive_actor = True
    ).to(device)

    optimizer = Adam(wm.parameters(), lr = lr)

    novelty_reward = None
    if use_intrinsic:
        with torch.random.fork_rng():
            torch.manual_seed(seed + 101)
            novelty_reward = LearnableNoveltyReward(
                LearnableNovelty(
                    dim_data = dim,
                    mlp_reservoir_dim = reservoir_dim,
                    mlp_reservoir_layers = reservoir_layers,
                    ridge_penalty = ridge_penalty,
                    target_scale = target_scale,
                    eta = eta,
                    use_loo = use_loo
                ),
                horizon = window_horizon,
                scale = novelty_scale,
                stride = increment_stride
            )

    fitness_fn, reset_history = make_fitness(
        gamma = gamma,
        novelty_reward = novelty_reward,
        imagined_novelty_weight = imagined_novelty_weight,
        history_len = history_len
    )

    buffer = None
    buffer_folder = f'./montezuma-memories{suffix}'
    csv_path = f'./montezuma{suffix}.csv'
    unique_seen = set()
    wm_loss = reward_loss = discount_loss = 0.

    fieldnames = [
        'episode', 'condition', 'seed', 'mean_steps', 'total_reward',
        'unique_new', 'unique_cum', 'max_room', 'rooms',
        'epiplexity', 'mean_bonus', 'wm_loss', 'wm_reward_loss', 'wm_discount_loss'
    ]

    with open(csv_path, 'w', newline = '') as f:
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()

        for episode in range(1, num_episodes + 1):
            wm.eval()
            reset_history()
            is_random_ep = episode <= random_action_eps

            experience = wm.interact_with_environment(
                env = env,
                max_steps = max_timesteps,
                fitness_fn = None if is_random_ep else fitness_fn,
                action_fn = (lambda: torch.randint(0, action_dim, (num_envs, 1))) if is_random_ep else None,
                horizon = cem_horizon,
                pop_size = cem_num_samples,
                generations = cem_num_iters,
                elite_frac = cem_elite_frac,
                commit_k_steps = commit_k_steps,
                seed_with_actor = 'reflexive' if (not is_random_ep and exists(seed_with_actor_after) and episode >= seed_with_actor_after) else None
            )

            if experience.actions.ndim == 3 and experience.actions.shape[-1] == 1:
                experience = experience._replace(actions = rearrange(experience.actions, 'b n 1 -> b n'))

            states = experience.states.cpu().numpy()
            episode_lens = experience.episode_len.reshape(-1).tolist()
            base_rewards = experience.rewards.to(device)
            rewards = base_rewards.clone()

            if exists(novelty_reward):
                bonus, epiplexities = novelty_reward(experience.state_latents, lens = episode_lens, return_epiplexity = True)
                rewards = rewards + bonus.to(device) * intrinsic_reward_scale
                epiplexity = epiplexities.mean().item()
                mean_bonus = bonus.mean().item()
            else:
                epiplexity = mean_bonus = 0.

            if existence_penalty != 0.:
                rewards = rewards + existence_penalty

            if death_penalty != 0.:
                terminated = experience.terminated.to(device)
                newly_done = terminated & ~cat((terminated.new_zeros(num_envs, 1), terminated[:, :-1]), dim = 1)
                rewards = rewards + death_penalty * newly_done.float()

            is_done = experience.terminated.to(device)
            discounts = torch.full((num_envs,), gamma, device = device)
            returns = calc_returns(rewards, is_done, torch.zeros(num_envs, device = device), discounts)

            experience = experience._replace(
                rewards = rewards.cpu(),
                base_reward = base_rewards.cpu(),
                returns = (discounts.cpu(), returns.cpu())
            )

            episode_unique, unique_cum = count_unique_frames(states, episode_lens, unique_seen)
            max_room = max(e.max_room for e in env.envs)
            rooms_seen = len({r for e in env.envs for r in e.rooms_seen})

            buffer = store_experience_in_replay_buffer(
                experience,
                max_episodes = max_episodes_buffer,
                max_timesteps = max_timesteps + 1,
                folder = buffer_folder,
                buffer = buffer,
                overwrite = (episode == 1)
            )

            if divisible_by(episode, train_every_eps):
                wm_loss, reward_loss, discount_loss = train_wm_epoch(
                    wm, buffer, optimizer, device,
                    batch_size, epochs_per_train, grad_accum_steps, train_window, batches_per_epoch
                )

            mean_steps = float(np.mean(episode_lens))
            total_reward = float(experience.cumulative_rewards.sum().item())

            row = dict(
                episode = episode,
                condition = condition,
                seed = seed,
                mean_steps = mean_steps,
                total_reward = total_reward,
                unique_new = episode_unique,
                unique_cum = unique_cum,
                max_room = max_room,
                rooms = rooms_seen,
                epiplexity = epiplexity,
                mean_bonus = mean_bonus,
                wm_loss = wm_loss,
                wm_reward_loss = reward_loss,
                wm_discount_loss = discount_loss
            )
            writer.writerow(row)
            f.flush()

            print(
                f'ep {episode:3d} [{condition:8s}] | steps {mean_steps:5.1f} | rew {total_reward:5.0f} '
                f'| unique {episode_unique:4d} (cum {unique_cum:5d}) | room {max_room:2d} '
                f'| epx {epiplexity:5.2f} | bonus {mean_bonus:.4f} | wm {wm_loss:.4f}',
                flush = True
            )

            if exists(wandb_mod):
                wandb_mod.log(row)

    if record_video and env.envs[0]._video:
        import imageio.v2 as imageio
        os.makedirs('montezuma-videos', exist_ok = True)
        video_path = f'./montezuma-videos/{condition}-s{seed}-ep{num_episodes}.mp4'
        imageio.mimsave(video_path, env.envs[0]._video, fps = video_fps, codec = 'libx264', quality = 8)
        print(f'wrote video -> {video_path}', flush = True)

    env.close()
    if exists(wandb_mod):
        wandb_mod.finish()

    print(f'done | condition={condition} | unique={unique_cum} | max_room={max_room}', flush = True)

if __name__ == '__main__':
    fire.Fire(main)
