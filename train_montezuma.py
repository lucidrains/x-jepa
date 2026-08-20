# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate",
#     "ale-py",
#     "einops>=0.8.0",
#     "fire",
#     "gymnasium>=1.0",
#     "memmap-replay-buffer",
#     "Pillow",
#     "torch>=2.4",
#     "wandb>=0.25.1"
# ]
# ///

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import csv
from collections import deque

import fire
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from PIL import Image

from accelerate import Accelerator
import gymnasium as gym
import ale_py
import wandb

from moviepy import ImageSequenceClip

from einops import einsum, reduce, rearrange
from torch_einops_utils import tree_map_tensor_to_device
from x_mlps_pytorch import MLP

from x_jepa import Agent
from x_jepa.x_jepa import Transformer, calc_returns
from x_jepa.intrinsic import CoinFlipNetwork
from x_jepa.utils import store_experience_in_replay_buffer, divisible_by, exists

# envs

class MontezumaEnv:
    """ale/montezuma, grayscale downsampled frames, flattened and stacked."""

    def __init__(self, seed, frame_size = 42, stack_frames = 2, max_episode_steps = 400, record = False):
        self.frame_size = frame_size
        self.stack_frames = stack_frames
        self.max_episode_steps = max_episode_steps
        self.record = record
        self.env = gym.make('ALE/MontezumaRevenge-v5', frameskip = 4, full_action_space = False)
        self.action_space = gym.spaces.Discrete(self.env.action_space.n)
        self.observation_space = gym.spaces.Box(-1., 1., (frame_size ** 2 * stack_frames,), dtype = np.float32)
        self.seed = seed
        self.room = 0
        self.max_room = 0
        self.rooms_seen = set()
        self.steps = 0
        self._frames = None
        self._video = []

    def _preprocess(self, rgb):
        img = Image.fromarray(rgb).convert('L').resize((self.frame_size, self.frame_size), Image.BILINEAR)
        return np.asarray(img, dtype = np.float32) / 255.0

    def _room(self):
        return int(self.env.unwrapped.ale.getRAM()[3]) + 1

    def _obs(self):
        return np.concatenate([f.ravel() for f in self._frames]).astype(np.float32)

    def reset(self, seed = None, options = None):
        self.seed = seed if seed is not None else self.seed
        obs, _ = self.env.reset(seed = self.seed)
        self.steps = 0
        self.room = self._room()
        self.max_room = self.room
        self.rooms_seen = {self.room}
        self._video = [obs.copy()] if self.record else []
        self._frames = deque([self._preprocess(obs)] * self.stack_frames, maxlen = self.stack_frames)
        return self._obs(), {}

    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(int(np.asarray(action).squeeze().item()))
        self.steps += 1
        self.room = self._room()
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
    """no auto-reset - done envs freeze so each env yields exactly one episode per rollout."""

    def __init__(self, seeds, max_episode_steps = 400, frame_size = 42, record = False):
        self.envs = [MontezumaEnv(seed = s, max_episode_steps = max_episode_steps, frame_size = frame_size, record = record) for s in seeds]
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self._done = [False] * self.num_envs
        self._last_obs = [None] * self.num_envs

    def reset(self, seed = None, options = None):
        self._done = [False] * self.num_envs
        obs = []
        for i, env in enumerate(self.envs):
            o, _ = env.reset(seed = seed)
            self._last_obs[i] = o
            obs.append(o)
        return np.stack(obs), {}

    def step(self, actions):
        actions = np.asarray(actions)
        if actions.ndim == 2 and actions.shape[0] == 1:
            actions = actions[0]

        obs, rewards, terminateds, truncateds = [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if self._done[i]:
                obs.append(self._last_obs[i])
                rewards.append(0.0)
                terminateds.append(True)
                truncateds.append(False)
                continue

            o, r, terminated, truncated, _ = env.step(action)
            self._last_obs[i] = o
            if terminated or truncated:
                self._done[i] = True

            obs.append(o)
            rewards.append(r)
            terminateds.append(terminated)
            truncateds.append(truncated)

        return np.stack(obs), np.array(rewards, dtype = np.float32), np.array(terminateds, dtype = bool), np.array(truncateds, dtype = bool), {}

    def close(self):
        for env in self.envs:
            env.close()

# coverage tracking

def frame_hash(obs_flat, frame_size):
    return np.round(obs_flat[-frame_size ** 2:] * 15).astype(np.uint8).tobytes()

def count_new_unique(states, episode_lens, frame_size, unique_seen):
    unique = 0
    for i, length in enumerate(episode_lens):
        hashes = {frame_hash(states[i, t], frame_size) for t in range(length)}
        unique += len(hashes)
        unique_seen.update(hashes)
    return unique, len(unique_seen)

# planning fitness - four ways:
#   'value' (old baseline): sum of per-step bootstrap values (risk-neutral mean over
#     value statistics for probabilistic value heads)
#   'value_risk': LCB sum of per-step (mean - risk_aversion * std) over the value
#     statistics - risk-averse planning prefers low-uncertainty values
#   'rollout' (stephen chung way): discounted rollout return with hard termination gating -
#     rewards after a predicted termination and the bootstrap both drop out; the continuation
#     head predicts per-step continuation, < 0.5 counts as an episode end
#   'rollout_risk': rollout return (risk-neutral mean over heads) minus risk_aversion times
#     the value-head std of the final-state bootstrap - goal-reaching via the dense reward
#     head, risk-aversion via value uncertainty

def _value_mean_std(pred_values):
    if pred_values.ndim == 4:
        mean = pred_values.mean(dim = -1)
        std = pred_values.std(dim = -1)
    else:
        mean, std = pred_values, None
    return mean, std

def make_fitness(mode = 'rollout', gamma = 0.99, risk_aversion = 0.0):
    if mode == 'value':
        def value_fitness(pred_values):
            mean, _ = _value_mean_std(pred_values)
            return reduce(mean, 'b p h -> b p', 'sum')
        return value_fitness

    if mode == 'value_risk':
        def value_risk_fitness(pred_values):
            mean, std = _value_mean_std(pred_values)
            score = mean if not exists(std) else mean - risk_aversion * std
            return reduce(score, 'b p h -> b p', 'sum')
        return value_risk_fitness

    assert mode in ('rollout', 'rollout_risk'), f'unknown fitness mode: {mode}'

    def rollout_fitness(pred_rewards, pred_discounts, pred_values):
        survived = torch.cumprod((pred_discounts >= 0.5).float(), dim = -1)
        alive = torch.cat((torch.ones_like(survived[..., :1]), survived[..., :-1]), dim = -1)
        horizon = pred_rewards.shape[-1]
        discounts = gamma ** torch.arange(horizon, device = pred_rewards.device)
        rollout_return = einsum(pred_rewards * alive, discounts, 'b p h, h -> b p')
        bootstrap_values, bootstrap_std = _value_mean_std(pred_values)
        bootstrap = bootstrap_values[:, :, -1] * (gamma ** horizon) * survived[:, :, -1]

        if mode == 'rollout_risk' and exists(bootstrap_std):
            bootstrap_risk = bootstrap_std[:, :, -1] * (gamma ** horizon) * survived[:, :, -1]
            bootstrap = bootstrap - risk_aversion * bootstrap_risk

        return rollout_return + bootstrap

    return rollout_fitness

# novelty bonus (Lobel et al. 2023) - coin-flip pseudocounts B(s) ~ 1/sqrt(N(s)), or knn distance
# to the nearest previously seen latent

def make_novelty(mode, dim, device):
    """returns (bonus_fn, update_fn); update stores pairs / trains after the bonus is consumed"""
    if mode == 'knn':
        ref = torch.empty(0, dim, device = device)

        def bonus_fn(latents):
            if ref.shape[0] == 0:
                return torch.zeros(latents.shape[0], device = latents.device)
            sq = (latents ** 2).sum(-1, keepdim = True) - 2 * latents @ ref.t() + (ref ** 2).sum(-1).t()
            return sq.clamp(min = 0).sqrt().min(dim = -1).values

        def update_fn(latents):
            nonlocal ref
            ref = torch.cat((ref, latents))
            if ref.shape[0] > 4096:
                ref = ref[torch.randperm(ref.shape[0], device = ref.device)[:4096]]

        return bonus_fn, update_fn

    cfn = CoinFlipNetwork(net = MLP(dim, dim * 2, dim), dim = dim).to(device)
    optimizer = Adam(cfn.parameters(), lr = 3e-4)

    def bonus_fn(latents):
        return cfn.compute_bonus(latents).float().cpu()

    def update_fn(latents):
        cfn.train()
        for _ in range(48):
            loss = cfn.compute_loss(latents, batch_size = 512)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        cfn.eval()

    return bonus_fn, update_fn

# world model training

def train_wm_epoch(wm, buffer, optimizer, device, batch_size, epochs_per_train, grad_accum_steps, train_window, batches_per_epoch):
    dl = buffer.dataloader(
        batch_size = batch_size,
        shuffle = True,
        n_steps = train_window,
        sequence_fields = ('states', 'actions', 'returns', 'rewards', 'dones', 'base_rewards'),
        fieldname_map = {'seq_states': 'states', 'seq_actions': 'actions', 'seq_returns': 'returns', 'seq_rewards': 'rewards', 'seq_dones': 'dones', 'seq_base_rewards': 'base_rewards'}
    )
    dl_iter = iter(dl)
    wm.train()
    loss_val = 0.0
    reward_loss_val = 0.0
    discount_loss_val = 0.0
    total_batches = epochs_per_train * batches_per_epoch

    for step in range(total_batches):
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)

        batch = tree_map_tensor_to_device(batch, device)
        states, actions, returns, base_rewards, dones = batch['states'], batch['actions'], batch['returns'], batch['base_rewards'], batch['dones']
        lens = batch.get('n_step_lens', batch.get('episode_len', batch.get('episode_lens')))

        loss, loss_breakdown = wm(states, actions, lens = lens, returns = returns, rewards = base_rewards, dones = dones, behavior_clone = True)
        (loss / grad_accum_steps).backward()

        if divisible_by(step + 1, grad_accum_steps) or step + 1 == total_batches:
            optimizer.step()
            optimizer.zero_grad()
            wm.update()
            loss_val = loss.item()
            reward_loss_val = loss_breakdown.reward_pred.item()
            discount_loss_val = loss_breakdown.discount_pred.item()

    return loss_val, reward_loss_val, discount_loss_val

# main

def main(
    dim = 32,
    hidden_dim = 256,
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
    random_action_eps = 10, # warmup episodes with random actions, so the model trains before planning
    gamma = 0.99,
    use_intrinsic = True,
    novelty = 'coinflip',
    fitness_mode = 'rollout', # 'rollout' = reward + continuation heads with termination gating, 'value' = sum of predicted values, 'value_risk' = LCB over probabilistic values
    risk_aversion = 0.0, # LCB penalty for 'value_risk' fitness - higher = more risk-averse planning
    intrinsic_reward_scale = 1.0, # novelty bonus shaping for the value critic returns
    existence_penalty = -0.01, # per-step penalty, gives the reward head a stationary signal
    death_penalty = -6.0, # terminal penalty, makes dying states bad for the fitness
    seed_with_actor_after = 25, # seed the planner with the reflexive actor once it is trained - anchors the search so novel-state exploration cannot run away
    print_every_eps = 1,
    seed = 0,
    tag = '',
    frame_size = 42,
    record_video = False,
    record_env_index = 0,
    video_fps = 15,
    project_name = 'x-jepa-montezuma',
    use_wandb = False,
    cpu = False,
    start_episode = 1
):
    device = Accelerator(cpu = cpu).device
    print(f'using device: {device}', flush = True)

    use_intrinsic = str(use_intrinsic).lower() == 'true' if isinstance(use_intrinsic, str) else bool(use_intrinsic)
    use_wandb = str(use_wandb).lower() == 'true' if isinstance(use_wandb, str) else bool(use_wandb)
    record_video = str(record_video).lower() == 'true' if isinstance(record_video, str) else bool(record_video)
    fitness_mode = str(fitness_mode).lower()
    novelty = str(novelty).lower()
    start_episode = int(start_episode)

    condition = 'intrinsic' if use_intrinsic else 'baseline'
    suffix = f'-{tag}' if tag else f'-{condition}-s{seed}'
    print(f'condition: {condition} | seed: {seed} | fitness: {fitness_mode} | novelty: {novelty}', flush = True)

    if use_wandb:
        wandb.init(project = project_name, config = dict(condition = condition, seed = seed, use_intrinsic = use_intrinsic))

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = BatchedEnv([seed * 1000 + i for i in range(num_envs)], max_episode_steps = max_timesteps, frame_size = frame_size, record = record_video)
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    wm = Agent(
        state_encoder = MLP(state_dim, *((hidden_dim,) * (depth - 1)), dim),
        action_encoder = nn.Embedding(action_dim, dim),
        action_decoder = nn.Linear(dim, action_dim),
        model = Transformer(dim = dim, depth = depth, causal = True),
        dim_action = action_dim,
        continuous_actions = False,
        transition_action_space = 'local',
        state_linear_rnn_depth = 1,
        action_linear_rnn_depth = 1,
        value_loss_weight = 1.0,
        discount_factor = gamma,
        add_reflexive_actor = True
    ).to(device)

    optimizer = Adam(wm.parameters(), lr = lr)

    novelty_bonus = None
    novelty_update = None
    if use_intrinsic:
        novelty_bonus, novelty_update = make_novelty(novelty, dim, device)

    fitness_fn = make_fitness(fitness_mode, gamma, risk_aversion)

    buffer = None
    wm_loss_val = 0.0
    wm_reward_loss_val = 0.0
    wm_discount_loss_val = 0.0
    buffer_folder = f'./montezuma-memories{suffix}'
    csv_path = f'./montezuma-{tag}-{condition}-s{seed}.csv' if tag else f'./montezuma-{condition}-s{seed}.csv'
    unique_seen = set()
    resuming = start_episode > 1

    if resuming:
        with open(csv_path) as f:
            prev_rows = list(csv.DictReader(f))
        prev_unique = int(prev_rows[-1]['unique_cum']) if prev_rows else 0
        unique_seen = {f'prev-{i}' for i in range(prev_unique)}

    with open(csv_path, 'a' if resuming else 'w', newline = '') as f:
        writer = csv.DictWriter(f, fieldnames = ['episode', 'condition', 'seed', 'mode', 'mean_steps', 'total_reward', 'unique_new', 'unique_cum', 'max_room', 'rooms', 'mean_bonus', 'bonus_std', 'wm_loss', 'wm_reward_loss', 'wm_discount_loss'])
        if not resuming:
            writer.writeheader()

        for episode in range(start_episode, num_episodes + 1):
            wm.eval()

            is_random_ep = episode <= random_action_eps

            experience = wm.interact_with_environment(
                env = env,
                max_steps = max_timesteps,
                fitness_fn = None if is_random_ep else fitness_fn,
                action_fn = (lambda env: np.asarray([env.action_space.sample() for _ in range(env.num_envs)])[:, None]) if is_random_ep else None,
                horizon = cem_horizon,
                pop_size = cem_num_samples,
                generations = cem_num_iters,
                elite_frac = cem_elite_frac,
                commit_k_steps = commit_k_steps,
                seed_with_actor = 'reflexive' if (not is_random_ep and seed_with_actor_after is not None and episode >= seed_with_actor_after) else None
            )

            # discrete random actions come back with a trailing unit dim

            if experience.actions.ndim == 3 and experience.actions.shape[-1] == 1:
                experience = experience._replace(actions = rearrange(experience.actions, 'b n 1 -> b n'))

            states = experience.states.cpu().numpy()
            episode_lens = experience.episode_len.tolist()
            total_rewards = experience.cumulative_rewards.tolist()

            # reward shaping: novelty bonus + existence penalty + death penalty -> shaped rewards ->
            # returns for the value critic. the reward head keeps training on the base (unshaped)
            # rewards - the planner must maximize a stationary reward function, not a moving novelty
            # proxy, or the two form an exploitation loop

            if use_intrinsic:
                with torch.no_grad():
                    latents = rearrange(experience.state_latents, 'b n d -> (b n) d').to(device)
                    bonus = novelty_bonus(latents).to(device)
                    bonus = rearrange(bonus, '(b n) -> b n', b = num_envs)
                    mean_bonus = bonus.mean().item()
                    bonus_std = bonus.std().item()

                rewards = experience.rewards.to(device) + bonus * intrinsic_reward_scale
            else:
                mean_bonus = 0.0
                bonus_std = 0.0
                rewards = experience.rewards.to(device)

            if existence_penalty != 0.0:
                rewards = rewards + existence_penalty

            if death_penalty != 0.0:
                terminated = experience.terminated.to(device)
                newly_done = terminated & ~torch.cat((terminated.new_zeros(num_envs, 1), terminated[:, :-1]), dim = 1)
                rewards = rewards + death_penalty * newly_done.float()

            is_done = experience.terminated.to(device)
            discounts = torch.full((num_envs,), gamma, device = device)
            returns = calc_returns(rewards, is_done, torch.zeros(num_envs, device = device), discounts)
            experience = experience._replace(rewards = rewards.cpu(), returns = (discounts.cpu(), returns.cpu()))

            if use_intrinsic:
                novelty_update(latents)  # store pairs after shaping

            episode_unique, unique_cum = count_new_unique(states, episode_lens, frame_size, unique_seen)
            max_room = max((env.max_room for env in env.envs), default = 1)
            rooms_seen = len({room for env in env.envs for room in env.rooms_seen})

            buffer = store_experience_in_replay_buffer(
                experience,
                max_episodes = max_episodes_buffer,
                max_timesteps = max_timesteps + 1,
                folder = buffer_folder,
                buffer = buffer,
                overwrite = (not resuming and episode == 1)
            )

            if divisible_by(episode, train_every_eps):
                wm_loss_val, wm_reward_loss_val, wm_discount_loss_val = train_wm_epoch(wm, buffer, optimizer, device, batch_size, epochs_per_train, grad_accum_steps, train_window, batches_per_epoch)

            mean_steps = float(np.mean(episode_lens))
            total_reward = float(np.sum(total_rewards))

            row = dict(episode = episode, condition = condition, seed = seed, mode = fitness_mode, mean_steps = mean_steps, total_reward = total_reward, unique_new = episode_unique, unique_cum = unique_cum, max_room = max_room, rooms = rooms_seen, mean_bonus = mean_bonus, bonus_std = bonus_std, wm_loss = wm_loss_val, wm_reward_loss = wm_reward_loss_val, wm_discount_loss = wm_discount_loss_val)
            writer.writerow(row)
            f.flush()

            if divisible_by(episode, print_every_eps):
                print(f'episode {episode:3d} [{condition:9s} {fitness_mode:9s}] | steps: {mean_steps:6.1f} | reward: {total_reward:6.0f} | unique: {episode_unique:4d} (cum {unique_cum:5d}) | max_room: {max_room:2d} | bonus: {mean_bonus:.4f} +/- {bonus_std:.4f} | wm_loss: {wm_loss_val:.4f} | wm_reward_loss: {wm_reward_loss_val:.4f} | wm_discount_loss: {wm_discount_loss_val:.4f}', flush = True)

            if use_wandb:
                wandb.log(row)

    # replay video of the last rollout
    if record_video and 0 <= record_env_index < num_envs:
        frames = env.envs[record_env_index]._video
        if frames:
            os.makedirs('montezuma-videos', exist_ok = True)
            video_path = f'./montezuma-videos/{condition}-s{seed}-ep{num_episodes}.mp4'
            print(f'writing replay video ({len(frames)} frames) -> {video_path}', flush = True)
            clip = ImageSequenceClip(frames, fps = video_fps)
            clip.write_videofile(video_path, codec = 'libx264', preset = 'veryfast', audio = False, logger = None)
            clip.close()
            if use_wandb:
                wandb.log({'video_replay': wandb.Video(video_path, fps = video_fps, caption = f'{condition} s{seed} final episode')})

    env.close()

    if use_wandb:
        wandb.finish()

    print(f'\ndone! condition={condition} seed={seed} | cumulative unique states: {unique_cum} | max room: {max_room}', flush = True)

if __name__ == '__main__':
    fire.Fire(main)
