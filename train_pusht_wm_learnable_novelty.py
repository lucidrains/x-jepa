# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "einops>=0.8.0",
#     "ema-pytorch>=0.8.3",
#     "fire",
#     "imageio[ffmpeg]",
#     "matplotlib",
#     "pillow",
#     "torch>=2.4",
#     "torch-einops-utils",
#     "x-mlps-pytorch",
# ]
# ///
#
# Push-T world model + CEM planner with LearnableNovelty bonus (arXiv:2607.18433)
# vs pure sparse-reward ablation.
#
#   python train_pusht_wm_learnable_novelty.py                                      # side-by-side vs ablation
#   python train_pusht_wm_learnable_novelty.py --side_by_side False                 # +LearnableNovelty only
#   python train_pusht_wm_learnable_novelty.py --side_by_side False --use_learnable_novelty False

from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import time

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import fire
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from einops import rearrange
from ema_pytorch import EMA
from torch_einops_utils import lens_to_mask
from x_mlps_pytorch import MLP

from env_faster_pusht import PushTEnv, render
from x_jepa import Agent, StateSpaceHeads
from x_jepa.intrinsic import LearnableNovelty
from x_jepa.state_heads import StateValue, n_step_returns
from x_jepa.utils import exists, default, divisible_by
from x_jepa.x_jepa import Transformer

# constants

STATE_DIM, ACTION_DIM = 6, 2
ARENA_CENTER, BLOCK_COG_OFFSET = 256., 45.
GOAL_ANGLE = float(np.pi / 4)
GOAL_COM = np.array([ARENA_CENTER - BLOCK_COG_OFFSET * np.sin(GOAL_ANGLE), ARENA_CENTER + BLOCK_COG_OFFSET * np.cos(GOAL_ANGLE)], dtype = np.float32)

# helpers

def sample_pusht_states(rng, num_envs, spawn_mode, min_dist, max_dist, max_angle):
    states = []
    for _ in range(num_envs):
        agent_x, agent_y = rng.integers(50, 450, size = 2).astype(np.float32)
        if spawn_mode == 'native':
            block_x, block_y = rng.integers(100, 400, size = 2).astype(np.float32)
            states.append(np.array([agent_x, agent_y, block_x, block_y, float(rng.uniform(-np.pi, np.pi))], dtype = np.float32))
        else:
            radial_angle = float(rng.uniform(0., 2. * np.pi))
            radial_dist = float(rng.uniform(min_dist, max_dist))
            offset = radial_dist * np.array([np.cos(radial_angle), np.sin(radial_angle)], dtype = np.float32)
            block_com = np.clip(GOAL_COM + offset, [110., 155.], [390., 435.])
            angle = float(GOAL_ANGLE + rng.uniform(-max_angle, max_angle))
            states.append(np.array([agent_x, agent_y, block_com[0], block_com[1] - BLOCK_COG_OFFSET, angle], dtype = np.float32))
    return np.stack(states)

def discounted_returns(rewards, gamma, terminated_at_end = False):
    returns = torch.zeros_like(rewards)
    running = 0.
    for step in reversed(range(len(rewards))):
        running = float(rewards[step]) + (0. if (terminated_at_end and step == len(rewards) - 1) else gamma * running)
        returns[step] = running
    return returns

# sparse push-t environment (raw 6d state, pure binary {0, 1} reward)

class SparsePushT:
    def __init__(self, num_envs, max_steps, step_scale, win_coverage, spawn_mode, min_dist, max_dist, max_angle, seed, fixed_states = None):
        self.env = PushTEnv(num_envs = num_envs, max_episode_steps = max_steps, auto_reset = False, solver = 'jacobi', solver_iterations = 4, device = 'cpu')
        self.num_envs, self.max_steps, self.step_scale, self.win_coverage = num_envs, max_steps, step_scale, float(win_coverage)
        self.spawn_kwargs = dict(spawn_mode = spawn_mode, min_dist = float(min_dist), max_dist = float(max_dist), max_angle = float(max_angle))
        self.rng, self.fixed_states = np.random.default_rng(seed), fixed_states

        self.done = torch.zeros(num_envs, dtype = torch.bool)
        self.won = torch.zeros(num_envs, dtype = torch.bool)
        self.contact_steps = torch.zeros(num_envs, dtype = torch.long)
        self.block_displacement = torch.zeros(num_envs)
        self.max_coverage = torch.zeros(num_envs)
        self.random_action = torch.zeros(num_envs, ACTION_DIM)
        self.last_obs, self.obs_history, self.trace = None, [], []

    def _encode_obs(self):
        theta = self.env.block_theta[:, None]
        return torch.cat((self.env.agent_pos / ARENA_CENTER - 1., self.env.block_p / ARENA_CENTER - 1., theta.cos(), theta.sin()), dim = -1)

    def reset(self, seed = None, options = None):
        if exists(options) and 'reset_to_state' in options:
            states = np.asarray(options['reset_to_state'], dtype = np.float32)
        else:
            states = self.fixed_states if exists(self.fixed_states) else sample_pusht_states(self.rng, self.num_envs, **self.spawn_kwargs)

        _, info = self.env.reset(seed = seed, options = dict(reset_to_state = states))
        if not exists(options) and not exists(self.fixed_states):
            for _ in range(10):
                solved = info['coverage'].float() >= self.win_coverage
                if not solved.any():
                    break
                states[solved.numpy()] = sample_pusht_states(self.rng, int(solved.sum()), **self.spawn_kwargs)
                _, info = self.env.reset(seed = seed, options = dict(reset_to_state = states))

        self.done.zero_()
        self.won.zero_()
        self.contact_steps.zero_()
        self.block_displacement.zero_()
        self.max_coverage = info['coverage'].float().clone()
        self.random_action.uniform_(-1., 1.)
        self.last_obs = self._encode_obs()
        self.obs_history = [self.last_obs.clone()]
        self.trace = [(self.env.agent_pos.clone(), self.env.block_pose.clone(), self.won.clone(), self.max_coverage.clone())]
        return self.last_obs

    def sample_correlated_action(self, step = 0):
        if divisible_by(step, 3):
            self.random_action.uniform_(-1., 1.)
        return self.random_action.clone()

    def step(self, action):
        action = torch.as_tensor(action, dtype = torch.float32).cpu().reshape(self.num_envs, ACTION_DIM)
        prev_obs, prev_agent, prev_block, prev_theta = self.last_obs, self.env.agent_pos.clone(), self.env.block_p.clone(), self.env.block_theta.clone()

        target_pos = (prev_agent + action.clamp(-1., 1.) * self.step_scale).clamp(18., 494.)
        _, _, _, truncated, info = self.env.step(torch.where(self.done[:, None], prev_agent, target_pos))

        self.env.agent_pos = torch.where(self.done[:, None], prev_agent, self.env.agent_pos)
        self.env.block_p = torch.where(self.done[:, None], prev_block, self.env.block_p)
        self.env.block_theta = torch.where(self.done, prev_theta, self.env.block_theta)

        active = ~self.done
        coverage = info['coverage'].float()
        terminated = coverage >= self.win_coverage
        just_won = terminated & active

        self.contact_steps += ((info['n_contacts'] > 0) & active).long()
        self.block_displacement += torch.where(active, (self.env.block_p - prev_block).norm(dim = -1), 0.)
        self.max_coverage = torch.maximum(self.max_coverage, torch.where(active, coverage, 0.))
        self.won |= just_won

        self.last_obs = torch.where(self.done[:, None], prev_obs, self._encode_obs())
        self.obs_history.append(self.last_obs.clone())
        self.done |= terminated | truncated
        self.trace.append((self.env.agent_pos.clone(), self.env.block_pose.clone(), self.won.clone(), coverage.clone()))
        info['final_obs'] = self.last_obs
        return self.last_obs, just_won.float(), terminated, truncated, info

    def snapshot_trace(self, max_envs = 16):
        num_saved = min(self.num_envs, max_envs)
        return {name: torch.stack([frame[idx][:num_saved] for frame in self.trace]) for idx, name in enumerate(('agent_pos', 'block_pose', 'won', 'coverage'))}

# trajectory learnable novelty built on x_jepa.intrinsic.LearnableNovelty

class TrajectoryLearnableNovelty:
    def __init__(self, state_dim, window_horizon, reservoir_dim, reservoir_layers, ridge_penalty, eta, novelty_scale, seed):
        with torch.random.fork_rng():
            torch.manual_seed(seed + 101)
            self.estimator = LearnableNovelty(dim_data = state_dim, mlp_reservoir_dim = reservoir_dim, mlp_reservoir_layers = reservoir_layers, ridge_penalty = ridge_penalty, eta = eta).eval()
        self.window_horizon, self.novelty_scale = window_horizon, novelty_scale
        self._batched_score = torch.vmap(self.estimator)

    def _window_pairs(self, trajectory):
        windows = trajectory.unfold(-2, self.window_horizon + 1, 1)
        return windows[..., 0], rearrange(windows[..., 1:], '... d horizon -> ... (horizon d)')

    @torch.no_grad()
    def score_trajectories(self, trajectories):
        orig_device = trajectories.device
        trajectories_cpu = trajectories.detach().cpu().float()
        if trajectories_cpu.shape[1] <= self.window_horizon + 1:
            return torch.zeros(trajectories_cpu.shape[0], device = orig_device)
        scores = self._batched_score(*self._window_pairs(trajectories_cpu))
        return torch.nan_to_num(scores, nan = 0., posinf = 0., neginf = 0.).to(orig_device)

    @torch.no_grad()
    def step_increments(self, trajectory):
        trajectory_cpu = trajectory.detach().cpu().float()
        num_steps = trajectory_cpu.shape[0] - 1
        cumulative = torch.zeros(num_steps + 1)
        for prefix_len in range(self.window_horizon + 2, num_steps + 2):
            score = self.estimator(*self._window_pairs(trajectory_cpu[:prefix_len]))
            prev = cumulative[prefix_len - 2].item()
            cumulative[prefix_len - 1] = torch.nan_to_num(score, nan = prev, posinf = prev, neginf = prev)
        increments = torch.nan_to_num((cumulative[1:] - cumulative[:-1]) / self.novelty_scale, nan = 0., posinf = 0., neginf = 0.)
        return increments, float(cumulative[-1])

# decoupled novelty critic heads & episode replay

class NoveltyHeads(nn.Module):
    def __init__(self, latent_dim, state_dim, action_dim, hidden_dim, ema_beta):
        super().__init__()
        self.value = StateValue(state_dim, hidden_dim)
        self.ema_value = EMA(self.value, beta = ema_beta, update_after_step = 0, update_every = 1)
        self.reward_head = MLP(latent_dim + action_dim, hidden_dim, hidden_dim, 1, activation = nn.SiLU())

    def update(self):
        self.ema_value.update()

    def predict_reward(self, state_latents, actions):
        return self.reward_head(torch.cat((state_latents, actions), dim = -1)).squeeze(-1)

    def loss(self, state_latents, states, actions, rewards, returns, dones, discounts, lens, n_step):
        td_targets = n_step_returns(states, returns, self.ema_value, discounts = discounts, lens = lens, n_step = n_step, rewards = rewards, dones = dones, value_scale = 1.)
        valid_mask = lens_to_mask(lens, states.shape[1]).float()
        value_loss = self.value.loss(states, td_targets, discounts = discounts, mask = valid_mask)
        steps = min(state_latents.shape[1], actions.shape[1], rewards.shape[1])
        pred_rewards = self.predict_reward(state_latents[:, :steps], actions[:, :steps])
        reward_loss = (F.smooth_l1_loss(pred_rewards, rewards[:, :steps], reduction = 'none') * valid_mask[:, :steps]).sum() / valid_mask[:, :steps].sum().clamp_min(1.)
        return value_loss + reward_loss

class EpisodeReplay:
    def __init__(self, capacity, seq_len, recent_capacity):
        self.episodes = deque(maxlen = capacity)
        self.seq_len, self.recent_capacity, self.num_successes = seq_len, recent_capacity, 0

    def add_batch(self, experience, final_obs, won_mask, novelty_scorer, gamma):
        full_states = torch.cat((experience.states.cpu(), final_obs.cpu()[:, None]), dim = 1)
        actions, episode_lens = experience.actions.cpu(), experience.episode_len.cpu().reshape(-1)
        epiplexities = []

        for idx in range(full_states.shape[0]):
            ep_len = min(max(int(episode_lens[idx]), 1), self.seq_len - 1)
            won = bool(won_mask[idx])
            win_pad = min(2, self.seq_len - (ep_len + 1)) if won else 0
            padded_len = ep_len + 1 + win_pad

            real_states = full_states[idx, :ep_len + 1]
            nov_rewards, epiplexity = novelty_scorer.step_increments(real_states)
            epiplexities.append(epiplexity)

            states = torch.zeros(self.seq_len, STATE_DIM)
            states[:ep_len + 1] = real_states
            if win_pad > 0:
                states[ep_len + 1 : padded_len] = real_states[-1:]

            ep_actions = torch.zeros(self.seq_len - 1, ACTION_DIM)
            ep_actions[:ep_len] = actions[idx, :ep_len]

            task_rewards, task_returns, dones = torch.zeros(self.seq_len), torch.zeros(self.seq_len), torch.zeros(self.seq_len, dtype = torch.bool)
            if won:
                task_rewards[ep_len - 1 : padded_len] = 1.
                dones[ep_len - 1 : padded_len] = True
                task_returns[:padded_len] = gamma ** (ep_len - torch.arange(padded_len, dtype = torch.float32)).clamp_min(0.)
                self.num_successes += 1

            padded_nov_rewards, padded_nov_returns = torch.zeros(self.seq_len), torch.zeros(self.seq_len)
            padded_nov_rewards[:ep_len] = nov_rewards[:ep_len]
            padded_nov_returns[:ep_len] = discounted_returns(nov_rewards[:ep_len], gamma, terminated_at_end = won)

            self.episodes.append(dict(
                states = states, actions = ep_actions, task_rewards = task_rewards, novelty_rewards = padded_nov_rewards,
                dones = dones, task_returns = task_returns, novelty_returns = padded_nov_returns,
                episode_lens = torch.tensor(padded_len, dtype = torch.long), won = won
            ))

        return epiplexities

    def sample(self, batch_size, success_frac, recent_frac, device):
        all_eps = list(self.episodes)
        recent_eps = all_eps[-self.recent_capacity:]
        success_eps = [ep for ep in all_eps if ep['won']]

        num_success = int(batch_size * success_frac) if success_eps else 0
        num_recent = int(batch_size * recent_frac)
        num_all = max(1, batch_size - num_success - num_recent)

        def draw(pool, count):
            return [] if count == 0 else [pool[int(i)] for i in torch.randint(0, len(pool), (count,))]

        chosen = draw(all_eps, num_all) + draw(recent_eps, num_recent) + draw(success_eps, num_success)
        keys = ('states', 'actions', 'task_rewards', 'novelty_rewards', 'dones', 'task_returns', 'novelty_returns', 'episode_lens')
        batch = {k: torch.stack([ep[k] for ep in chosen]).to(device) for k in keys}
        batch['non_success_mask'] = torch.cat((torch.ones(num_all + num_recent, dtype = torch.bool), torch.zeros(num_success, dtype = torch.bool))).to(device)
        return batch

# world model, CEM fitness & training

def build_models(dim, hidden_dim, depth, horizon, ema_beta, gamma, device):
    agent = Agent(
        state_encoder = MLP(STATE_DIM, hidden_dim, dim), action_encoder = MLP(ACTION_DIM, hidden_dim, dim),
        model = Transformer(dim = dim, depth = depth, causal = True), dim_action = ACTION_DIM,
        continuous_actions = True, transition_action_space = 'raw',
        state_linear_rnn_depth = 1, action_linear_rnn_depth = 1, value_loss_weight = 1.,
        discount_factor = gamma, ema_beta = ema_beta, add_reflexive_actor = False,
        temporal_straightening_loss_weight = 0.01, transition_horizon = horizon, state_transition_depth = 2
    ).to(device)
    task_heads = StateSpaceHeads(dim, STATE_DIM, hidden_dim, value_scale = 1., ema_beta = ema_beta).to(device)
    novelty_heads = NoveltyHeads(dim, STATE_DIM, ACTION_DIM, hidden_dim = hidden_dim, ema_beta = ema_beta).to(device)
    return agent, task_heads, novelty_heads

def make_planner_fitness(task_heads, novelty_heads, env, novelty_scorer, task_weight, novelty_weight, imagined_novelty_weight, gamma, history_len):
    def fitness(pred_state_latents, state_latents, actions, pred_rewards, pred_discounts, pred_values, discount_factor = None):
        batch_size, pop_size, horizon, _ = pred_state_latents.shape
        device, dtype = pred_state_latents.device, pred_state_latents.dtype
        effective_gamma = default(discount_factor, gamma)
        gamma_tensor = torch.as_tensor(effective_gamma, dtype = dtype, device = device).reshape(-1, 1, 1)

        decoded_states = task_heads.decode(pred_state_latents)
        future_discounts = gamma_tensor ** torch.arange(1, horizon + 1, dtype = dtype, device = device)
        total_score = torch.zeros((batch_size, pop_size), dtype = dtype, device = device)

        if task_weight > 0.:
            task_values = 0.5 * pred_values + 0.5 * task_heads.state_value(decoded_states, discount = effective_gamma)
            win_signal = (pred_rewards + (1. - pred_discounts)) * future_discounts
            discounted_values = task_values * future_discounts
            total_score = total_score + task_weight * (3. * (win_signal.mean(-1) + win_signal.max(-1).values) + discounted_values.mean(-1) + discounted_values[..., -1])

        if novelty_weight > 0.:
            novelty_values = novelty_heads.value(decoded_states, discount = effective_gamma)
            prev_latents = torch.cat((state_latents.unsqueeze(2), pred_state_latents[:, :, :-1]), dim = 2)
            step_discounts = gamma_tensor ** torch.arange(horizon, dtype = dtype, device = device)
            novelty_score = (novelty_heads.predict_reward(prev_latents, actions) * step_discounts).sum(-1) + (gamma_tensor[..., 0] ** horizon) * novelty_values[..., -1] + 0.25 * (novelty_values * step_discounts).mean(-1)

            if imagined_novelty_weight > 0. and len(env.obs_history) > 0:
                recent_history = torch.stack(env.obs_history[-history_len:], dim = 1).to(device = device, dtype = dtype)
                imagined_trajs = torch.cat((recent_history.unsqueeze(1).expand(batch_size, pop_size, -1, -1), decoded_states.clamp(-1., 1.)), dim = 2)
                future_epiplexity = novelty_scorer.score_trajectories(rearrange(imagined_trajs, 'b p t d -> (b p) t d')).reshape(batch_size, pop_size)
                history_epiplexity = novelty_scorer.score_trajectories(recent_history).unsqueeze(1)
                novelty_score = novelty_score + imagined_novelty_weight * (future_epiplexity - history_epiplexity) / novelty_scorer.novelty_scale

            total_score = total_score + novelty_weight * novelty_score

        return total_score
    return fitness

def train_world_model(agent, task_heads, novelty_heads, optimizer, replay, batch_size, success_frac, recent_frac, num_steps, horizon, n_step, td_weight, train_novelty_heads, device):
    agent.train()
    task_heads.train()
    novelty_heads.train()
    world_model = agent.get_world_model(0)
    all_params = list(agent.parameters()) + list(task_heads.parameters()) + list(novelty_heads.parameters())
    last_decoder_loss = 0.

    for _ in range(num_steps):
        batch = replay.sample(batch_size, success_frac, recent_frac, device)
        states, actions, lens = batch['states'], batch['actions'], batch['episode_lens']
        discounts = torch.empty(states.shape[0], device = device).uniform_(0.88, 0.97)

        task_td = task_heads.n_step_returns(states, batch['task_returns'], discounts, lens = lens, n_step = n_step, td_weight = td_weight, rewards = batch['task_rewards'], dones = batch['dones'])
        wm_loss, _, outputs = agent(states, actions, lens = lens, returns = (discounts, task_td), rewards = batch['task_rewards'], dones = batch['dones'], return_outputs = True)
        state_latents = outputs['state_latents_full']
        head_losses = task_heads.losses(state_latents, states, actions, returns = (discounts, task_td), rewards = batch['task_rewards'], dones = batch['dones'], lens = lens, world_model = world_model)

        # multi-step open-loop rollout supervision
        usable_steps = min(state_latents.shape[1] - 1, actions.shape[1])
        if usable_steps >= horizon:
            valid_mask = lens_to_mask(lens, states.shape[1]).float()
            action_windows = rearrange(actions[:, :usable_steps].unfold(1, horizon, 4), 'b w d h -> b w h d')
            target_windows = rearrange(states[:, 1 : usable_steps + 1].unfold(1, horizon, 4), 'b w d h -> b w h d')
            mask_windows = valid_mask[:, 1 : usable_steps + 1].unfold(1, horizon, 4)
            anchor_latents = state_latents[:, : usable_steps - horizon + 1 : 4].detach().unsqueeze(2)
            pred_deltas = world_model.state_transition_for_planning.predict(anchor_latents, action_windows)
            pred_states = task_heads.decode(world_model.apply_transition_residual(anchor_latents, pred_deltas))
            rollout_loss = (F.smooth_l1_loss(pred_states, target_windows, reduction = 'none').mean(-1) * mask_windows).sum() / mask_windows.sum().clamp_min(1.)
        else:
            rollout_loss = states.new_zeros(())

        total_loss = wm_loss + 20. * (head_losses['decoder'] + head_losses['decoder_next']) + 30. * rollout_loss + head_losses['value'] + head_losses['reward'] + head_losses['discount']
        if train_novelty_heads:
            non_success = batch['non_success_mask']
            total_loss = total_loss + novelty_heads.loss(state_latents[non_success].detach(), states[non_success], actions[non_success], batch['novelty_rewards'][non_success], batch['novelty_returns'][non_success], batch['dones'][non_success], discounts[non_success], lens[non_success], n_step = n_step)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.)
        optimizer.step()
        optimizer.zero_grad(set_to_none = True)
        agent.update()
        task_heads.update()
        if train_novelty_heads:
            novelty_heads.update()
        last_decoder_loss = float(head_losses['decoder'].detach())

    return last_decoder_loss

# visualization (macroblock-aligned side-by-side videos + curves)

def render_trace_grid(trace, cell_size, rows, cols):
    num_cells, num_frames = min(trace['agent_pos'].shape[1], rows * cols), trace['agent_pos'].shape[0]
    agent_pos, block_pose, won = trace['agent_pos'][:, :num_cells], trace['block_pose'][:, :num_cells], trace['won'][:, :num_cells]
    frames = render(agent_pos.reshape(-1, 2), block_pose.reshape(-1, 3), cell_size, cell_size, antialias = 2).reshape(num_frames, num_cells, cell_size, cell_size, 3)

    border = torch.zeros((cell_size, cell_size), dtype = torch.bool)
    border[:4, :] = border[-4:, :] = border[:, :4] = border[:, -4:] = True
    frames = torch.where((won[:, :, None, None] & border[None, None])[..., None], torch.tensor([30, 200, 70], dtype = torch.uint8), frames)
    return rearrange(frames, 't (r c) h w ch -> t (r h) (c w) ch', r = rows, c = cols).cpu().numpy(), won.cpu().numpy(), trace['coverage'][:, :num_cells].cpu().numpy()

def save_side_by_side_video(left_trace, right_trace, mp4_path, gif_path = None, title = '', fps = 12):
    mp4_path = Path(mp4_path)
    mp4_path.parent.mkdir(parents = True, exist_ok = True)
    num_available = min(left_trace['agent_pos'].shape[1], right_trace['agent_pos'].shape[1])
    rows, cols, cell_size = (4, 4, 96) if num_available >= 16 else (2, 2, 144)

    left_grid, left_won, left_cov = render_trace_grid(left_trace, cell_size, rows, cols)
    right_grid, right_won, right_cov = render_trace_grid(right_trace, cell_size, rows, cols)
    num_frames, panel_w, panel_h, divider_w, header_h, num_cells = min(len(left_grid), len(right_grid)), cols * cell_size, rows * cell_size, 16, 48, rows * cols

    frames = []
    for step in range(num_frames):
        canvas = Image.new('RGB', (panel_w * 2 + divider_w, panel_h + header_h), color = (18, 20, 26))
        canvas.paste(Image.fromarray(left_grid[step]), (0, header_h))
        canvas.paste(Image.fromarray(right_grid[step]), (panel_w + divider_w, header_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), f'{title}  |  step {step:2d}/{num_frames - 1}', fill = (235, 240, 250))
        draw.text((8, 26), f'+ LearnableNovelty: wins {int(left_won[step].sum())}/{num_cells}, cov {left_cov[step].mean():.2f}', fill = (90, 225, 145))
        draw.text((panel_w + divider_w + 8, 26), f'Ablation (No Novelty): wins {int(right_won[step].sum())}/{num_cells}, cov {right_cov[step].mean():.2f}', fill = (250, 140, 110))
        frames.append(np.asarray(canvas))

    imageio.mimsave(str(mp4_path), frames, fps = fps, codec = 'libx264', quality = 8)
    if exists(gif_path):
        imageio.mimsave(str(gif_path), frames[::2], fps = max(6, fps // 2), loop = 0)

def plot_comparison(novelty_history, ablation_history, out_path):
    fig, axes = plt.subplots(1, 3, figsize = (15., 4.4), facecolor = 'white')
    for history, label, color in ((novelty_history, '+ LearnableNovelty', '#1b9e77'), (ablation_history, 'Ablation (No Novelty)', '#d95f02')):
        eval_rows = [row for row in history if 'eval' in row]
        eval_eps = [row['episodes'] for row in eval_rows]
        train_eps = [row['episodes'] for row in history]
        contacts = np.array([row['contact_rate'] * 100. for row in history])
        kernel = min(5, len(contacts))

        axes[0].plot(train_eps, [row['rolling_sr'] * 100. for row in history], color = color, alpha = 0.25, lw = 1.3, ls = '--')
        axes[0].plot(eval_eps, [row['eval']['win_rate'] * 100. for row in eval_rows], color = color, marker = 'o', ms = 4.5, lw = 2.2, label = label)
        axes[1].plot(train_eps, contacts, color = color, alpha = 0.2, lw = 1.)
        axes[1].plot(train_eps, np.convolve(contacts, np.ones(kernel) / kernel, mode = 'same'), color = color, lw = 2.2, label = label)
        axes[2].plot(eval_eps, [row['eval']['mean_max_coverage'] for row in eval_rows], color = color, marker = '^', ms = 4.5, lw = 2.2, label = label)

    for ax, title, ylabel in zip(axes, ('Task Success Rate (Pure Task Eval)', 'Contact-Rich Exploration', 'Held-Out Mean Max Coverage'), ('Success Rate (%)', 'Steps in Contact (%)', 'Mean Max Coverage')):
        ax.set_title(title, fontsize = 10.5, fontweight = 'bold')
        ax.set_xlabel('Training Episodes')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha = 0.25)
        ax.legend(fontsize = 8.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylim(-2, 102)
    fig.suptitle('Push-T World Model + CEM Planner: LearnableNovelty vs Ablation', fontsize = 12, fontweight = 'bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi = 180, bbox_inches = 'tight')
    plt.close(fig)

# main (all hyperparameters live here)

def main(
    side_by_side = True,
    use_learnable_novelty = True,
    out_dir = './runs/pusht_wm_ln_side_by_side',
    seed = 0,
    device = 'mps' if torch.backends.mps.is_available() else 'cpu',
    num_threads = 4,
    spawn_mode = 'reachable',
    min_dist = 32.,
    max_dist = 92.,
    max_angle = 0.65,
    win_coverage = 0.72,
    num_envs = 8,
    eval_envs = 16,
    max_steps = 80,
    step_scale = 36.,
    dim = 96,
    hidden_dim = 192,
    depth = 2,
    lr = 1e-3,
    gamma = 0.93,
    ema_beta = 0.99,
    n_step = 8,
    td_weight = 0.25,
    batches = 160,
    random_batches = 6,
    pretrain_steps = 120,
    train_steps = 35,
    batch_size = 24,
    recent_capacity = 96,
    recent_frac = 0.25,
    success_frac = 0.25,
    horizon = 8,
    commit_k_steps = 2,
    population = 48,
    elites = 8,
    generations = 3,
    task_weight = 0.5,
    novelty_weight = 0.5,
    imagined_novelty_weight = 0.4,
    history_len = 20,
    window_horizon = 4,
    reservoir_dim = 32,
    reservoir_layers = 4,
    ridge_penalty = 0.03,
    eta = 1.,
    novelty_scale = 6.,
    eval_every = 10,
    num_video_checkpoints = 5,
    target_eval_sr = 0.875,
    target_streak = 2
):
    torch.set_num_threads(num_threads)
    out_path = Path(out_dir)
    out_path.mkdir(parents = True, exist_ok = True)

    spawn_kwargs = dict(spawn_mode = spawn_mode, min_dist = min_dist, max_dist = max_dist, max_angle = max_angle)
    eval_states = sample_pusht_states(np.random.default_rng(seed + 987_654), eval_envs, **spawn_kwargs)

    def run_experiment(enable_novelty, run_dir, max_batches, stop_sr):
        label = '+LearnableNovelty' if enable_novelty else 'Ablation(NoNov)'
        active_task_weight = float(task_weight) if enable_novelty else 1.
        active_novelty_weight = float(novelty_weight) if enable_novelty else 0.
        active_imagined_weight = float(imagined_novelty_weight) if enable_novelty else 0.

        run_dir = Path(run_dir)
        trace_dir = run_dir / 'traces'
        trace_dir.mkdir(parents = True, exist_ok = True)

        torch.manual_seed(seed)
        np.random.seed(seed)

        novelty_scorer = TrajectoryLearnableNovelty(STATE_DIM, window_horizon, reservoir_dim, reservoir_layers, ridge_penalty, eta, novelty_scale, seed)
        agent, task_heads, novelty_heads = build_models(dim, hidden_dim, depth, horizon, ema_beta, gamma, device)
        optimizer = Adam(list(agent.parameters()) + list(task_heads.parameters()) + list(novelty_heads.parameters()), lr = lr)

        train_env = SparsePushT(num_envs, max_steps, step_scale, win_coverage, seed = seed, **spawn_kwargs)
        eval_env = SparsePushT(eval_envs, max_steps, step_scale, win_coverage, seed = seed + 1, fixed_states = eval_states, **spawn_kwargs)
        replay = EpisodeReplay(max_batches * num_envs + 32, max_steps + 4, recent_capacity)

        rollout_discounts = torch.linspace(0.90, 0.96, num_envs, device = device)
        eval_discounts = torch.full((eval_envs,), gamma, device = device)
        plan_kwargs = dict(horizon = horizon, pop_size = population, generations = generations, elite_frac = elites / population, commit_k_steps = commit_k_steps, warm_start = True, cem_min_var = 0.08, use_ema_for_planning = False)
        eval_pop = max(population, 64)
        eval_plan_kwargs = dict(plan_kwargs, pop_size = eval_pop, generations = max(generations, 4), elite_frac = 8 / eval_pop, cem_min_var = 0.06)

        history, recent_wins, pretrained, best_eval_sr, streak = [], [], False, -1., 0
        run_start = time.time()

        for batch_idx in range(1, max_batches + 1):
            batch_start = time.time()
            is_warmup = batch_idx <= random_batches

            agent.eval()
            task_heads.eval()
            novelty_heads.eval()

            train_fitness = None if is_warmup else make_planner_fitness(task_heads, novelty_heads, train_env, novelty_scorer, active_task_weight, active_novelty_weight, active_imagined_weight, gamma, history_len)
            experience = agent.interact_with_environment(train_env, max_steps = max_steps, fitness_fn = train_fitness, action_fn = train_env.sample_correlated_action if is_warmup else None, discount_factor = rollout_discounts, **plan_kwargs)

            won_flags = train_env.won.cpu().numpy().copy()
            recent_wins.extend(won_flags.tolist())
            epiplexities = replay.add_batch(experience, train_env.last_obs, won_flags, novelty_scorer, gamma)

            if divisible_by(batch_idx, eval_every) or batch_idx == max_batches:
                torch.save(train_env.snapshot_trace(max_envs = 4), trace_dir / f'train_b{batch_idx:03d}.pt')

            decoder_loss = 0.
            if batch_idx >= random_batches:
                rolling_sr = float(np.mean(recent_wins[-64:]))
                adaptive_success_frac = min(success_frac, max(0.08, 1.5 * rolling_sr))
                decoder_loss = train_world_model(agent, task_heads, novelty_heads, optimizer, replay, batch_size, adaptive_success_frac, recent_frac, pretrain_steps if not pretrained else train_steps, horizon, n_step, td_weight, enable_novelty, device)
                pretrained = True

            record = dict(
                batch = batch_idx, episodes = batch_idx * num_envs, rolling_sr = float(np.mean(recent_wins[-64:])),
                total_wins = replay.num_successes, contact_rate = float((train_env.contact_steps.float() / max_steps).mean()),
                block_displacement_px = float(train_env.block_displacement.mean()), mean_max_coverage = float(train_env.max_coverage.mean()),
                epiplexity = float(np.mean(epiplexities)), decoder_loss = decoder_loss, elapsed_s = time.time() - batch_start
            )

            eval_summary = ''
            if pretrained and (divisible_by(batch_idx, eval_every) or batch_idx == max_batches):
                agent.eval()
                task_heads.eval()
                novelty_heads.eval()
                eval_fitness = make_planner_fitness(task_heads, novelty_heads, eval_env, novelty_scorer, 1., 0., 0., gamma, history_len)
                with torch.random.fork_rng():
                    torch.manual_seed(seed + 50_000)
                    agent.interact_with_environment(eval_env, max_steps = max_steps, fitness_fn = eval_fitness, discount_factor = eval_discounts, **eval_plan_kwargs)

                eval_metrics = dict(win_rate = float(eval_env.won.float().mean()), wins = int(eval_env.won.sum()), mean_max_coverage = float(eval_env.max_coverage.mean()))
                record['eval'] = eval_metrics
                best_eval_sr = max(best_eval_sr, eval_metrics['win_rate'])
                streak = (streak + 1) if eval_metrics['win_rate'] >= stop_sr else 0
                torch.save(eval_env.snapshot_trace(max_envs = eval_envs), trace_dir / f'eval_b{batch_idx:03d}.pt')
                eval_summary = f' || EVAL: sr={eval_metrics["win_rate"]:5.1%}({eval_metrics["wins"]}/{eval_envs}) cov={eval_metrics["mean_max_coverage"]:.2f} best={best_eval_sr:5.1%}'

            history.append(record)
            (run_dir / 'history.json').write_text(json.dumps(history, indent = 2))

            if batch_idx == 1 or divisible_by(batch_idx, 5) or 'eval' in record:
                print(f'[{label}] b{batch_idx:3d}/{max_batches} (ep {record["episodes"]:4d}) | sr64={record["rolling_sr"]:5.1%} wins={record["total_wins"]:3d} | cr={record["contact_rate"]:.2f} disp={record["block_displacement_px"]:5.1f}px ({record["elapsed_s"]:.1f}s){eval_summary}', flush = True)

            if streak >= target_streak:
                break

        print(f'[{label}] done in {(time.time() - run_start) / 60:.1f} min | best_eval_sr={best_eval_sr:.1%} | wins={replay.num_successes}', flush = True)
        return history

    if not side_by_side:
        subdir = out_path / ('with_ln' if use_learnable_novelty else 'ablation_no_ln')
        run_experiment(bool(use_learnable_novelty), subdir, max_batches = batches, stop_sr = target_eval_sr)
        return

    novelty_dir, ablation_dir = out_path / 'with_ln', out_path / 'ablation_no_ln'
    novelty_history = run_experiment(True, novelty_dir, max_batches = batches, stop_sr = target_eval_sr)
    ablation_history = run_experiment(False, ablation_dir, max_batches = len(novelty_history), stop_sr = 1.1)

    plot_comparison(novelty_history, ablation_history, out_path / 'comparison_curves.png')

    eval_batches = [row['batch'] for row in novelty_history if 'eval' in row]
    selected = [eval_batches[i] for i in sorted(set(int(round(k * (len(eval_batches) - 1) / (num_video_checkpoints - 1))) for k in range(num_video_checkpoints)))] if len(eval_batches) > num_video_checkpoints else eval_batches

    for batch_idx in selected:
        episodes = batch_idx * num_envs
        for split in ('eval', 'train'):
            left_path, right_path = novelty_dir / 'traces' / f'{split}_b{batch_idx:03d}.pt', ablation_dir / 'traces' / f'{split}_b{batch_idx:03d}.pt'
            if left_path.exists() and right_path.exists():
                save_side_by_side_video(
                    torch.load(left_path, map_location = 'cpu', weights_only = False),
                    torch.load(right_path, map_location = 'cpu', weights_only = False),
                    mp4_path = out_path / 'videos' / f'{split}_ep{episodes:04d}.mp4',
                    gif_path = out_path / 'videos' / f'{split}_ep{episodes:04d}.gif',
                    title = f'{split.title()} @ Ep {episodes} (Batch {batch_idx})'
                )

if __name__ == '__main__':
    fire.Fire(main)
