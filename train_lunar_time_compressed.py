# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate",
#     "einops>=0.8.0",
#     "ema-pytorch>=0.8.3",
#     "fire",
#     "gymnasium[box2d]",
#     "memmap-replay-buffer",
#     "torch>=2.4",
#     "wandb>=0.25.1",
#     "moviepy"
# ]
# ///

import os
import shutil
from pathlib import Path
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import fire
import numpy as np

import torch
from torch import tensor
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from accelerate import Accelerator
import gymnasium as gym
import wandb

from ema_pytorch import EMATensor
from torch_einops_utils import tree_map_tensor_to_device
from x_mlps_pytorch import MLP

from x_jepa import Agent
from x_jepa.x_jepa import Transformer, round_up
from x_jepa.utils import (
    store_experience_in_replay_buffer,
    divisible_by,
    exists,
    default
)

# goals

def get_random_goal():
    return np.array([
        np.random.uniform(-0.8, 0.8), # x
        np.random.uniform(0.0, 1.2),  # y
        np.random.uniform(-0.5, 0.5), # vx
        np.random.uniform(-0.5, 0.5), # vy
        np.random.uniform(-0.5, 0.5), # angle
        np.random.uniform(-0.5, 0.5), # angular velocity
        np.random.choice([0.0, 1.0]), # left leg
        np.random.choice([0.0, 1.0])  # right leg
    ], dtype = np.float32)

ideal_goal_state = np.zeros(8, dtype = np.float32)
ideal_goal_state[-2:] = 1.0

# fitness

def fitness_fn(pred_next_encoded_states, encoded_goal, pred_values):
    dist_cost = F.smooth_l1_loss(
        pred_next_encoded_states,
        encoded_goal.expand_as(pred_next_encoded_states),
        reduction = 'none'
    ).sum(dim = -1)
    return (-dist_cost + 5.0 * pred_values).sum(dim = -1)

# helpers

def calc_curriculum_horizon(
    wm_loss_ema,
    min_horizon = 2,
    max_horizon = 10,
    high_loss_thresh = 40.0,
    low_loss_thresh = 15.0,
    temporal_compression = 2
):
    if not exists(wm_loss_ema) or wm_loss_ema >= high_loss_thresh:
        return min_horizon

    if wm_loss_ema <= low_loss_thresh:
        return max_horizon

    frac = (high_loss_thresh - wm_loss_ema) / (high_loss_thresh - low_loss_thresh)
    raw_horizon = min_horizon + frac * (max_horizon - min_horizon)
    return round_up(int(raw_horizon), temporal_compression)

# World model training step helper

def train_wm_epoch(
    wm,
    buffer,
    optimizer,
    device,
    batch_size,
    epochs_per_train,
    grad_accum_steps,
    max_grad_norm = 1.0
):
    dl = buffer.dataloader(batch_size = batch_size, shuffle = True)
    wm.train()
    loss_val = 0.0

    for _ in range(epochs_per_train):
        optimizer.zero_grad()
        for step, batch in enumerate(dl):
            batch = tree_map_tensor_to_device(batch, device)
            b_states, b_actions, b_returns = batch['states'], batch['actions'], batch['returns']
            b_lens = batch.get('episode_len', batch.get('episode_lens'))

            loss, _ = wm(b_states, b_actions, lens = b_lens, returns = b_returns, behavior_clone = True)
            (loss / grad_accum_steps).backward()

            if divisible_by(step + 1, grad_accum_steps):
                if exists(max_grad_norm) and max_grad_norm > 0:
                    clip_grad_norm_(wm.parameters(), max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()
                wm.update()
                loss_val = loss.item()

    return loss_val

# main

def main(
    env_name = 'LunarLanderContinuous-v3',
    dim = 64,
    hidden_dim = 256,
    depth = 2,
    batch_size = 16,
    grad_accum_steps = 2,
    lr = 5e-4,
    max_grad_norm = 1.0,
    wm_loss_ema_beta = 0.9,
    temporal_compression = 2,
    curriculum_cem_horizon = True,
    cem_min_horizon = 2,
    cem_max_horizon = 10,
    high_loss_thresh = 40.0,
    low_loss_thresh = 15.0,
    num_episodes = 5000,
    train_every_eps = 5,
    epochs_per_train = 2,
    max_episodes_buffer = 400,
    max_timesteps = 1000,
    cem_num_samples = 256,
    cem_num_iters = 4,
    cem_elite_frac = 0.125,
    random_action_eps = 32,
    start_train_eps = 32,
    gamma = 0.99,
    reward_avg_window = 100,
    print_every_eps = 10,
    record_every_eps = 50,
    project_name = 'x-jepa-lunar-time-compressed',
    record_video = False,
    use_wandb = False,
    cpu = False
):
    device = Accelerator(cpu = cpu).device
    print(f"using device: {device}", flush = True)

    if use_wandb:
        wandb.init(
            project = project_name,
            config = dict(
                temporal_compression = temporal_compression,
                curriculum_cem_horizon = curriculum_cem_horizon,
                cem_min_horizon = cem_min_horizon,
                cem_max_horizon = cem_max_horizon,
                high_loss_thresh = high_loss_thresh,
                low_loss_thresh = low_loss_thresh,
                wm_loss_ema_beta = wm_loss_ema_beta,
                batch_size = batch_size,
                start_train_eps = start_train_eps,
                lr = lr
            )
        )

    env = gym.make(env_name, render_mode = "rgb_array" if record_video else None)

    if record_video:
        video_folder = Path("lunar-jepa-videos-time-compressed")
        if video_folder.exists():
            shutil.rmtree(video_folder)
        video_folder.mkdir(parents = True, exist_ok = True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder = str(video_folder),
            episode_trigger = lambda ep: divisible_by(ep, record_every_eps),
            disable_logger = True
        )

    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])

    state_encoder = MLP(state_dim, *((hidden_dim,) * (depth - 1)), dim)
    action_encoder = MLP(action_dim, *((hidden_dim,) * (depth - 1)), dim)

    model = Transformer(dim = dim, depth = depth, causal = True)

    wm = Agent(
        state_encoder = state_encoder,
        action_encoder = action_encoder,
        model = model,
        dim_action = action_dim,
        continuous_actions = True,
        transition_action_space = 'raw',
        state_linear_rnn_depth = 1,
        action_linear_rnn_depth = 1,
        value_loss_weight = 1.0,
        discount_factor = gamma,
        temporal_compression = temporal_compression
    ).to(device)

    optimizer = Adam(wm.parameters(), lr = lr)

    rewards_history = []
    buffer = None
    best_avg_reward = -float('inf')
    wm_loss_val = 0.0

    wm_loss_tensor = tensor(0.0)
    wm_loss_ema = EMATensor(wm_loss_tensor, beta = wm_loss_ema_beta, update_after_step = 0, update_every = 1)

    env_slug = env_name.lower().replace('-v3', '')
    buffer_folder = Path(f"./{env_slug}-time-compressed-memories")
    best_model_path = Path(f"./{env_slug}-time-compressed-best.pt")

    for episode in range(1, num_episodes + 1):
        is_ideal_goal = np.random.rand() < 0.5
        episode_goal = ideal_goal_state if is_ideal_goal else get_random_goal()
        goal_tensor = torch.from_numpy(episode_goal).float().unsqueeze(0).to(device)

        is_random_ep = episode <= random_action_eps

        effective_horizon = cem_max_horizon

        if curriculum_cem_horizon:
            current_wm_loss_ema = wm_loss_ema.ema_model.item() if wm_loss_ema.step.item() > 0 else None
            effective_horizon = calc_curriculum_horizon(
                current_wm_loss_ema,
                min_horizon = cem_min_horizon,
                max_horizon = cem_max_horizon,
                high_loss_thresh = high_loss_thresh,
                low_loss_thresh = low_loss_thresh,
                temporal_compression = temporal_compression
            )

        wm.eval()

        experience = wm.interact_with_environment(
            env = env,
            max_steps = max_timesteps,
            goal_state = goal_tensor if not is_random_ep else None,
            fitness_fn = fitness_fn if not is_random_ep else None,
            action_fn = (lambda env: env.action_space.sample()) if is_random_ep else None,
            horizon = effective_horizon,
            pop_size = cem_num_samples,
            generations = cem_num_iters,
            elite_frac = cem_elite_frac,
            commit_k_steps = 2
        )

        total_reward = experience.cumulative_rewards.item()
        steps = experience.episode_len.item()

        buffer = store_experience_in_replay_buffer(
            experience,
            max_episodes = max_episodes_buffer,
            max_timesteps = max_timesteps + 1,
            folder = str(buffer_folder),
            buffer = buffer,
            overwrite = (episode == 1)
        )

        should_train = episode >= start_train_eps and divisible_by(episode, train_every_eps)

        if should_train:
            wm_loss_val = train_wm_epoch(
                wm, buffer, optimizer, device,
                batch_size, epochs_per_train, grad_accum_steps, max_grad_norm
            )

            wm_loss_tensor.copy_(tensor(wm_loss_val))
            wm_loss_ema.update()

        # metrics and logging

        track_reward = is_ideal_goal
        avg_reward = 0.0

        if track_reward:
            rewards_history.append(total_reward)
            avg_reward = float(np.mean(rewards_history[-reward_avg_window:]))

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(wm.state_dict(), best_model_path)

        log_dict = dict(
            episode = episode,
            episode_length = steps,
            reward = total_reward,
            effective_cem_horizon = effective_horizon
        )

        if track_reward:
            log_dict.update({
                'avg_reward': avg_reward,
                f'avg{reward_avg_window}_reward': avg_reward
            })

        if is_ideal_goal:
            log_dict.update(ideal_goal_reward = total_reward, ideal_goal_avg_reward = avg_reward)
        else:
            log_dict.update(random_goal_reward = total_reward)

        if divisible_by(episode, print_every_eps):
            goal_type = 'ideal' if is_ideal_goal else 'random'
            phase_str = f"[WM compress2 {goal_type}]"
            avg_str = f" | avg{reward_avg_window}: {avg_reward:8.2f}" if track_reward else ""
            horizon_str = f" | horizon: {effective_horizon:2d}"
            print(f"episode {episode:4d} {phase_str:25s} | reward: {total_reward:8.2f}{avg_str}{horizon_str}", flush = True)

        if should_train:
            ema_val = wm_loss_ema.ema_model.item()
            log_dict.update(wm_loss = wm_loss_val, wm_loss_ema = ema_val)
            print(f"  --> trained time compressed world model | wm_loss: {wm_loss_val:.4f} (ema: {ema_val:.4f})", flush = True)

        if use_wandb:
            wandb.log(log_dict)

    env.close()

if __name__ == '__main__':
    fire.Fire(main)
