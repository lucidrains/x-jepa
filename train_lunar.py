# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate",
#     "einops>=0.8.0",
#     "fire",
#     "gymnasium[box2d]",
#     "memmap-replay-buffer",
#     "torch>=2.4",
#     "wandb>=0.25.1",
#     "moviepy"
# ]
# ///

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import fire
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.optim import Adam

from accelerate import Accelerator
import gymnasium as gym
import wandb

from torch_einops_utils import tree_map_tensor_to_device
from x_mlps_pytorch import MLP

from x_jepa import Agent
from x_jepa.x_jepa import Transformer
from x_jepa.rl import ppo_loss, tpo_loss
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

# RL training step helper

@dataclass
class RLLossStats:
    total_loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0

def train_rl_epoch(
    wm,
    buffer,
    rl_loss_fn,
    rl_optimizer,
    device,
    rl_batch_size,
    rl_epochs,
    rl_type,
    rl_update_state_encoder
):
    wm.train()
    dl = buffer.dataloader(batch_size = rl_batch_size, shuffle = True, drop_last = True)
    stats = RLLossStats()

    for _ in range(rl_epochs):
        for batch in dl:
            batch = tree_map_tensor_to_device(batch, device)
            rl_optimizer.zero_grad()

            if rl_type == 'ppo':
                loss, (pol, val, ent) = rl_loss_fn(
                    wm, batch,
                    actor_module = 'reflexive',
                    detach_base_state_encoder = (not rl_update_state_encoder),
                    return_breakdown = True
                )
                stats = RLLossStats(
                    total_loss = loss.item(),
                    policy_loss = pol.item(),
                    value_loss = val.item(),
                    entropy_loss = ent.item()
                )
            else:
                loss = rl_loss_fn(
                    wm, batch,
                    actor_module = 'reflexive',
                    detach_base_state_encoder = (not rl_update_state_encoder)
                )
                stats = RLLossStats(total_loss = loss.item())

            loss.backward()
            rl_optimizer.step()
            wm.update()

    return stats

# print helpers

def format_phase_str(use_rl, rl_type, is_ideal_goal):
    if use_rl:
        return f"[{rl_type.upper()} phase=1]"

    goal_type = 'ideal' if is_ideal_goal else 'random'
    return f"[WM phase=0 {goal_type}]"

def format_loss_str(use_rl, rl_type, rl_stats):
    if not use_rl:
        return ""

    if rl_type == 'ppo':
        return f" | ppo_loss: {rl_stats.total_loss:.4f} (pol: {rl_stats.policy_loss:.4f}, val: {rl_stats.value_loss:.4f}, ent: {rl_stats.entropy_loss:.4f})"

    return f" | {rl_type}_loss: {rl_stats.total_loss:.4f}"

# World model training step helper

def train_wm_epoch(
    wm,
    buffer,
    optimizer,
    device,
    batch_size,
    epochs_per_train,
    grad_accum_steps,
    use_rl
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

            loss, _ = wm(b_states, b_actions, lens = b_lens, returns = b_returns, behavior_clone = (not use_rl))
            (loss / grad_accum_steps).backward()

            if divisible_by(step + 1, grad_accum_steps):
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
    lr = 1e-3,
    rl_type = 'tpo',            # 'ppo' or 'tpo'
    rl_lr = 3e-4,
    rl_batch_size = 16,
    rl_start_episode = 1000,    # switch to RL policy after 1000 episodes
    rl_update_state_encoder = False,
    rl_epochs = None,
    num_episodes = 1500,
    train_every_eps = 10,
    epochs_per_train = 1,
    max_episodes_buffer = 400,
    max_timesteps = 1000,
    cem_num_samples = 64,
    cem_num_iters = 2,
    cem_elite_frac = 0.1,
    cem_horizon = 1,
    random_action_eps = 10,
    gamma = 0.99,
    reward_avg_window = 100,
    print_every_eps = 10,
    project_name = 'x-jepa-lunar-lander',
    record_video = False,
    use_wandb = True,
    cpu = False,
    q_filter_samples = 0,     # when > 0, seed the WM-phase planner with the reflexive actor, sampling this many candidate actions per step and keeping the highest-Q one (Q-Planning, Giridhar et al. arXiv 2608.21204)
    q_lr = 3e-4,
    q_epochs = 1,
    seed = 0
):
    assert rl_type in ('ppo', 'tpo'), f"rl_type must be either 'ppo' or 'tpo', got {rl_type}"
    rl_epochs = default(rl_epochs, 2 if rl_type == 'ppo' else 4)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = Accelerator(cpu = cpu).device
    print(f"using device: {device}", flush = True)

    if use_wandb:
        wandb.init(
            project = project_name,
            config = dict(
                rl_type = rl_type,
                rl_start_episode = rl_start_episode,
                rl_update_state_encoder = rl_update_state_encoder
            )
        )

    env = gym.make(env_name, render_mode = "rgb_array" if record_video else None)

    if record_video:
        os.makedirs("lunar-jepa-videos", exist_ok = True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder = "lunar-jepa-videos",
            episode_trigger = lambda ep: ep % 50 == 0,
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
        add_reflexive_actor = True
    ).to(device)

    optimizer = Adam(wm.parameters(), lr = lr)

    rl_loss_fn = ppo_loss if rl_type == 'ppo' else tpo_loss
    rl_params = wm.get_actor_parameters('reflexive')
    if rl_type == 'ppo':
        rl_params.extend(wm.get_value_parameters())
    if rl_update_state_encoder:
        rl_params.extend(wm.state_encoder.parameters())

    rl_optimizer = Adam(rl_params, lr = rl_lr)

    rewards_history = []
    buffer = None
    best_avg_reward = -float('inf')
    wm_loss_val = 0.0
    buffer_folder = f"./{env_name.lower().replace('-v3', '')}-memories"

    for episode in range(1, num_episodes + 1):
        use_rl = episode > rl_start_episode

        if episode == rl_start_episode + 1:
            print(f"\n=== Episode {episode}: Switching from World Model Planning to {rl_type.upper()} Actor Optimization ===\n", flush = True)

        is_ideal_goal = np.random.rand() < 0.5
        episode_goal = ideal_goal_state if is_ideal_goal else get_random_goal()
        goal_tensor = torch.from_numpy(episode_goal).float().unsqueeze(0).to(device)

        is_random_ep = episode <= random_action_eps

        wm.eval()

        seed_kwargs = dict()
        if q_filter_samples > 0 and not use_rl and not is_random_ep:
            seed_kwargs = dict(seed_with_actor = 'reflexive', value_filter_samples = q_filter_samples)

        experience = wm.interact_with_environment(
            env = env,
            max_steps = max_timesteps,
            goal_state = goal_tensor if (not is_random_ep and not use_rl) else None,
            fitness_fn = fitness_fn if not use_rl else None,
            action_fn = (lambda env: env.action_space.sample()) if is_random_ep else None,
            actor_module = 'reflexive' if use_rl else None,
            horizon = cem_horizon,
            pop_size = cem_num_samples,
            generations = cem_num_iters,
            elite_frac = cem_elite_frac,
            **seed_kwargs
        )

        total_reward = experience.cumulative_rewards.item()
        steps = experience.episode_len.item()

        buffer = store_experience_in_replay_buffer(
            experience,
            max_episodes = max_episodes_buffer,
            max_timesteps = max_timesteps + 1,
            folder = buffer_folder,
            buffer = buffer,
            overwrite = (episode == 1)
        )

        rl_stats = RLLossStats()

        if use_rl:
            rl_stats = train_rl_epoch(
                wm, buffer, rl_loss_fn, rl_optimizer, device,
                rl_batch_size, rl_epochs, rl_type, rl_update_state_encoder
            )

        q_loss_val = None

        if divisible_by(episode, train_every_eps):
            wm_loss_val = train_wm_epoch(
                wm, buffer, optimizer, device,
                batch_size, epochs_per_train, grad_accum_steps, use_rl
            )

            if q_filter_samples > 0:
                q_loss_val = wm.train_q(
                    buffer.get_all_data(),
                    steps = q_epochs * max(buffer.num_episodes // batch_size, 1),
                    batch_size = batch_size,
                    lr = q_lr,
                    discount = gamma
                )

        # metrics and logging

        track_reward = use_rl or is_ideal_goal
        avg_reward = 0.0

        if track_reward:
            rewards_history.append(total_reward)
            avg_reward = float(np.mean(rewards_history[-reward_avg_window:]))

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(wm.state_dict(), f"./{env_name.lower().replace('-v3', '')}-best.pt")

        log_dict = dict(
            episode = episode,
            episode_length = steps,
            phase = 1 if use_rl else 0,
            reward = total_reward
        )

        if q_loss_val is not None:
            log_dict['q_loss'] = q_loss_val

        if track_reward:
            log_dict.update({
                'avg_reward': avg_reward,
                f'avg{reward_avg_window}_reward': avg_reward
            })

        if use_rl:
            log_dict.update(
                rl_reward = total_reward,
                rl_avg_reward = avg_reward,
                rl_loss = rl_stats.total_loss,
                rl_policy_loss = rl_stats.policy_loss,
                rl_value_loss = rl_stats.value_loss,
                rl_entropy_loss = rl_stats.entropy_loss,
                rl_type = rl_type
            )
        elif is_ideal_goal:
            log_dict.update(ideal_goal_reward = total_reward, ideal_goal_avg_reward = avg_reward)
        else:
            log_dict.update(random_goal_reward = total_reward)

        if divisible_by(episode, print_every_eps):
            phase_str = format_phase_str(use_rl, rl_type, is_ideal_goal)
            loss_str = format_loss_str(use_rl, rl_type, rl_stats)
            avg_str = f" | avg{reward_avg_window}: {avg_reward:8.2f}" if track_reward else ""
            print(f"episode {episode:4d} {phase_str:20s} | reward: {total_reward:8.2f}{avg_str}{loss_str}", flush = True)

        if divisible_by(episode, train_every_eps):
            log_dict.update(train_loss = wm_loss_val)
            print(f"  --> trained world model | loss: {wm_loss_val:.4f}", flush = True)

        if use_wandb:
            wandb.log(log_dict)

if __name__ == '__main__':
    fire.Fire(main)
