# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate",
#     "einops>=0.8.0",
#     "fire",
#     "gymnasium",
#     "memmap-replay-buffer",
#     "torch>=2.4"
# ]
# ///
#
# continuous cartpole, controlled purely by latent-space CEM planning through a
# world model with configurable temporal compression
#
#   python train_cartpole_hierarchical.py                              # compression 1
#   python train_cartpole_hierarchical.py --temporal_compression 2
#   python train_cartpole_hierarchical.py --temporal_compression 2 --hierarchical true
#
# compression = c means the world model predicts on a lane of every c-th timestep,
# with the c raw actions of each chunk mean-pooled into a single decision. the plan
# horizon is measured in raw steps and must be divisible by the compression.
#
# `--hierarchical true` stacks a coarse world model (compression c) with a fine one
# (compression 1): the coarse plan's actions seed the fine plan.
#
# sweep the compression x horizon grid with cartpole_compression_sweep.py

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import fire
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from accelerate import Accelerator

from torch_einops_utils import tree_map_tensor_to_device
from x_mlps_pytorch import MLP

from x_jepa import Agent
from x_jepa.x_jepa import Transformer
from x_jepa.utils import store_experience_in_replay_buffer, divisible_by

from env_cartpole import BatchedEnv

# planning fitness - reward reaching and staying at the upright state (goal = all zeros)

def make_fitness():
    def fitness_fn(pred_next_encoded_states, encoded_goal):
        dist_cost = F.smooth_l1_loss(
            pred_next_encoded_states,
            encoded_goal.expand_as(pred_next_encoded_states),
            reduction = 'none'
        ).sum(dim = -1)
        return -dist_cost.sum(dim = -1)

    return fitness_fn

# world model training - one pass per world model index (compressed and fine lanes)

def train_wm_epoch(wm, buffer, optimizer, device, batch_size, epochs_per_train, grad_accum_steps, train_window, batches_per_epoch, world_model_indices = (0,)):
    dl = buffer.dataloader(
        batch_size = batch_size,
        shuffle = True,
        n_steps = train_window,
        sequence_fields = ('states', 'actions', 'returns'),
        fieldname_map = {'seq_states': 'states', 'seq_actions': 'actions', 'seq_returns': 'returns'}
    )
    dl_iter = iter(dl)
    wm.train()
    loss_val = 0.0
    total_batches = epochs_per_train * batches_per_epoch

    for step in range(total_batches):
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)

        batch = tree_map_tensor_to_device(batch, device)
        states, actions, returns = batch['states'], batch['actions'], batch['returns']
        lens = batch.get('n_step_lens', batch.get('episode_len', batch.get('episode_lens')))

        loss = sum(
            wm(states, actions, lens = lens, returns = returns, behavior_clone = True, world_model_index = idx)[0]
            for idx in world_model_indices
        )

        (loss / grad_accum_steps).backward()

        if divisible_by(step + 1, grad_accum_steps) or (step + 1) == total_batches:
            optimizer.step()
            optimizer.zero_grad()
            wm.update()
            loss_val = loss.item()

    return loss_val

# action selection - replan every `horizon` steps, execute the plan in order

def make_action_fn(wm, device, fitness_fn, goal_state, horizon, pop_size, generations, elite_frac, hierarchical):
    plan_queue = []

    def plan_from_state(state):
        state_seq = torch.as_tensor(state, device = device).reshape(-1, 1, state.shape[-1])
        batch = state_seq.shape[0]
        empty_actions = torch.empty(batch, 0, 1, device = device)

        if not hierarchical:
            return wm.plan(
                states = state_seq,
                actions = empty_actions,
                fitness_fn = fitness_fn,
                goal_state = goal_state,
                horizon = horizon,
                pop_size = pop_size,
                generations = generations,
                elite_frac = elite_frac,
                world_model_index = 0
            )

        # coarse plan (compression c) seeds the fine plan (compression 1)

        coarse = wm.plan(
            states = state_seq,
            actions = empty_actions,
            fitness_fn = fitness_fn,
            goal_state = goal_state,
            horizon = horizon,
            pop_size = pop_size,
            generations = generations,
            elite_frac = elite_frac,
            world_model_index = 0
        )

        return wm.plan(
            states = state_seq,
            actions = empty_actions,
            fitness_fn = fitness_fn,
            goal_state = goal_state,
            horizon = horizon,
            pop_size = pop_size,
            generations = generations,
            elite_frac = elite_frac,
            seed_action = coarse,
            world_model_index = 1
        )

    def action_fn(state, **kwargs):
        nonlocal plan_queue
        if not plan_queue:
            planned = plan_from_state(state)
            plan_queue = [*planned.unbind(dim = 1)]
        return plan_queue.pop(0)

    return action_fn

def main(
    temporal_compression = 1,
    hierarchical = False,
    plan_horizon = None,
    dim = 32,
    hidden_dim = 128,
    depth = 2,
    batch_size = 32,
    grad_accum_steps = 2,
    lr = 3e-3,
    num_episodes = 25,
    num_envs = 6,
    max_timesteps = 200,
    train_every_eps = 1,
    epochs_per_train = 3,
    batches_per_epoch = 6,
    train_window = 64,
    max_episodes_buffer = 200,
    cem_num_samples = 512,
    cem_num_iters = 8,
    cem_elite_frac = 0.1,
    random_action_eps = 12,
    transition_lookahead = 3, # mtp lookahead for the prefix state transition on both lanes
    target_avg_reward = 100.,
    reward_avg_window = 10,
    print_every_eps = 1,
    seed = 0,
    tag = '',
    cpu = False
):
    temporal_compression = int(temporal_compression)
    hierarchical = (str(hierarchical).lower() == 'true') if isinstance(hierarchical, str) else bool(hierarchical)
    cpu = (str(cpu).lower() == 'true') if isinstance(cpu, str) else bool(cpu)

    device = Accelerator(cpu = cpu).device
    print(f"device: {device} | compression: {temporal_compression} | hierarchical: {hierarchical}", flush = True)

    temporal_compressions = (temporal_compression, 1) if hierarchical else (temporal_compression,)
    world_model_indices = tuple(range(len(temporal_compressions)))

    # horizon is in raw steps and must be divisible by the compression

    if plan_horizon is None:
        plan_horizon = max(8, 2 * temporal_compression) if temporal_compression > 1 else 8
    plan_horizon = int(plan_horizon)
    assert divisible_by(plan_horizon, temporal_compression), f'plan horizon {plan_horizon} must be divisible by compression {temporal_compression}'

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = BatchedEnv([seed * 1000 + i for i in range(num_envs)], max_episode_steps = max_timesteps)
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[-1])

    wm = Agent(
        state_encoder = MLP(state_dim, *((hidden_dim,) * (depth - 1)), dim),
        action_encoder = MLP(action_dim, hidden_dim, dim),
        model = Transformer(dim = dim, depth = depth, causal = True),
        dim_action = action_dim,
        continuous_actions = True,
        transition_action_space = 'raw',
        state_linear_rnn_depth = 1,
        action_linear_rnn_depth = 1,
        value_loss_weight = 1.0,
        discount_factor = 0.99,
        reg_next_state_weight = 0.1,
        temporal_compression = temporal_compressions,
        transition_lookahead = transition_lookahead
    ).to(device)

    optimizer = Adam(wm.parameters(), lr = lr)

    rewards_history = []
    buffer = None
    wm_loss_val = 0.0
    solved_episode = None
    best_avg_reward = -float('inf')
    suffix = f"-{tag}" if tag else ""
    buffer_folder = f"./cartpole-wm-memories-c{temporal_compression}-h{plan_horizon}{'-hier' if hierarchical else ''}{suffix}-s{seed}"

    for episode in range(1, num_episodes + 1):
        is_random_ep = episode <= random_action_eps
        wm.eval()

        if is_random_ep:
            experience = wm.interact_with_environment(
                env = env,
                max_steps = max_timesteps,
                action_fn = (lambda env: env.action_space.sample())
            )
        else:
            fitness_fn = make_fitness()
            goal_state = torch.zeros((1, state_dim), device = device)
            experience = wm.interact_with_environment(
                env = env,
                max_steps = max_timesteps,
                action_fn = make_action_fn(
                    wm, device, fitness_fn, goal_state,
                    horizon = plan_horizon,
                    pop_size = cem_num_samples,
                    generations = cem_num_iters,
                    elite_frac = cem_elite_frac,
                    hierarchical = hierarchical
                )
            )

        buffer = store_experience_in_replay_buffer(
            experience,
            max_episodes = max_episodes_buffer,
            max_timesteps = max_timesteps + 1,
            folder = buffer_folder,
            buffer = buffer,
            overwrite = (episode == 1)
        )

        if divisible_by(episode, train_every_eps):
            wm_loss_val = train_wm_epoch(wm, buffer, optimizer, device, batch_size, epochs_per_train, grad_accum_steps, train_window, batches_per_epoch, world_model_indices)

        rewards_history.extend(experience.cumulative_rewards.tolist())
        avg_reward = float(np.mean(rewards_history[-reward_avg_window:]))
        best_avg_reward = max(best_avg_reward, avg_reward)

        if divisible_by(episode, print_every_eps):
            print(f"episode {episode:4d} | reward: {float(np.mean(experience.cumulative_rewards.tolist())):6.2f} | avg{reward_avg_window}: {avg_reward:6.2f} | wm_loss: {wm_loss_val:.4f}", flush = True)

        if avg_reward >= target_avg_reward:
            solved_episode = episode
            print(f"reached target avg{reward_avg_window} reward of {target_avg_reward} at episode {episode}", flush = True)
            break

    env.close()
    status = 'solved' if solved_episode is not None else 'failed'
    print(f"result {status} {solved_episode if solved_episode is not None else num_episodes} {best_avg_reward:.1f}", flush = True)

if __name__ == '__main__':
    fire.Fire(main)
