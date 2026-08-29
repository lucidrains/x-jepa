# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate",
#     "einops>=0.8.0",
#     "fire",
#     "gymnasium",
#     "memmap-replay-buffer",
#     "torch>=2.4",
#     "wandb>=0.25.1"
# ]
# ///
#
# continuous cartpole, trained purely with world model planning (CEM in the latent space, no RL)
#
# state transition can be the fast-lewm action-prefix transformer (https://arxiv.org/abs/2606.26217) -
# tokens [state latent] [action 1] [action 2] ... predict all future latents after each action prefix
# in parallel - or a classic one-step MLP transition (--transition_type mlp)
#
# default run: 6 vectorized envs (one per seed), 20 episodes, converges (avg reward >= 100)
# within ~13-15 episodes; verify with:
#
#   python train_cartpole.py
#   python train_cartpole.py --transition_type mlp
#
# overridable via fire, e.g.:
#   python train_cartpole.py --num_envs 16 --num_episodes 30 --cpu true

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import fire
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from accelerate import Accelerator
import wandb

from torch_einops_utils import tree_map_tensor_to_device
from x_mlps_pytorch import MLP

from x_jepa import Agent
from x_jepa.x_jepa import Transformer
from x_jepa.utils import store_experience_in_replay_buffer, divisible_by

from env_cartpole import BatchedEnv, DelayedObsEnv

# fitness - reach and stay at the upright state (goal = all zeros)
# distance between predicted next-step encoded states and the encoded goal, both in the same
# ema-encoder space, so the fitness is well calibrated for CEM

def make_fitness():
    def fitness_fn(pred_next_encoded_states, encoded_goal):
        dist_cost = F.smooth_l1_loss(
            pred_next_encoded_states,
            encoded_goal.expand_as(pred_next_encoded_states),
            reduction = 'none'
        ).sum(dim = -1)

        return -dist_cost.sum(dim = -1)

    return fitness_fn

# training helpers

def train_wm_epoch(wm, buffer, optimizer, device, batch_size, epochs_per_train, grad_accum_steps, train_window = 64, batches_per_epoch = 6, train_reward_heads = True):
    # fixed-length window sampling - keeps training cost bounded regardless of episode length

    dl = buffer.dataloader(
        batch_size = batch_size,
        shuffle = True,
        n_steps = train_window,
        sequence_fields = ('states', 'actions', 'returns', 'rewards', 'dones'),
        fieldname_map = {'seq_states': 'states', 'seq_actions': 'actions', 'seq_returns': 'returns', 'seq_rewards': 'rewards', 'seq_dones': 'dones'}
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
        states, actions, returns = batch['states'], batch['actions'], batch['returns']
        lens = batch.get('n_step_lens', batch.get('episode_len', batch.get('episode_lens')))

        wm_kwargs = dict(lens = lens, returns = returns, behavior_clone = True)

        if train_reward_heads:
            wm_kwargs.update(rewards = batch['rewards'], dones = batch['dones'])

        loss, loss_breakdown = wm(states, actions, **wm_kwargs)
        (loss / grad_accum_steps).backward()

        if divisible_by(step + 1, grad_accum_steps) or (step + 1) == total_batches:
            optimizer.step()
            optimizer.zero_grad()
            wm.update()
            loss_val = loss.item()
            reward_loss_val = loss_breakdown.reward_pred.item()
            discount_loss_val = loss_breakdown.discount_pred.item()

    return loss_val, reward_loss_val, discount_loss_val

# main

def main(
    env_name = 'continuous-cartpole',
    dim = 32,
    hidden_dim = 128,
    depth = 2,
    batch_size = 32,
    grad_accum_steps = 2,
    lr = 3e-3,
    num_episodes = 20,
    num_envs = 6,
    num_seed_groups = 6,
    train_every_eps = 1,
    epochs_per_train = 3,
    batches_per_epoch = 6,
    train_window = 64,
    max_episodes_buffer = 200,
    max_timesteps = 200,
    cem_num_samples = 512,
    cem_num_iters = 8,
    cem_elite_frac = 0.1,
    cem_horizon = 8,
    commit_k_steps = 8,
    reg_next_state_weight = 0.1,
    random_action_eps = 12,
    gamma = 0.99,
    use_action_delay = False, # train the transition on delayed action pairs, conditioned on the delay (geometric, params below)
    action_delay_prob = 0.5,
    action_delay_max = 4,
    action_delay_q = 0.5, # geometric rate of the delay draw
    action_delay_uniform_mix = 0.25,
    action_delay_pred_loss_weight = 0., # auxiliary categorical delay prediction head - estimate the lag from (state, action); ~0.1 is the empirically sensible weight
    plan_delay_predict = False, # plan with the delay predicted by the auxiliary head per rollout step (requires action_delay_pred_loss_weight > 0)
    world_delay = 0, # delayed perception stream - the env feeds observations `world_delay` ticks old (for use with use_action_delay)
    world_delay_max = 0, # stochastic delayed perception - per-step lag drawn from the training-time geometric distribution, capped at this max
    reward_avg_window = 10,
    target_avg_reward = 100.,
    print_every_eps = 1,
    seed = 0,
    tag = '',
    project_name = 'x-jepa-cartpole',
    use_wandb = False,
    cpu = False,
    transition_type = 'prefix', # 'prefix' for fast-lewm action-prefix transformer, 'mlp' for classic one-step
    transition_lookahead = 3, # mtp lookahead for the prefix transition - each prefix predicts the next `lookahead` latents
    plan_lookahead = None, # plan-time lookahead (prefix transition) - steps predicted per re-anchored pass; fall back to fewer and redo the prediction until the horizon is covered; defaults to the full horizon in one parallel pass
    train_reward_heads = True # train the reward and continuation heads (requires rewards/dones in the buffer); False exercises the heads-free training path
):
    device = Accelerator(cpu = cpu).device
    print(f"using device: {device}", flush = True)

    # fire passes cli booleans as strings - coerce so `--use_wandb false` actually disables logging

    use_wandb = (str(use_wandb).lower() == 'true') if isinstance(use_wandb, str) else bool(use_wandb)

    if use_wandb:
        wandb.init(project = project_name, config = dict(seed = seed))

    torch.manual_seed(seed)
    np.random.seed(seed)

    group_size = max(num_envs // num_seed_groups, 1)
    env = BatchedEnv([(i // group_size) * 1000 + (i % group_size) for i in range(num_envs)], max_episode_steps = max_timesteps)

    if world_delay_max > 0:
        env = DelayedObsEnv(env, delay_max = world_delay_max, seed = seed)
    elif world_delay > 0:
        env = DelayedObsEnv(env, delay = world_delay, seed = seed)

    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[-1])

    state_encoder = MLP(state_dim, *((hidden_dim,) * (depth - 1)), dim)
    action_encoder = MLP(action_dim, hidden_dim, dim)

    assert transition_type in ('prefix', 'mlp'), f'unknown transition_type: {transition_type}'

    state_transition = None

    if transition_type == 'mlp':
        state_transition = MLP(dim + action_dim, *((dim * 2,) * 2), dim)

    wm = Agent(
        state_encoder = state_encoder,
        action_encoder = action_encoder,
        model = Transformer(dim = dim, depth = depth, causal = True),
        state_transition = state_transition,
        dim_action = action_dim,
        continuous_actions = True,
        transition_action_space = 'raw',
        state_linear_rnn_depth = 1,
        action_linear_rnn_depth = 1,
        value_loss_weight = 1.0,
        discount_factor = gamma,
        reg_next_state_weight = reg_next_state_weight,
        transition_horizon = cem_horizon,
        transition_lookahead = transition_lookahead,
        use_action_delay = use_action_delay,
        action_delay_prob = action_delay_prob,
        action_delay_max = action_delay_max,
        action_delay_q = action_delay_q,
        action_delay_uniform_mix = action_delay_uniform_mix,
        action_delay_pred_loss_weight = action_delay_pred_loss_weight,
        add_reflexive_actor = True
    ).to(device)

    optimizer = Adam(wm.parameters(), lr = lr)

    rewards_history = []
    buffer = None
    best_avg_reward = -float('inf')
    wm_loss_val = 0.0
    wm_reward_loss_val = 0.0
    wm_discount_loss_val = 0.0
    suffix = f"-{tag}" if tag else ""
    buffer_folder = f"./{env_name.lower().replace('-v1', '')}-memories{suffix}"

    for episode in range(1, num_episodes + 1):
        is_random_ep = episode <= random_action_eps

        wm.eval()

        experience = wm.interact_with_environment(
            env = env,
            max_steps = max_timesteps,
            goal_state = None if is_random_ep else torch.zeros((1, state_dim), device = device),
            fitness_fn = None if is_random_ep else make_fitness(),
            action_fn = (lambda env: env.action_space.sample()) if is_random_ep else None,
            horizon = cem_horizon,
            pop_size = cem_num_samples,
            generations = cem_num_iters,
            elite_frac = cem_elite_frac,
            commit_k_steps = commit_k_steps,
            plan_lookahead = plan_lookahead,
            plan_delay_predict = plan_delay_predict
        )

        total_rewards = experience.cumulative_rewards.tolist()

        buffer = store_experience_in_replay_buffer(
            experience,
            max_episodes = max_episodes_buffer,
            max_timesteps = max_timesteps + 1,
            folder = buffer_folder,
            buffer = buffer,
            overwrite = (episode == 1)
        )

        if divisible_by(episode, train_every_eps):
            wm_loss_val, wm_reward_loss_val, wm_discount_loss_val = train_wm_epoch(wm, buffer, optimizer, device, batch_size, epochs_per_train, grad_accum_steps, train_window = train_window, batches_per_epoch = batches_per_epoch, train_reward_heads = train_reward_heads)

        # metrics and logging

        rewards_history.extend(total_rewards)
        avg_reward = float(np.mean(rewards_history[-reward_avg_window:]))
        batch_mean = float(np.mean(total_rewards))
        batch_max = float(np.max(total_rewards))

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save(wm.state_dict(), f"./{env_name.lower().replace('-v1', '')}-best{suffix}.pt")

        if divisible_by(episode, print_every_eps):
            group_info = ""
            if num_seed_groups > 1:
                gmeans = [float(np.mean(total_rewards[g * group_size:(g + 1) * group_size])) for g in range(num_seed_groups)]
                group_info = " | groups: " + " ".join(f"{g:5.1f}" for g in gmeans)
            print(f"episode {episode:4d} [WM planning]  | reward mean: {batch_mean:6.2f} max: {batch_max:6.2f} | avg{reward_avg_window}: {avg_reward:6.2f} | wm_loss: {wm_loss_val:.4f} | reward_loss: {wm_reward_loss_val:.4f} | discount_loss: {wm_discount_loss_val:.4f}{group_info}", flush = True)

        if use_wandb:
            wandb.log(dict(episode = episode, reward = batch_mean, avg_reward = avg_reward))

        # arrest once the average reward clears the target

        if avg_reward >= target_avg_reward:
            print(f"\nreached target avg{reward_avg_window} reward of {target_avg_reward} - stopping early", flush = True)
            break

    env.close()
    print(f"\ndone! best avg{reward_avg_window} reward: {best_avg_reward:.2f}", flush = True)

if __name__ == '__main__':
    fire.Fire(main)
