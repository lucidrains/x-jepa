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
# default run: 6 vectorized envs (one per seed), 20 episodes, converges (avg reward >= 100)
# within ~13-15 episodes; verify with:
#
#   python train_cartpole.py
#
# overridable via fire, e.g.:
#   python train_cartpole.py --num_envs 16 --num_episodes 30 --cpu true

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import math

import fire
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from accelerate import Accelerator
import gymnasium as gym
from gymnasium import spaces
import wandb

from torch_einops_utils import tree_map_tensor_to_device
from x_mlps_pytorch import MLP

from x_jepa import Agent
from x_jepa.x_jepa import Transformer
from x_jepa.utils import store_experience_in_replay_buffer, divisible_by

# continuous cartpole - same physics as CartPole-v1, but the action is the raw force in [-1, 1]

class ContinuousCartPole(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, force_mag = 10.0, max_episode_steps = 500, seed = None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.tau = 0.02
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.force_mag = force_mag
        self.max_episode_steps = max_episode_steps

        self.observation_space = spaces.Box(-np.inf, np.inf, (4,), dtype = np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), dtype = np.float32)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        self.steps = 0
        self.state = np.array([0.0, 0.0, self.np_random.uniform(-0.05, 0.05), 0.0], dtype = np.float32)
        return self.state.copy(), {}

    def step(self, action):
        action = np.clip(np.asarray(action).squeeze(), -1.0, 1.0).item()
        force = action * self.force_mag
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot], dtype = np.float32)

        self.steps += 1
        terminated = bool(abs(x) > self.x_threshold or abs(theta) > self.theta_threshold_radians)
        truncated = self.steps >= self.max_episode_steps
        reward = 1.0
        return self.state.copy(), reward, terminated, truncated, {}

# batched wrapper (no auto-reset - done envs freeze so each env contributes exactly one episode per rollout)

class BatchedEnv:
    def __init__(self, seeds, max_episode_steps = 500):
        self.envs = [ContinuousCartPole(seed = s, max_episode_steps = max_episode_steps) for s in seeds]
        self.num_envs = len(self.envs)
        self.observation_space = spaces.Box(-np.inf, np.inf, (4,), dtype = np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.num_envs, 1), dtype = np.float32)
        self._done = [False] * self.num_envs
        self._last_obs = [None] * self.num_envs

    def reset(self, seed = None, options = None):
        self._done = [False] * self.num_envs
        obs = []
        for i, e in enumerate(self.envs):
            o, _ = e.reset()
            self._last_obs[i] = o
            obs.append(o)
        return np.stack(obs), {}

    def step(self, actions):
        obs, rewards, terminateds, truncateds = [], [], [], []
        for i, (e, a) in enumerate(zip(self.envs, actions)):
            if self._done[i]:
                obs.append(self._last_obs[i])
                rewards.append(0.0)
                terminateds.append(True)
                truncateds.append(False)
                continue

            o, r, terminated, truncated, _ = e.step(a)
            self._last_obs[i] = o
            if terminated or truncated:
                self._done[i] = True

            obs.append(o)
            rewards.append(r)
            terminateds.append(terminated)
            truncateds.append(truncated)

        return np.stack(obs), np.array(rewards, dtype = np.float32), np.array(terminateds, dtype = bool), np.array(truncateds, dtype = bool), {}

    def close(self):
        for e in self.envs:
            e.close()

# fitness - reach and stay at the upright state (goal = all zeros)
#
# distance is measured between the predicted next-step encoded states and the encoded goal.
# both live in the same ema-encoder space, so the fitness is well calibrated for CEM.
# (the alternative of comparing CEM rollout latents to a context-free goal latent is
#  miscalibrated - rollout latents are contextualized, the goal latent is not - which
#  produces a flat fitness landscape and coin-flip planning)

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

def train_wm_epoch(wm, buffer, optimizer, device, batch_size, epochs_per_train, grad_accum_steps, train_window = 64, batches_per_epoch = 6):
    # fixed-length window sampling - keeps training cost bounded regardless of episode length
    # (full-sequence training on long episodes is O(T^2) in the causal attention and is very slow on mps)

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

        loss, _ = wm(states, actions, lens = lens, returns = returns, behavior_clone = True)
        (loss / grad_accum_steps).backward()

        if divisible_by(step + 1, grad_accum_steps) or (step + 1) == total_batches:
            optimizer.step()
            optimizer.zero_grad()
            wm.update()
            loss_val = loss.item()

    return loss_val

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
    reward_avg_window = 10,
    target_avg_reward = 100.,
    print_every_eps = 1,
    seed = 0,
    tag = '',
    project_name = 'x-jepa-cartpole',
    use_wandb = False,
    cpu = False
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

    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[-1])

    state_encoder = MLP(state_dim, *((hidden_dim,) * (depth - 1)), dim)
    action_encoder = MLP(action_dim, hidden_dim, dim)

    wm = Agent(
        state_encoder = state_encoder,
        action_encoder = action_encoder,
        model = Transformer(dim = dim, depth = depth, causal = True),
        dim_action = action_dim,
        continuous_actions = True,
        transition_action_space = 'raw',
        state_linear_rnn_depth = 1,
        action_linear_rnn_depth = 1,
        value_loss_weight = 1.0,
        discount_factor = gamma,
        reg_next_state_weight = reg_next_state_weight,
        add_reflexive_actor = True
    ).to(device)

    optimizer = Adam(wm.parameters(), lr = lr)

    rewards_history = []
    buffer = None
    best_avg_reward = -float('inf')
    wm_loss_val = 0.0
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
            commit_k_steps = commit_k_steps
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
            wm_loss_val = train_wm_epoch(wm, buffer, optimizer, device, batch_size, epochs_per_train, grad_accum_steps, train_window = train_window, batches_per_epoch = batches_per_epoch)

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
            print(f"episode {episode:4d} [WM planning]  | reward mean: {batch_mean:6.2f} max: {batch_max:6.2f} | avg{reward_avg_window}: {avg_reward:6.2f} | wm_loss: {wm_loss_val:.4f}{group_info}", flush = True)

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
