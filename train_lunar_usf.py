# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate",
#     "einops>=0.8.0",
#     "einx>=0.3.0",
#     "fire",
#     "gymnasium[box2d]",
#     "memmap-replay-buffer",
#     "pillow",
#     "torch>=2.4",
# ]
# ///
#
# continuous lunar lander with FDM + SigReg + Universal Successor Features (USF)
#
# reference:
#   Zero-Shot Control with Gaussian Representations and Test-Time Adjustment
#   https://openreview.net/forum?id=WeUNn9rlIi

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import fire
import gymnasium as gym
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.optim import Adam

from accelerate import Accelerator
from einops import rearrange, reduce
from torch_einops_utils import tree_map_tensor_to_device, lens_to_mask
from x_mlps_pytorch import MLP

from x_jepa import Agent
from x_jepa.x_jepa import Transformer, calc_returns
from x_jepa.utils import exists, default, store_experience_in_replay_buffer, divisible_by

# helpers

def cycle(dl):
    while True:
        yield from dl

# batched continuous lunar lander environment

class BatchedLunarEnv:
    def __init__(self, seeds, max_episode_steps = 300):
        self.envs = [gym.make('LunarLanderContinuous-v3', max_episode_steps = max_episode_steps) for _ in seeds]
        self.seeds = seeds
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.num_envs, 2), dtype = np.float32)
        self.action_space.seed(0)
        self._done = [False] * self.num_envs
        self._last_obs = [None] * self.num_envs

    def reset(self, seed = None, options = None):
        self._done = [False] * self.num_envs
        obs = []

        for i, (env, s) in enumerate(zip(self.envs, self.seeds)):
            o, _ = env.reset(seed = s)
            self._last_obs[i] = o
            obs.append(o)

        return np.stack(obs), {}

    def step(self, actions):
        obs, rewards, terminateds, truncateds = [], [], [], []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if self._done[i]:
                obs.append(self._last_obs[i])
                rewards.append(0.0)
                terminateds.append(True)
                truncateds.append(False)
                continue

            o, r, terminated, truncated, _ = env.step(np.asarray(action, dtype = np.float32))
            self._last_obs[i] = o

            if terminated or truncated:
                self._done[i] = True

            obs.append(o)
            rewards.append(float(r))
            terminateds.append(bool(terminated))
            truncateds.append(bool(truncated))

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

def save_lunar_gif(actions, seed, path = './lunar-usf.gif', max_episode_steps = 300, fps = 40):
    env = gym.make('LunarLanderContinuous-v3', render_mode = 'rgb_array', max_episode_steps = max_episode_steps)
    env.reset(seed = seed)
    frames = [Image.fromarray(env.render())]

    for action in actions:
        _, _, terminated, truncated, _ = env.step(np.asarray(action, dtype = np.float32))
        frames.append(Image.fromarray(env.render()))
        if terminated or truncated:
            break

    env.close()
    frames[0].save(path, save_all = True, append_images = frames[1:], duration = int(1000 / fps), loop = 0)

# fitness - FDM rollout + USF goal bootstrap + USF return-regressed task Q

def make_usf_fitness(task_state, gamma = 0.99, usf_weight = 0.5):
    def fitness_fn(
        pred_next_encoded_states,
        pred_state_latents,
        encoded_goal,
        goal_latent,
        actions,
        usf,
        discount_factor = None
    ):
        disc = default(discount_factor, gamma)

        fdm_cost = reduce(
            F.smooth_l1_loss(pred_next_encoded_states, encoded_goal.expand_as(pred_next_encoded_states), reduction = 'none'),
            'b p h d -> b p',
            'sum'
        )

        z_terminal = pred_state_latents[:, :, -1]
        goal_task = rearrange(goal_latent, 'b 1 1 d -> b 1 d') - z_terminal
        usf_q = (1. - disc) * usf.q(z_terminal, actions[:, :, -1], goal_task, discount = disc, ema = True)

        reward_task = task_state.get('w')
        if exists(reward_task):
            usf_q = usf_q + reduce(usf.q(pred_state_latents, actions, reward_task, discount = disc, ema = True), 'b p h -> b p', 'mean')

        return -fdm_cost + usf_weight * usf_q

    return fitness_fn

# training

def train_wm_epoch(
    wm,
    buffer,
    optimizer,
    device,
    task_state,
    gamma = 0.99,
    batch_size = 32,
    steps = 24,
    grad_accum_steps = 2,
    train_window = 64
):
    dl = cycle(buffer.dataloader(
        batch_size = batch_size,
        shuffle = True,
        n_steps = train_window,
        return_indices = True,
        sequence_fields = ('states', 'actions', 'rewards', 'dones'),
        fieldname_map = dict(seq_states = 'states', seq_actions = 'actions', seq_rewards = 'rewards', seq_dones = 'dones')
    ))

    wm.train()
    latents, actions, returns = [], [], []

    for step in range(1, steps + 1):
        batch = tree_map_tensor_to_device(next(dl), device)
        lens = batch.get('n_step_lens', batch['episode_len'])

        if '_indices' in batch:
            lens = torch.minimum(lens, (batch['episode_len'] - batch['_indices'][:, 1]).clamp(min = 0))

        loss, breakdown, outputs = wm(
            batch['states'],
            batch['actions'],
            lens = lens,
            dones = batch['dones'],
            behavior_clone = False,
            return_outputs = True
        )
        (loss / grad_accum_steps).backward()

        if divisible_by(step, grad_accum_steps) or step == steps:
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.)
            optimizer.step()
            optimizer.zero_grad()
            wm.update()

        with torch.no_grad():
            z = outputs['state_latents_full']
            m = min(z.shape[1] - 1, batch['actions'].shape[1], batch['rewards'].shape[1])

            if m > 0:
                valid = lens_to_mask(lens - 1, m)
                disc_returns = calc_returns(batch['rewards'][:, :m], batch['dones'][:, :m].bool(), 0., gamma, mask = valid) / 100.

                if valid.any():
                    latents.append(z[:, :m][valid])
                    actions.append(batch['actions'][:, :m][valid])
                    returns.append(disc_returns[valid])

    if latents:
        task_state['w'] = wm.usf.infer_task(
            torch.cat(latents, dim = 0),
            torch.cat(returns, dim = 0),
            actions = torch.cat(actions, dim = 0),
            discount = gamma,
            ridge_lambda = 1e-2
        )

    return loss.item(), breakdown.reg_next_state.item(), breakdown.usf.item()

# main

def main(
    dim = 64,
    hidden_dim = 128,
    depth = 2,
    batch_size = 32,
    grad_accum_steps = 2,
    lr = 3e-3,
    num_episodes = 13,
    num_envs = 6,
    random_train_steps = 24,
    plan_train_steps = 6,
    train_window = 64,
    max_episodes_buffer = 200,
    max_timesteps = 300,
    cem_num_samples = 512,
    cem_num_iters = 6,
    cem_elite_frac = 0.1,
    cem_horizon = 8,
    commit_k_steps = 2,
    reg_next_state_weight = 0.1,
    usf_loss_weight = 1.0,
    usf_weight = 0.5,
    random_action_eps = 8,
    gamma = 0.99,
    usf_discount_range = (0.9, 0.995),
    target_mean_reward = 80.,
    video_path = './lunar-usf.gif',
    seed = 0,
    cpu = False
):
    device = Accelerator(cpu = cpu).device
    print(f"using device: {device}", flush = True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    seeds = [i * 100 for i in range(num_envs)]
    env = BatchedLunarEnv(seeds, max_episode_steps = max_timesteps)

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
        value_loss_weight = 0.,
        reward_pred_loss_weight = 0.,
        discount_pred_loss_weight = 0.,
        discount_factor = gamma,
        reg_next_state_weight = reg_next_state_weight,
        transition_horizon = cem_horizon,
        transition_lookahead = 3,
        use_usf = True,
        usf_loss_weight = usf_loss_weight,
        usf_discount_factor = usf_discount_range,
        usf_frac_gradient = 0.1
    ).to(device)

    optimizer = Adam(wm.parameters(), lr = lr)

    goal = torch.zeros((1, state_dim), device = device)
    task_state = dict(w = None)
    fitness_fn = make_usf_fitness(task_state, gamma = gamma, usf_weight = usf_weight)
    buffer = None

    for episode in range(1, num_episodes + 1):
        is_random = episode <= random_action_eps

        wm.eval()

        experience = wm.interact_with_environment(
            env = env,
            max_steps = max_timesteps,
            goal_state = None if is_random else goal,
            fitness_fn = None if is_random else fitness_fn,
            action_fn = (lambda env: env.action_space.sample()) if is_random else None,
            horizon = cem_horizon,
            pop_size = cem_num_samples,
            generations = cem_num_iters,
            elite_frac = cem_elite_frac,
            commit_k_steps = 1 if is_random else commit_k_steps,
            discount_factor = gamma
        )

        buffer = store_experience_in_replay_buffer(
            experience,
            max_episodes = max_episodes_buffer,
            max_timesteps = max_timesteps + 1,
            folder = './continuous-lunar-memories-usf',
            buffer = buffer,
            overwrite = (episode == 1)
        )

        wm_loss, sigreg_loss, usf_loss = train_wm_epoch(
            wm, buffer, optimizer, device, task_state,
            gamma = gamma,
            batch_size = batch_size,
            steps = random_train_steps if is_random else plan_train_steps,
            grad_accum_steps = grad_accum_steps,
            train_window = train_window
        )

        total_rewards = experience.cumulative_rewards.tolist()
        mean_reward = float(np.mean(total_rewards))

        mode_str = 'random' if is_random else 'FDM+USF'
        print(
            f"episode {episode:3d} [{mode_str:7s}] | reward mean: {mean_reward:7.2f} max: {np.max(total_rewards):7.2f} "
            f"| loss: {wm_loss:.4f} (sigreg: {sigreg_loss:.4f}, usf: {usf_loss:.4f})",
            flush = True
        )

        if not is_random and mean_reward >= target_mean_reward:
            print(f"\nreached target mean reward of {target_mean_reward} - converged!", flush = True)
            break

    if exists(video_path):
        best_idx = int(experience.cumulative_rewards.argmax())
        ep_len = int(experience.episode_len[best_idx])
        save_lunar_gif(
            experience.actions[best_idx, :ep_len - 1].cpu().numpy(),
            seed = seeds[best_idx],
            path = video_path,
            max_episode_steps = max_timesteps
        )
        print(f"saved rollout gif ({ep_len} steps, reward {experience.cumulative_rewards[best_idx]:.2f}) -> {video_path}", flush = True)

    env.close()

if __name__ == '__main__':
    fire.Fire(main)
