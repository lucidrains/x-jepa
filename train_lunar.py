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

import torch
import torch.nn.functional as F
from torch.optim import Adam

from accelerate import Accelerator

import gymnasium as gym

import wandb

from torch_einops_utils import tree_map_tensor_to_device

from x_mlps_pytorch import MLP

from x_jepa import WorldModel
from x_jepa.x_jepa import Transformer
from x_jepa.rl import ppo_loss, tpo_loss
from x_jepa.utils import store_experience_in_replay_buffer, divisible_by, exists

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

# main

def main(
    env_name = 'LunarLanderContinuous-v3',
    dim = 64,
    hidden_dim = 256,
    depth = 2,
    batch_size = 16,
    grad_accum_steps = 2,
    lr = 1e-3,
    rl_type = 'tpo',             # 'ppo' or 'tpo'
    rl_lr = 3e-4,
    rl_start_episode = 500,      # switch to RL policy after 500 episodes
    rl_epochs = 4,
    num_episodes = 1000,
    train_every_eps = 10,
    epochs_per_train = 2,
    max_episodes_buffer = 400,
    max_timesteps = 1000,
    cem_num_samples = 256,
    cem_num_iters = 3,
    cem_elite_frac = 0.1,
    cem_horizon = 1,
    random_action_eps = 10,
    gamma = 0.99,
    print_every_eps = 10,
    project_name = 'x-jepa-lunar-lander',
    record_video = False,
    use_wandb = True,
    cpu = False
):
    assert rl_type in ('ppo', 'tpo'), f"rl_type must be either 'ppo' or 'tpo', got {rl_type}"

    accelerator = Accelerator(cpu = cpu)
    device = accelerator.device
    print(f"using device: {device}", flush = True)

    if use_wandb:
        wandb.init(project = project_name, config = dict(rl_type = rl_type, rl_start_episode = rl_start_episode))

    render_mode = "rgb_array" if record_video else None
    env = gym.make(env_name, render_mode = render_mode)

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

    wm = WorldModel(
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
    if rl_type == 'ppo':
        rl_params = list(wm.actors['reflexive'].parameters()) + list(wm.value_network.parameters())
    else:
        rl_params = list(wm.actors['reflexive'].parameters())

    rl_optimizer = Adam(rl_params, lr = rl_lr)

    def fitness_fn(pred_next_encoded_states, encoded_goal, pred_values):
        dist_cost = F.smooth_l1_loss(
            pred_next_encoded_states,
            encoded_goal.expand_as(pred_next_encoded_states),
            reduction = 'none'
        ).sum(dim = -1)
        fitness = -dist_cost + 5.0 * pred_values
        return fitness.sum(dim = -1)

    rewards_history = []
    buffer = None
    best_avg_reward = -float('inf')
    loss_val = 0.0

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

        action_fn = (lambda env: env.action_space.sample()) if is_random_ep else None
        actor_module = 'reflexive' if use_rl else None

        experience = wm.interact_with_environment(
            env = env,
            max_steps = max_timesteps,
            goal_state = goal_tensor if (not is_random_ep and not use_rl) else None,
            fitness_fn = fitness_fn if not use_rl else None,
            action_fn = action_fn,
            actor_module = actor_module,
            horizon = cem_horizon,
            pop_size = cem_num_samples,
            generations = cem_num_iters,
            elite_frac = cem_elite_frac
        )

        total_reward = experience.cumulative_rewards.item()
        steps = experience.episode_len.item()

        rl_loss_val = 0.0
        if use_rl:
            wm.train()
            for _ in range(rl_epochs):
                rl_optimizer.zero_grad()
                r_loss = rl_loss_fn(wm, experience, actor_module = 'reflexive')
                r_loss.backward()
                rl_optimizer.step()
                wm.update()
                rl_loss_val = r_loss.item()

        buffer = store_experience_in_replay_buffer(
            experience,
            max_episodes = max_episodes_buffer,
            max_timesteps = max_timesteps + 1,
            folder = buffer_folder,
            buffer = buffer,
            overwrite = (episode == 1)
        )

        if divisible_by(episode, train_every_eps):
            dl = buffer.dataloader(batch_size = batch_size, shuffle = True)
            wm.train()

            for epoch in range(epochs_per_train):
                optimizer.zero_grad()
                for step, batch in enumerate(dl):
                    batch = tree_map_tensor_to_device(batch, device)
                    b_states, b_actions, b_returns = batch['states'], batch['actions'], batch['returns']
                    b_lens = batch.get('episode_len', batch.get('episode_lens'))

                    loss, _ = wm(b_states, b_actions, lens = b_lens, returns = b_returns, behavior_clone = (not use_rl))
                    loss = loss / grad_accum_steps
                    loss.backward()

                    if divisible_by(step + 1, grad_accum_steps):
                        optimizer.step()
                        optimizer.zero_grad()
                        wm.update()
                        loss_val = loss.item() * grad_accum_steps

        phase = 1 if use_rl else 0
        log_dict = dict(episode = episode, episode_length = steps, phase = phase)

        if use_rl:
            rewards_history.append(total_reward)
            avg_reward = float(np.mean(rewards_history[-100:]))
            log_dict.update(
                reward = total_reward,
                avg100_reward = avg_reward,
                rl_reward = total_reward,
                rl_avg100_reward = avg_reward,
                rl_loss = rl_loss_val,
                rl_type = rl_type
            )

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(wm.state_dict(), f"./{env_name.lower().replace('-v3', '')}-best.pt")

            if divisible_by(episode, print_every_eps):
                print(f"episode {episode:4d} [{rl_type.upper()} phase=1] | reward: {total_reward:8.2f} | avg100: {avg_reward:8.2f} | {rl_type}_loss: {rl_loss_val:.4f}", flush = True)

        elif is_ideal_goal:
            rewards_history.append(total_reward)
            avg_reward = float(np.mean(rewards_history[-100:]))
            log_dict.update(
                reward = total_reward,
                avg100_reward = avg_reward,
                ideal_goal_reward = total_reward,
                ideal_goal_avg100_reward = avg_reward
            )

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(wm.state_dict(), f"./{env_name.lower().replace('-v3', '')}-best.pt")

            if divisible_by(episode, print_every_eps):
                print(f"episode {episode:4d} [WM phase=0 ideal] | reward: {total_reward:8.2f} | avg100: {avg_reward:8.2f}", flush = True)
        else:
            log_dict.update(random_goal_reward = total_reward, reward = total_reward)

            if divisible_by(episode, print_every_eps):
                print(f"episode {episode:4d} [WM phase=0 random] | reward: {total_reward:8.2f}", flush = True)

        if divisible_by(episode, train_every_eps):
            log_dict.update(train_loss = loss_val)
            print(f"  --> trained world model | loss: {loss_val:.4f}", flush = True)

        if use_wandb:
            wandb.log(log_dict)

if __name__ == '__main__':
    fire.Fire(main)
