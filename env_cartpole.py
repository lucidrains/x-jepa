# continuous cartpole - same physics as CartPole-v1, but the action is the raw force in [-1, 1]

import math
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

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
        self.np_random = np.random.default_rng(seed if seed is not None else 0)

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        self.steps = 0
        self.state = np.array([0.0, 0.0, self.np_random.uniform(-0.05, 0.05), 0.0], dtype = np.float32)
        return self.state.copy(), {}

    def step(self, action):
        force = float(np.clip(np.asarray(action).squeeze(), -1.0, 1.0)) * self.force_mag
        x, x_dot, theta, theta_dot = self.state
        costheta, sintheta = math.cos(theta), math.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        self.state = np.array([
            x + self.tau * x_dot,
            x_dot + self.tau * xacc,
            theta + self.tau * theta_dot,
            theta_dot + self.tau * thetaacc
        ], dtype = np.float32)

        self.steps += 1
        terminated = bool(abs(self.state[0]) > self.x_threshold or abs(self.state[2]) > self.theta_threshold_radians)
        truncated = self.steps >= self.max_episode_steps
        return self.state.copy(), 1.0, terminated, truncated, {}

class BatchedEnv:
    """no auto-reset - done envs freeze so each env yields exactly one episode per rollout"""

    def __init__(self, seeds, max_episode_steps = 500):
        self.envs = [ContinuousCartPole(seed = s, max_episode_steps = max_episode_steps) for s in seeds]
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = spaces.Box(-1.0, 1.0, (self.num_envs, 1), dtype = np.float32)
        self._done = [False] * self.num_envs
        self._last_obs = [None] * self.num_envs

    def reset(self, seed = None, options = None):
        self._done = [False] * self.num_envs
        obs = []
        for i, env in enumerate(self.envs):
            o, _ = env.reset()
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

class DelayedObsEnv:
    """delayed perception wrapper - the agent sees the environment state `delay` ticks old; with
    `delay_max` the lag is drawn per step from the geometric-family distribution of the augmentation"""

    def __init__(self, env, delay = 1, delay_max = None, prob = 0.5, q = 0.5, uniform_mix = 0.25, seed = None):
        self.env = env
        self.delay = delay
        self.delay_max = delay_max
        self.prob = prob
        self.q = q
        self.uniform_mix = uniform_mix
        self.stochastic = delay_max is not None
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._queue = [deque(maxlen = max(delay, delay_max or 0) + 1) for _ in range(self.num_envs)]
        self._rng = np.random.default_rng(seed)

    def _sample_delays(self):
        # same family as the training-time augmentation - flag between the clean channel and a
        # truncated geometric mixed with a uniform floor

        if not self.stochastic:
            return np.full(self.num_envs, self.delay, dtype = np.int64)

        flag = self._rng.random(self.num_envs) < self.prob
        geometric = np.minimum(self._rng.geometric(self.q, self.num_envs), self.delay_max) # support 1.., P(k) = q(1 - q)^(k - 1)
        uniform = self._rng.integers(1, self.delay_max + 1, self.num_envs)
        mixed = np.where(self._rng.random(self.num_envs) < self.uniform_mix, uniform, geometric)

        return np.where(flag, mixed, 0).astype(np.int64)

    def reset(self, seed = None, options = None):
        obs, info = self.env.reset()
        for i in range(self.num_envs):
            self._queue[i].clear()
            for _ in range(max(self.delay, self.delay_max or 0) + 1):
                self._queue[i].append(obs[i])

        return obs, info

    def step(self, actions):
        obs, rewards, terminateds, truncateds, info = self.env.step(actions)
        self._d = self._sample_delays()

        delayed_obs = []
        for i, q in enumerate(self._queue):
            q.append(obs[i])
            delayed_obs.append(q[-(int(self._d[i]) + 1)])

        return np.stack(delayed_obs), rewards, terminateds, truncateds, info

    def close(self):
        self.env.close()
