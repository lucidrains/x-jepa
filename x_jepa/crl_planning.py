from __future__ import annotations

from collections import deque
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter

import einx
from einops import rearrange
from torch_einops_utils import cross_entropy, safe_cat, safe_stack, slerp

# helpers

def exists(value):
    return value is not None

def default(value, fallback):
    return value if exists(value) else fallback

def l2norm(tensor):
    return F.normalize(tensor, dim = -1)

# cayley transform -> orthogonal rotation

def cayley_rotation(matrix, eye = None):
    omega = matrix - matrix.transpose(-1, -2)
    eye = default(eye, torch.eye(matrix.shape[-1], device = matrix.device, dtype = matrix.dtype))
    return torch.linalg.solve(eye + omega, eye - omega)

# hyperspherical infonce
#
# f(s, g) = psi(s)^T R psi(g)
#
# asymmetric by default, with a learned cayley rotation on the goal side

class HypersphericalInfoNCE(Module):

    def __init__(
        self,
        dim = None,
        *,
        symmetric = False,
        learn_rotation = True,
        learned_temp = True,
        init_scale = 10.
    ):
        super().__init__()

        assert not learn_rotation or exists(dim)
        assert init_scale > 0.

        self.symmetric = symmetric
        self.has_rotation = learn_rotation

        self.rotation_param = Parameter(torch.zeros(dim, dim)) if learn_rotation else None
        self.register_buffer('eye', torch.eye(dim) if learn_rotation else None, persistent = False)

        self.learned_log_temp = Parameter(torch.log(torch.tensor(float(init_scale)))) if learned_temp else None
        self.scale_value = init_scale

    @property
    def scale(self):
        return self.learned_log_temp.exp() if exists(self.learned_log_temp) else self.scale_value

    @property
    def rotation(self):
        return cayley_rotation(self.rotation_param, self.eye) if self.has_rotation else None

    def encode(self, embeds, rotate = False):
        embeds = l2norm(embeds)

        if rotate and self.has_rotation:
            embeds = einx.dot('i j, ... j -> ... i', self.rotation, embeds)

        return embeds

    def score(self, state_embeds, goal_embeds):
        return einx.dot('... d, ... d -> ...', self.encode(state_embeds), self.encode(goal_embeds, rotate = True))

    def _infonce(self, anchors, positives, extra_negatives = None, rotate_anchors = False):
        anchors = self.encode(anchors, rotate = rotate_anchors)
        candidates = self.encode(positives, rotate = not rotate_anchors)

        if exists(extra_negatives):
            extra_negatives = self.encode(extra_negatives, rotate = not rotate_anchors)
            extra_negatives = rearrange(extra_negatives, '... d -> (...) d')
            candidates = safe_cat((candidates, extra_negatives), dim = 0)

        logits = einx.dot('... d, n d -> ... n', anchors, candidates) * self.scale
        labels = torch.arange(anchors.shape[0], device = anchors.device)

        return cross_entropy(logits, labels, class_dim = -1)

    def forward(self, state_embeds, goal_embeds, extra_negatives = None):
        loss = self._infonce(state_embeds, goal_embeds, extra_negatives)

        if not self.symmetric:
            return loss

        reverse_loss = self._infonce(goal_embeds, state_embeds, extra_negatives, rotate_anchors = True)
        return (loss + reverse_loss) * 0.5

# fifo of trajectories

class Trajectory(NamedTuple):
    states: Tensor
    actions: Tensor | None = None

class CRLBatch(NamedTuple):
    states: Tensor
    goals: Tensor
    actions: Tensor | None = None
    negatives: Tensor | None = None

class FIFOBuffer:

    def __init__(self, capacity: int):
        assert capacity > 0

        self.capacity = capacity
        self.trajectories = deque()
        self.num_states = 0

    def __len__(self):
        return self.num_states

    def push(self, states, actions = None):
        states = torch.as_tensor(states).detach().clone()
        actions = torch.as_tensor(actions).detach().clone() if exists(actions) else None

        assert states.ndim >= 1 and len(states) > 0
        assert not exists(actions) or len(actions) == len(states)

        self.trajectories.append(Trajectory(states, actions))
        self.num_states += len(states)

        self._trim()
        return self

    def _trim(self):
        while self.num_states > self.capacity:
            oldest = self.trajectories[0]
            overflow = self.num_states - self.capacity

            if overflow >= len(oldest.states):
                self.trajectories.popleft()
                self.num_states -= len(oldest.states)
                continue

            actions = oldest.actions[overflow:] if exists(oldest.actions) else None
            self.trajectories[0] = Trajectory(oldest.states[overflow:], actions)
            self.num_states -= overflow

    def _length_probs(self):
        lengths = torch.tensor([len(trajectory.states) for trajectory in self.trajectories], dtype = torch.float)
        return lengths / lengths.sum()

    def _sample_indices(self, batch_size, generator = None):
        probs = self._length_probs()
        return torch.multinomial(probs, batch_size, replacement = True, generator = generator).tolist()

    def _sample_positions(self, traj_indices, generator = None):
        return [
            int(torch.randint(0, len(self.trajectories[traj_index].states), (1,), generator = generator).item())
            for traj_index in traj_indices
        ]

    def sample_states(self, batch_size, *, generator = None):
        traj_indices = self._sample_indices(batch_size, generator = generator)
        positions = self._sample_positions(traj_indices, generator = generator)
        return safe_stack([
            self.trajectories[traj_index].states[position]
            for traj_index, position in zip(traj_indices, positions)
        ])

    def sample(self, batch_size, *, discount = 0.99, num_negatives = 0, generator = None):
        assert len(self.trajectories) > 0
        assert 0. < discount <= 1.
        assert num_negatives >= 0

        num_with_actions = sum(exists(trajectory.actions) for trajectory in self.trajectories)
        assert num_with_actions in (0, len(self.trajectories)), 'either all trajectories need actions or none'
        has_actions = num_with_actions > 0

        traj_indices = self._sample_indices(batch_size, generator = generator)
        starts = self._sample_positions(traj_indices, generator = generator)

        offsets = torch.empty(batch_size).geometric_(1. - discount).long().tolist() if discount < 1. else [1] * batch_size

        goals = [
            min(start + offset, len(self.trajectories[traj_index].states) - 1)
            for traj_index, start, offset in zip(traj_indices, starts, offsets)
        ]

        states = safe_stack([
            self.trajectories[traj_index].states[start]
            for traj_index, start in zip(traj_indices, starts)
        ])
        goals = safe_stack([
            self.trajectories[traj_index].states[goal_index]
            for traj_index, goal_index in zip(traj_indices, goals)
        ])
        actions = safe_stack([
            self.trajectories[traj_index].actions[start]
            for traj_index, start in zip(traj_indices, starts)
        ]) if has_actions else None

        negatives = None

        if num_negatives > 0:
            negatives = self.sample_states(batch_size * num_negatives, generator = generator)
            negatives = rearrange(negatives, '(b n) ... -> b n ...', b = batch_size, n = num_negatives)

        return CRLBatch(states, goals, actions, negatives)

# learn a crl critic on top of x-jepa's non-contextual state encoder

class XJepaCRLLearner:

    def __init__(
        self,
        agent,
        dim = None,
        *,
        critic = None,
        encoder = None,
        train_encoder = False,
        lr = 3e-4,
        **critic_kwargs
    ):
        self.agent = agent
        self.train_encoder = train_encoder
        self.encoder = default(encoder, agent.state_encoder if train_encoder else agent.ema_state_encoder)

        dim = default(dim, agent.get_world_model().model.dim)
        self.critic = default(critic, HypersphericalInfoNCE(dim, **critic_kwargs))

        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)

    def parameters(self):
        params = list(self.critic.parameters())

        if self.train_encoder:
            params.extend(self.encoder.parameters())

        return params

    def encode(self, states):
        embed, _, _ = self.agent.encode_states(self.encoder, states)
        return embed

    def loss(self, batch):
        state_embeds = self.encode(batch.states)
        goal_embeds = self.encode(batch.goals)
        extra_negatives = self.encode(batch.negatives) if exists(batch.negatives) else None

        return self.critic(state_embeds, goal_embeds, extra_negatives = extra_negatives)

    def learn(
        self,
        buffer,
        *,
        steps = 1,
        batch_size = 256,
        discount = 0.99,
        num_negatives = 8,
        generator = None,
        grad_clip = None
    ):
        assert steps > 0

        for _ in range(steps):
            batch = buffer.sample(batch_size, discount = discount, num_negatives = num_negatives, generator = generator)

            loss = self.loss(batch)

            self.optimizer.zero_grad()
            loss.backward()

            if exists(grad_clip):
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

            self.optimizer.step()

            if self.train_encoder:
                for ema in self.agent.ema_state_encoder:
                    ema.update()

        return loss.item()

# c-slerp waypoint inference
#
# mu_i = (R^T)^i slerp(psi(s), R^(K+1) psi(g), i / (K + 1))

class CSlerpPlan(NamedTuple):
    waypoints: Tensor          # (K + 1, d), last waypoint is the goal
    num_waypoints: int         # K intermediates
    selected_index: int | None
    subgoal: Tensor | None

def c_slerp_plan(
    state_embed,
    goal_embed,
    rotation = None,
    *,
    num_waypoints = None,
    max_num_waypoints = 4,
    waypoint_temperature = 0.1,
    entropy_weight = 0.,
    threshold = 0.4,
    epsilon = 1e-7,
    score_fn = None,
    generator = None
):
    assert state_embed.ndim == 1 and goal_embed.shape == state_embed.shape
    assert max_num_waypoints >= 0 and entropy_weight >= 0.

    if exists(num_waypoints):
        assert 0 <= num_waypoints <= max_num_waypoints

    device, dtype = state_embed.device, state_embed.dtype
    state, goal = map(l2norm, (state_embed, goal_embed))
    rotation = default(rotation, torch.eye(state.shape[-1], device = device, dtype = dtype)).to(device = device, dtype = dtype)

    num_candidates = max_num_waypoints + 1
    rotation_powers = safe_stack([torch.linalg.matrix_power(rotation, i) for i in range(1, num_candidates + 1)])
    rotated_goals = einx.dot('n i j, j -> n i', rotation_powers, goal)
    alignments = einx.dot('n i, i -> n', rotated_goals, state)

    if exists(num_waypoints):
        num_selected_waypoints = num_waypoints
    elif waypoint_temperature > 0.:
        waypoint_logits = einx.softmax('n', alignments / waypoint_temperature)
        num_selected_waypoints = torch.multinomial(waypoint_logits, 1, generator = generator).item()
    else:
        num_selected_waypoints = alignments.argmax().item()

    rotated_goal = rotated_goals[num_selected_waypoints]
    means = goal.new_zeros((0, goal.shape[-1]))

    if num_selected_waypoints > 0:
        fractions = torch.arange(1, num_selected_waypoints + 1, device = device, dtype = dtype) / (num_selected_waypoints + 1)
        interpolated = slerp(state, rotated_goal, fractions, eps = epsilon)

        rotation_powers_k = rotation_powers[:num_selected_waypoints].transpose(-1, -2)
        means = l2norm(einx.dot('n i j, n j -> n i', rotation_powers_k, interpolated))

        if entropy_weight > 0.:
            theta = einx.sum('i', state * rotated_goal).clamp(-1. + epsilon, 1. - epsilon).acos()
            std = (entropy_weight / (2. * (theta / (num_selected_waypoints + 1)).cos().clamp(min = epsilon))).sqrt()
            means = l2norm(means + std * torch.empty_like(means).normal_(generator = generator))

    waypoints = safe_cat((means, goal[None]), dim = 0)

    if exists(score_fn):
        scores = score_fn(state, waypoints)
        positions = torch.arange(num_selected_waypoints + 1, device = device)
        selected_index = max(torch.where(scores >= threshold, positions, -1).max().item(), 0)
        subgoal = waypoints[selected_index]
    else:
        selected_index = subgoal = None

    return CSlerpPlan(waypoints, num_selected_waypoints, selected_index, subgoal)

# subgoal proposal
#
# direct  - final goal only
# crl     - c-slerp with the learned rotation
# uniform - c-slerp with identity rotation, fixed K

SUBGOAL_METHODS = ('direct', 'crl', 'uniform')

class SubgoalPlan(NamedTuple):
    subgoals: Tensor           # (N, d), final entry is the goal
    num_waypoints: int
    method: str
    selected_index: int | None = None

class SubgoalProposal:

    def __init__(
        self,
        method = 'crl',
        *,
        num_waypoints = None,
        max_num_waypoints = 4,
        waypoint_temperature = 0.1,
        entropy_weight = 0.,
        threshold = 0.4,
        epsilon = 1e-7
    ):
        assert method in SUBGOAL_METHODS

        self.method = method
        self.num_waypoints = num_waypoints
        self.max_num_waypoints = max_num_waypoints
        self.waypoint_temperature = waypoint_temperature
        self.entropy_weight = entropy_weight
        self.threshold = threshold
        self.epsilon = epsilon

    def propose(self, state_embed, goal_embed, *, rotation = None, score_fn = None, generator = None):
        state_embed, goal_embed = map(l2norm, (state_embed, goal_embed))

        if self.method == 'direct':
            return SubgoalPlan(goal_embed[None], 0, self.method)

        if self.method == 'uniform':
            rotation = None
            num_waypoints = default(self.num_waypoints, 2)
        else:
            num_waypoints = self.num_waypoints

        plan = c_slerp_plan(
            state_embed,
            goal_embed,
            rotation = rotation,
            num_waypoints = num_waypoints,
            max_num_waypoints = self.max_num_waypoints,
            waypoint_temperature = self.waypoint_temperature,
            entropy_weight = self.entropy_weight,
            threshold = self.threshold,
            epsilon = self.epsilon,
            score_fn = score_fn,
            generator = generator
        )

        return SubgoalPlan(plan.waypoints, plan.num_waypoints, self.method, plan.selected_index)

    def __call__(self, *args, **kwargs):
        return self.propose(*args, **kwargs)

def propose_subgoals(state_embed, goal_embed, method = 'crl', *, rotation = None, **kwargs):
    return SubgoalProposal(method, **kwargs).propose(state_embed, goal_embed, rotation = rotation)

# x-jepa adapter

def subgoal_fitness(subgoal_embed, reduce_horizon = 'sum'):
    assert subgoal_embed.ndim == 1

    subgoal = l2norm(subgoal_embed)

    def fitness(pred_next_encoded_states):
        score = einx.dot('b p h d, d -> b p h', l2norm(pred_next_encoded_states), subgoal)
        return score[..., -1] if reduce_horizon == 'last' else einx.sum('b p h -> b p', score)

    return fitness

class XJepaCRLPlanner:

    def __init__(self, agent, critic = None, subgoal_proposal = None):
        self.agent = agent
        self.critic = critic
        self.subgoal_proposal = default(subgoal_proposal, SubgoalProposal(method = 'crl'))

    def encode(self, states):
        return l2norm(self.agent.encode_states_noncontextual(states))

    def propose_subgoals(self, state, goal, **kwargs):
        assert state.shape[0] == goal.shape[0] == 1

        state_embed = self.encode(state)[0]
        goal_embed = self.encode(goal)[0]
        rotation = self.critic.rotation if exists(self.critic) else None

        return self.subgoal_proposal.propose(state_embed, goal_embed, rotation = rotation, **kwargs)

    def plan_to_subgoal(self, states, actions, subgoal_embed, **plan_kwargs):
        return self.agent.plan_to_goal_embedding(states, actions, subgoal_embed, **plan_kwargs)

# references
#
# park, son, shin, lee, kim, eysenbach
# hyperspherical contrastive representations unlock planning and exploration
# https://arxiv.org/abs/2610.20409
#
# eysenbach, zhang, levine, salakhutdinov
# contrastive learning as goal-conditioned reinforcement learning
# https://arxiv.org/abs/2206.07568
