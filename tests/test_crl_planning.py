import torch
from torch import nn
from torch.testing import assert_close
from einops import rearrange

from x_jepa import Agent, FIFOBuffer, SubgoalProposal
class IdentityModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, tokens, return_hiddens = False, return_memories = False, **kwargs):
        out = (tokens, None) if return_hiddens else tokens
        return (out, None) if return_memories else out
class IdentityRNN(nn.Module):
    def forward(self, tokens, memories = None, return_memories = False):
        return (tokens, memories) if return_memories else tokens
class EncodeAction(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    def forward(self, state_action):
        state, action = state_action
        return self.encoder(action)
class PredictNext(EncodeAction):
    def forward(self, state_action):
        state, action = state_action
        return state + self.encoder(action)

def make_agent(dim = 8):
    torch.manual_seed(0)

    state_encoder = nn.Linear(2, dim, bias = False)
    with torch.no_grad():
        orthonormal_basis, _ = torch.linalg.qr(torch.randn(dim, 2))
        state_encoder.weight.copy_(orthonormal_basis)

    action_encoder = nn.Linear(2, dim, bias = False)
    with torch.no_grad():
        action_encoder.weight.copy_(0.1 * state_encoder.weight)

    agent = Agent(
        state_encoder = state_encoder,
        action_encoder = action_encoder,
        model = IdentityModel(dim),
        dim_action = 2,
        dim_state_latent = dim,
        transition_action_space = 'raw',
        state_latent_clamp_value = 0.,
        state_transition = EncodeAction(action_encoder),
        state_transition_for_planning = EncodeAction(action_encoder)
    )

    agent.to_state_latent = nn.Identity()
    agent.state_linear_rnn = IdentityRNN()
    agent.action_linear_rnn = IdentityRNN()
    agent.get_world_model().to_next_encoded_state_pred = PredictNext(action_encoder)
    return agent

def test_crl_planning_end_to_end():
    torch.manual_seed(0)

    agent = make_agent()
    buffer = FIFOBuffer(10_000)
    step = torch.tensor([0.1, 0.0])

    for _ in range(32):
        start = torch.tensor([-0.3, 0.0]) + 0.1 * torch.randn(2)
        buffer.push(torch.stack([start + t * step for t in range(8)]))

    learner = agent.crl_learner(lr = 3e-3)
    first = learner.learn(buffer, steps = 1, batch_size = 256, num_negatives = 8)
    last = learner.learn(buffer, steps = 20, batch_size = 256, num_negatives = 8)
    assert last < first

    planner = agent.crl_planner(learner.critic, SubgoalProposal(method = 'crl', num_waypoints = 2, entropy_weight = 0.))
    start = torch.tensor([[-0.2, 0.0]])
    goal = torch.tensor([[0.4, 0.0]])

    plan = planner.propose_subgoals(start, goal)
    assert plan.subgoals.shape == (3, 8)
    assert_close(plan.subgoals[-1], planner.encode(goal)[0], atol = 1e-5, rtol = 0.)

    current = start
    for subgoal in plan.subgoals:
        actions = planner.plan_to_subgoal(
            rearrange(current, 'b d -> b 1 d'),
            torch.empty(1, 0, 2),
            subgoal,
            horizon = 2,
            pop_size = 32,
            generations = 2
        )
        current = (current + 0.1 * actions.sum(dim = 1)).clamp(-1., 1.)

    assert (current - goal).norm() < (start - goal).norm() - 0.05
