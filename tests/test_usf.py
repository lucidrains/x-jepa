import torch
from torch import nn
import torch.nn.functional as F
from torch.testing import assert_close

import einx
from einops import repeat

from x_jepa.usf import UniversalSuccessorFeatures
from x_jepa.x_jepa import Agent, Transformer


def test_usf_td_loss_and_uvfa_readout():
    torch.manual_seed(0)

    usf = UniversalSuccessorFeatures(
        dim_state_latent = 4,
        dim_action = 2,
        hidden_dim = 16,
        discount_factor = 0.9
    )

    states = torch.randn(2, 5, 4)
    actions = torch.randn(2, 4, 2)
    task = torch.randn(2, 4)

    loss = usf.compute_loss(states, actions)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in usf.parameters())

    psi = usf.successor_features(states[:, :4], actions)
    q = usf.q(states[:, :4], actions, task)

    assert psi.shape == (2, 4, 4)
    assert q.shape == (2, 4)
    assert_close(q, einx.dot('b m d, b d -> b m', psi, task))
    assert_close(usf.q(states[:, :4], actions, task[0]), einx.dot('b m d, d -> b m', psi, task[0]))

    batched_task = repeat(task, 'b d -> b m d', m = 4)
    assert_close(usf.q(states[:, :4], actions, batched_task), q)

    # goal vectors are task vectors in the contextual latent space

    assert_close(usf.goal_q(states[:, :4], actions, task), q)

    # scoring a population of candidate actions

    candidates = torch.randn(2, 7, 2)
    scores = usf.score_actions(states[:, 0], candidates, task)

    assert scores.shape == (2, 7)

    # a fully masked transition contributes nothing

    empty_mask = torch.zeros(2, 5, dtype = torch.bool)
    assert usf.compute_loss(states, actions, mask = empty_mask).item() == 0.

    # per-step discounts are sliced to the transition lane

    per_step_discount = torch.rand(2, 5)
    assert torch.isfinite(usf.compute_loss(states, actions, discount = per_step_discount))

    # discount conditioning changes successor feature predictions

    psi_low = usf.successor_features(states[:, :4], actions, discount = 0.5)
    psi_high = usf.successor_features(states[:, :4], actions, discount = 0.99)
    assert not torch.allclose(psi_low, psi_high)

    # dones with full state sequence length (m + 1) are sliced cleanly

    full_dones = torch.zeros(2, 5, dtype = torch.bool)
    assert torch.isfinite(usf.compute_loss(states, actions, dones = full_dones))

    # ema target update

    usf.update()


def test_usf_last_transition_bootstrap():
    torch.manual_seed(0)

    usf = UniversalSuccessorFeatures(
        dim_state_latent = 2,
        dim_action = 1,
        hidden_dim = 8,
        discount_factor = 0.5
    )

    with torch.no_grad():
        for param in usf.net.parameters():
            param.zero_()

    usf.ema_net.ema_model.load_state_dict(usf.net.state_dict())

    states = torch.tensor([[[1., 0.], [0., 1.]]])
    actions = torch.zeros(1, 1, 1)

    loss = usf.compute_loss(states, actions)

    target = states[:, 0] + 0.5 * states[:, 1]
    expected = F.smooth_l1_loss(torch.zeros_like(target), target) * 0.5

    assert_close(loss, expected)

    # a terminal transition has no future feature

    dones = torch.ones(1, 1, dtype = torch.bool)
    terminal_loss = usf.compute_loss(states, actions, dones = dones)
    terminal_expected = F.smooth_l1_loss(torch.zeros_like(states[:, 0]), states[:, 0]) * 0.5

    assert_close(terminal_loss, terminal_expected)


def test_agent_usf_trains_alongside_world_model():
    torch.manual_seed(0)

    def make_agent(use_usf):
        return Agent(
            state_encoder = nn.Linear(16, 32),
            action_encoder = nn.Linear(4, 32),
            model = Transformer(dim = 32, depth = 1, causal = True),
            dim_action = 4,
            use_usf = use_usf,
            usf_loss_weight = 1.,
            usf_hidden_dim = 32
        )

    agent = make_agent(True)

    states = torch.randn(2, 5, 16)
    actions = torch.randn(2, 4, 4).tanh()

    loss, loss_breakdown = agent(states, actions)

    assert torch.isfinite(loss)
    assert torch.isfinite(loss_breakdown.usf)

    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.usf.parameters())

    agent.update()

    # variable length sequences exercise the transition mask

    loss, loss_breakdown = agent(states, actions, lens = torch.tensor([5, 3]))
    assert torch.isfinite(loss)
    assert torch.isfinite(loss_breakdown.usf)

    # the successor model scores candidate actions directly

    state_latent = torch.randn(2, agent.dim_state_latent)
    candidates = torch.randn(2, 7, 4).tanh()
    task = torch.randn(2, agent.usf.feature_dim)

    scores = agent.usf.score_actions(state_latent, candidates, task)
    assert scores.shape == (2, 7)

    # disabled USF is a no-op

    agent_no_usf = make_agent(False)
    assert agent_no_usf.usf is None

    _, loss_breakdown = agent_no_usf(states, actions)
    assert loss_breakdown.usf == 0.


def test_usf_infer_task_generalizes_to_different_rewards():
    torch.manual_seed(0)

    usf = UniversalSuccessorFeatures(
        dim_state_latent = 8,
        dim_action = 2,
        hidden_dim = 16,
        discount_factor = 0.95
    )

    state_latents = torch.randn(4, 32, 8)
    actions = torch.randn(4, 32, 2)

    true_w1 = torch.randn(8)
    true_w2 = torch.randn(8)

    rewards_1 = einx.dot('b n d, d -> b n', state_latents, true_w1)
    rewards_2 = einx.dot('b n d, d -> b n', state_latents, true_w2)

    # single reward function recovery via ridge regression (test-time adjustment)

    inferred_w1 = usf.infer_task(state_latents, rewards_1, ridge_lambda = 1e-5)
    assert_close(inferred_w1, true_w1, atol = 1e-3, rtol = 1e-3)

    # zero-shot Gaussian estimator (ridge_lambda = None)

    zs_w1 = usf.infer_task(state_latents, rewards_1, ridge_lambda = None)
    assert F.cosine_similarity(zs_w1, true_w1, dim = 0) > 0.85

    # multiple simultaneous reward functions (k = 3) in one call

    true_multi = torch.stack((true_w1, true_w2, -true_w1), dim = 0)
    multi_rewards = einx.dot('b n d, k d -> b n k', state_latents, true_multi)

    mask = torch.ones(4, 32, dtype = torch.bool)
    mask[:, -4:] = False

    inferred_multi = usf.infer_task(state_latents, multi_rewards, mask = mask, ridge_lambda = 1e-5)
    assert inferred_multi.shape == (3, 8)
    assert_close(inferred_multi, true_multi, atol = 1e-3, rtol = 1e-3)

    # evaluating Q across all k reward tasks simultaneously

    q_multi = usf.q(state_latents, actions, inferred_multi)
    assert q_multi.shape == (4, 32, 3)
    assert_close(q_multi[..., 0], -q_multi[..., 2], atol = 1e-3, rtol = 1e-3)

