import pytest
param = pytest.mark.parametrize

import torch
from torch import nn, tensor, cat
import torch.nn.functional as F
from torch.testing import assert_close

from einops import reduce, rearrange
from einops.layers.torch import Rearrange

from x_mlps_pytorch import MLP

from x_jepa.x_jepa import Agent, WorldModel, Transformer, Actor, exists, AgentRolloutWrapper, TTTMetaLearningLoss, PrefixStateTransition, compressed_lane_bounds
from x_jepa.min_gru import minGRUBlocks
from x_jepa.regularizers import SigReg, VISReg, uniform_wasserstein_loss
from x_jepa.goals import GoalGenerator, MetricResidualNetwork
from x_jepa.flow_matching import FlowMatching
from x_jepa.latent_action_model import LatentActionModel
from x_jepa.utils import store_experience_in_replay_buffer, Experience, exists
from x_jepa.rl import ppo_loss, tpo_loss, coerce_experience

@param('plan_type', ('no_goal', 'goal', 'custom_goal'))
@param('transition_action_space', ('raw', 'local', 'global'))
@param('use_reg', (False, True))
@param('reg_type', ('sigreg', 'visreg'))
@param('probabilistic', (False, True))
@param('align_pre_state_action_repr', (False, True))
def test_world_model(
    plan_type,
    transition_action_space,
    use_reg,
    reg_type,
    probabilistic,
    align_pre_state_action_repr
):
    model = Transformer(
        dim = 32,
        depth = 1,
        causal = True
    )

    transition_action_is_raw = transition_action_space == 'raw'

    reg = SigReg() if reg_type == 'sigreg' else VISReg()

    world_model = Agent(
        state_encoder = nn.Linear(16, 32),
        action_encoder = nn.Linear(4, 32),
        action_decoder = nn.Linear(8, 4) if not transition_action_is_raw else None,
        transition_action_space = transition_action_space,
        dim_action = 4,
        dim_action_latent = 8,
        model = model,
        reg = reg,
        reg_next_state_weight = float(use_reg),
        reg_next_encoded_weight = float(use_reg),
        action_latent_wasserstein_loss_weight = float(use_reg and not transition_action_is_raw),
        probabilistic_state_transition = probabilistic,
        probabilistic_plan_state_transition = probabilistic,
        state_latent_clamp_value = 10.,
        align_pre_state_action_repr_loss_weight = 1. if align_pre_state_action_repr else 0.,
        align_pre_state_action_repr_sigreg_weight = 1. if align_pre_state_action_repr else 0.
    )

    states = torch.randn(2, 4, 16)
    actions = torch.randn(2, 3, 4).tanh()
    returns = torch.randn(2, 4)

    loss, loss_breakdown = world_model(states, actions, returns = returns)

    assert loss.ndim == 0
    loss.backward()

    # optimizer code

    world_model.update() # maybe update ema

    # planning

    if plan_type == 'goal':
        goal_state = torch.randn(2, 16)
        plan_kwargs = dict(goal_state = goal_state)

    elif plan_type == 'custom_goal':
        goal_state = torch.randn(2, 16)

        def custom_fitness_fn(pred_values, pred_next_encoded_states, encoded_goal):
            dist = torch.nn.functional.mse_loss(pred_next_encoded_states, encoded_goal.expand_as(pred_next_encoded_states), reduction = 'none')
            dist = reduce(dist, 'b p h d -> b p', 'sum')
            values = reduce(pred_values, 'b p h -> b p', 'sum')
            return values - dist

        plan_kwargs = dict(goal_state = goal_state, fitness_fn = custom_fitness_fn)

    else:
        fitness_fn = lambda pred_state_latents: reduce(pred_state_latents, 'b p ... -> b p', 'sum')
        plan_kwargs = dict(fitness_fn = fitness_fn)

    planned_actions = world_model.plan(states[:, :2], actions[:, :1], horizon = 2, **plan_kwargs)

    assert planned_actions.shape == (2, 2, 4)

def test_world_model_train_without_rewards_or_dones():
    """training must work without rewards/dones - the reward and continuation heads are optional,
    and their losses must be exactly zero when no targets are provided"""

    model = Transformer(dim = 32, depth = 1, causal = True)

    wm = Agent(
        state_encoder = nn.Linear(16, 32),
        action_encoder = nn.Linear(4, 32),
        action_decoder = nn.Linear(8, 4),
        transition_action_space = 'local',
        dim_action = 4,
        dim_action_latent = 8,
        model = model
    )

    states = torch.randn(2, 5, 16)
    actions = torch.randn(2, 4, 4).tanh()

    loss, lb = wm(states, actions)

    assert loss.ndim == 0
    assert lb.reward_pred == 0., 'reward head loss must be zero when no rewards are passed'
    assert lb.discount_pred == 0., 'continuation head loss must be zero when no dones are passed'

    loss.backward()
    wm.update()

    # with targets, both heads receive a loss and training still works

    rewards = torch.randn(2, 4)
    dones = torch.randint(0, 2, (2, 4)).bool()
    returns = torch.randn(2, 5)

    loss2, lb2 = wm(states, actions, returns = returns, rewards = rewards, dones = dones)

    assert torch.isfinite(loss2)
    assert lb2.reward_pred.item() > 0., 'reward head must receive a loss when rewards are passed'
    assert lb2.discount_pred.item() > 0., 'continuation head must receive a loss when dones are passed'

    loss2.backward()

def test_plan_rollout_fitness_requires_reward_and_discount_heads():
    """the rollout-return fitness (discounted sum of predicted rewards + bootstrap value) depends on
    the reward / continuation heads - a fitness requesting them from a world model without the heads
    must fail loudly, not silently inject None"""

    def make_wm():
        model = Transformer(dim = 32, depth = 1, causal = True)
        return Agent(
            state_encoder = nn.Linear(16, 32),
            action_encoder = nn.Linear(4, 32),
            action_decoder = nn.Linear(8, 4),
            transition_action_space = 'local',
            dim_action = 4,
            dim_action_latent = 8,
            model = model
        )

    def rollout_fitness(pred_rewards, pred_discounts, pred_values):
        ret = pred_values[:, :, -1]
        for t in range(pred_rewards.shape[-1] - 1, -1, -1):
            ret = pred_rewards[:, :, t] + pred_discounts[:, :, t] * ret
        return ret

    states = torch.randn(2, 3, 16)
    actions = torch.randn(2, 2, 4).tanh()

    # with the default world model (heads present) the rollout-return fitness works

    wm = make_wm()
    planned = wm.plan(states, actions, horizon = 2, pop_size = 8, generations = 2, fitness_fn = rollout_fitness)
    assert planned.shape == (2, 2, 4)

    # a world model without the reward head must be rejected

    wm_no_reward = make_wm()
    wm_no_reward.world_models[0].to_reward_pred = None

    with pytest.raises(AssertionError, match = 'pred_rewards'):
        wm_no_reward.plan(states, actions, horizon = 2, pop_size = 8, generations = 2, fitness_fn = rollout_fitness)

    # a world model without the continuation head must be rejected

    wm_no_discount = make_wm()
    wm_no_discount.world_models[0].to_discount_pred = None

    with pytest.raises(AssertionError, match = 'pred_discounts'):
        wm_no_discount.plan(states, actions, horizon = 2, pop_size = 8, generations = 2, fitness_fn = rollout_fitness)

def test_rollout_fitness_termination_gating():
    """the rollout-return fitness must hard-stop at a predicted termination - rewards after the
    termination flag and the bootstrap value must not be incorporated"""

    from train_montezuma import make_fitness

    fitness_fn = make_fitness(mode = 'rollout', gamma = 0.99)

    b, p, h = 2, 3, 8

    rewards = torch.full((b, p, h), 10.0)
    values = torch.full((b, p, h), 100.0)
    discounts = torch.full((b, p, h), 0.99)

    # no termination anywhere - fitness is the exact discounted sum of rewards + discounted bootstrap

    no_done = fitness_fn(rewards, discounts, values)
    expected = sum(10.0 * 0.99 ** t for t in range(h)) + 100.0 * 0.99 ** h
    assert_close(no_done, torch.full((b, p), expected), atol = 1e-4, rtol = 1e-4)

    # termination predicted at step 3 - rewards at steps 4..7 and the bootstrap must not contribute

    discounts_done = torch.full((b, p, h), 0.99)
    discounts_done[:, :, 3] = 0.1

    done_early = fitness_fn(rewards, discounts_done, values)
    expected_done = sum(10.0 * 0.99 ** t for t in range(4))
    assert_close(done_early, torch.full((b, p), expected_done), atol = 1e-4, rtol = 1e-4)

    # termination on the very first step - only the first reward counts, no bootstrap

    discounts_first = torch.full((b, p, h), 0.99)
    discounts_first[:, :, 0] = 0.1

    done_first = fitness_fn(rewards, discounts_first, values)
    assert_close(done_first, torch.full((b, p), 10.0), atol = 1e-4, rtol = 1e-4)

    # termination on the final step - the last reward still counts, the bootstrap does not

    discounts_last = torch.full((b, p, h), 0.99)
    discounts_last[:, :, -1] = 0.1

    done_last = fitness_fn(rewards, discounts_last, values)
    expected_last = sum(10.0 * 0.99 ** t for t in range(h))
    assert_close(done_last, torch.full((b, p), expected_last), atol = 1e-4, rtol = 1e-4)

@param('transition_action_space', ('raw', 'local', 'global'))
@param('search_space', ('raw', 'local_global', None))
def test_plan_search_spaces(
    transition_action_space,
    search_space
):
    if transition_action_space == 'raw' and search_space == 'local_global':
        pytest.skip('raw transition action space requires raw search space')

    if transition_action_space == 'global' and search_space == 'raw':
        pytest.skip('latent transition action space can only be searched in encoded_latent space for now')

    model = Transformer(
        dim = 32,
        depth = 1,
        causal = True
    )

    transition_action_is_raw = transition_action_space == 'raw'

    world_model = Agent(
        state_encoder = nn.Linear(16, 32),
        action_encoder = nn.Linear(4, 32),
        action_decoder = nn.Linear(8, 4) if not transition_action_is_raw else None,
        transition_action_space = transition_action_space,
        dim_action = 4,
        dim_action_latent = 8,
        model = model,
    )

    states = torch.randn(2, 4, 16)
    actions = torch.randn(2, 3, 4).tanh()

    planned_actions = world_model.plan(
        states[:, :2],
        actions[:, :1],
        horizon = 2,
        search_space = search_space,
        fitness_fn = lambda pred_state_latents: reduce(pred_state_latents, 'b p ... -> b p', 'sum')
    )

    assert planned_actions.shape == (2, 2, 4)

    planned_actions_mppi = world_model.plan(
        states[:, :2],
        actions[:, :1],
        horizon = 2,
        cem_temperature = 1.,
        search_space = search_space,
        fitness_fn = lambda pred_state_latents: reduce(pred_state_latents, 'b p ... -> b p', 'sum')
    )

    assert planned_actions_mppi.shape == (2, 2, 4)

@param('continuous_actions', (True, False))
@param('action_len', (3, 4))
@param('transition_action_space', ('raw', 'local', 'global'))
@param('pass_world_model_hiddens_to_actor', (True, False))
def test_behavior_cloning(
    continuous_actions,
    action_len,
    transition_action_space,
    pass_world_model_hiddens_to_actor
):
    transition_action_is_raw = transition_action_space == 'raw'

    if transition_action_is_raw and not continuous_actions:
        pytest.skip('raw state transition action space requires continuous actions')

    model = Transformer(
        dim = 32,
        depth = 1,
        causal = True
    )

    actor_model = Transformer(
        dim = 32,
        depth = 1,
        causal = True
    )

    dim_action = 4 if continuous_actions else 4
    dim_action_latent = 8

    world_model = Agent(
        state_encoder = nn.Linear(16, 32),
        action_encoder = nn.Linear(dim_action, 32) if continuous_actions else nn.Embedding(dim_action, 32),
        action_decoder = None if transition_action_is_raw else nn.Linear(dim_action_latent, dim_action),
        transition_action_space = transition_action_space,
        dim_action_latent = dim_action_latent,
        model = model,
        add_transformer_actor = True,
        actor_model = actor_model,
        pass_world_model_hiddens_to_actor = pass_world_model_hiddens_to_actor,
        dim_action = dim_action,
        continuous_actions = continuous_actions,
        actor_loss_weights = 1.
    )

    states = torch.randn(2, 4, 16)

    if continuous_actions:
        actions = torch.randn(2, action_len, dim_action).tanh()
    else:
        actions = torch.randint(0, dim_action, (2, action_len))

    loss, _ = world_model(states, actions)

    assert loss.ndim == 0
    loss.backward()

def test_weighted_behavior_cloning():
    """per-trajectory Float['batch'] weight on the behavior-clone loss, generalizing a mask -
    a 0./1. weight recovers masking; the weighted actor loss is returned per-batch (b,) for the
    caller to reduce, and is excluded from the total loss"""

    dim = 32
    dim_action = 2
    horizon = 4
    batch = 4

    model = Transformer(dim = dim, depth = 2, dim_head = 16, heads = 2, causal = True)

    agent = Agent(
        state_encoder = nn.Linear(3, dim),
        action_encoder = nn.Linear(dim_action, dim),
        model = model,
        dim_action = dim_action,
        add_reflexive_actor = True,
        continuous_actions = True
    )

    states = torch.randn(batch, horizon + 1, 3)
    actions = torch.randn(batch, horizon, dim_action)
    rewards = torch.randn(batch, horizon)

    loss_all, lb_all = agent(states, actions, rewards = rewards)
    loss_ones, lb_ones = agent(states, actions, rewards = rewards, behavior_clone_weight = torch.ones(batch))

    assert lb_ones.actor == 0.
    assert lb_ones.actor_losses['reflexive'].shape == (batch,)

    # the per-batch actor loss is excluded from the total loss, equals the ungated one

    assert_close(loss_ones, loss_all - lb_all.actor, atol = 1e-5, rtol = 1e-5)
    assert_close(lb_all.actor, lb_ones.actor_losses['reflexive'].mean(), atol = 1e-4, rtol = 1e-4)

    # 0./1. weights recover masking - only the weighted-in trajectories are behavior cloned

    cumulative_rewards = torch.linspace(0., 1., batch)
    bc_mask = cumulative_rewards >= cumulative_rewards.quantile(0.75)
    bc_mask_float = bc_mask.float()

    loss_partial, lb_partial = agent(states, actions, rewards = rewards, behavior_clone_weight = bc_mask_float)

    assert (lb_partial.actor_losses['reflexive'][~bc_mask] == 0.).all()
    assert (lb_partial.actor_losses['reflexive'][bc_mask] > 0.).all()
    assert_close(lb_partial.actor_losses['reflexive'][bc_mask], lb_ones.actor_losses['reflexive'][bc_mask], atol = 1e-7, rtol = 1e-7)

    # continuous weights scale the per-batch loss exactly

    bc_weight = torch.rand(batch)

    _, lb_weighted = agent(states, actions, rewards = rewards, behavior_clone_weight = bc_weight)

    assert_close(lb_weighted.actor_losses['reflexive'], lb_ones.actor_losses['reflexive'] * bc_weight, atol = 1e-7, rtol = 1e-7)

    # zero-weighted trajectories are excluded from backward - backprop only through the weighted-in

    actor_param = next(agent.actors['reflexive'].parameters())

    agent.zero_grad()
    loss_partial.backward()
    assert not exists(actor_param.grad)

    agent.zero_grad()
    lb_weighted.actor_losses['reflexive'].sum().backward()
    assert exists(actor_param.grad)

def test_prefix_transition_mtp():
    # fast-lewm prefix transition with mtp lookahead, end-to-end: each prefix predicts the next
    # `lookahead` state latents during training, planning still consumes the immediate next latent,
    # and can fall back to a shorter plan-time lookahead via re-anchored re-prediction

    dim = 32
    dim_action = 4
    lookahead = 3

    model = Transformer(
        dim = dim,
        depth = 1,
        causal = True
    )

    agent = Agent(
        state_encoder = nn.Linear(16, dim),
        action_encoder = nn.Linear(dim_action, dim),
        model = model,
        dim_action = dim_action,
        continuous_actions = True,
        transition_action_space = 'raw',
        transition_horizon = 8,
        transition_lookahead = lookahead,
        reg_next_state_weight = 0.1,
        add_reflexive_actor = True
    )

    transition = agent.world_models[0].state_transition

    assert isinstance(transition, PrefixStateTransition)
    assert transition.lookahead == lookahead

    # positional info for the prefix transformer - pope (rotary / polar positions)

    assert exists(transition.transformer.pope)

    # plan-time lookahead is clamped to the trained one - fewer offsets can be requested

    pred = transition.predict(torch.randn(2, 1, dim), torch.randn(2, 4, dim_action), lookahead = 1)
    assert pred.shape == (2, 4, dim)

    pred = transition.predict(torch.randn(2, 1, dim), torch.randn(2, 4, dim_action), lookahead = lookahead)
    assert pred.shape == (2, 4, lookahead, dim)

    states = torch.randn(2, 64, 16)
    actions = torch.randn(2, 63, dim_action).tanh()
    returns = torch.randn(2, 64)
    lens = torch.full((2,), 64, dtype = torch.long)

    optimizer = torch.optim.Adam(agent.parameters(), lr = 3e-3)

    losses = []

    for _ in range(3):
        loss, _ = agent(states, actions, returns = returns, lens = lens, behavior_clone = True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        agent.update()
        losses.append(loss.item())

    assert losses[-1] < losses[0]

    # planning rolls out with the immediate next latent after each prefix - full parallel pass by
    # default, single-step re-anchored fallback with plan_lookahead = 1

    goal_state = torch.randn(2, 16)

    for plan_lookahead in (1, None):
        planned_actions = agent.plan(
            states[:, :2],
            actions[:, :1],
            goal_state = goal_state,
            horizon = 8,
            pop_size = 16,
            generations = 2,
            plan_lookahead = plan_lookahead
        )

        assert planned_actions.shape == (2, 8, dim_action)

    # ema transition used for planning must share the lookahead

    planned_ema_transition = agent.world_models[0].ema_state_transition_for_planning.online_model

    assert planned_ema_transition.lookahead == lookahead

def test_prefix_transition_temporal_compression():
    # fast-lewm prefix transition under temporal compression, end-to-end: each compressed state step
    # is driven by a chunk of `temporal_compression` raw actions flattened into one token - training
    # on the chunked lane, then planning with chunked predicts, and degenerate short windows that
    # fall back to flat per-step actions

    dim = 32
    dim_action = 4

    for temporal_compression in (2, 4):
        model = Transformer(
            dim = dim,
            depth = 1,
            causal = True
        )

        agent = Agent(
            state_encoder = nn.Linear(16, dim),
            action_encoder = nn.Linear(dim_action, dim),
            model = model,
            dim_action = dim_action,
            continuous_actions = True,
            transition_action_space = 'raw',
            transition_horizon = 8,
            transition_lookahead = 2,
            temporal_compression = temporal_compression
        )

        transition = agent.world_models[0].state_transition

        assert isinstance(transition, PrefixStateTransition)
        assert transition.temporal_compression == temporal_compression
        assert transition.to_action_tokens_chunked.in_features == temporal_compression * dim_action

        seq_len = temporal_compression * 3 + 1
        states = torch.randn(2, seq_len, 16)
        actions = torch.randn(2, seq_len - 1, dim_action).tanh()
        lens = torch.full((2,), seq_len, dtype = torch.long)

        optimizer = torch.optim.Adam(agent.parameters(), lr = 3e-3)

        losses = []

        for _ in range(3):
            loss, _ = agent(states, actions, lens = lens, behavior_clone = True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            agent.update()
            losses.append(loss.item())

        assert losses[-1] < losses[0]

        # chunked predict - consumes (h, c, d) actions, returns the immediate next latent per step

        pred = transition.predict(
            torch.randn(2, 1, dim),
            torch.randn(2, 2, temporal_compression, dim_action)
        )

        assert pred.shape == (2, 2, dim)

        # planning searches raw actions; horizon must be divisible by the compression

        horizon = temporal_compression * 2
        planned_actions = agent.plan(
            states[:, :temporal_compression + 1],
            actions[:, :temporal_compression],
            goal_state = torch.zeros(2, 16),
            horizon = horizon,
            pop_size = 16,
            generations = 2
        )

        assert planned_actions.shape == (2, horizon, dim_action)

        # a sequence too short for a full compressed lane (fewer than 2 compression steps) falls
        # back to flat per-step actions for the prefix transition - must not crash

        short_seq_len = temporal_compression + 1
        short_states = torch.randn(2, short_seq_len, 16)
        short_actions = torch.randn(2, short_seq_len - 1, dim_action).tanh()
        short_lens = torch.full((2,), short_seq_len, dtype = torch.long)

        loss, _ = agent(short_states, short_actions, lens = short_lens, behavior_clone = True)

        assert torch.isfinite(loss)
        loss.backward()

@param('reg_type', ('sigreg', 'visreg'))
def test_reg_loss(reg_type):
    from x_jepa.regularizers import sigreg_loss, visreg_loss

    loss_fn = sigreg_loss if reg_type == 'sigreg' else visreg_loss

    x = torch.randn(256, 64).requires_grad_()

    loss = loss_fn(x)

    assert loss.ndim == 0
    loss.backward()

@param('samples', (tensor([[0.]]), tensor([[-0.5], [0.5]])))
def test_uniform_wasserstein_uses_bin_midpoints(samples):
    loss = uniform_wasserstein_loss(samples)
    assert_close(loss, torch.tensor(0.))

def test_linear_rnn_parallel_matches_sequential():
    from x_jepa.min_gru import minGRUBlocks

    batch_size = 2
    seq_len = 10
    dim = 32

    model = minGRUBlocks(dim = dim, depth = 2)

    x = torch.randn(batch_size, seq_len, dim)

    # get parallel

    out = model(x)
    assert out.shape == x.shape

    # get sequential

    out_seqs = []

    kwargs = {}
    for i in range(seq_len):
        x_i = x[:, i:i+1, :]
        out_i, memories = model(x_i, return_memories = True, **kwargs)

        out_seqs.append(out_i)
        kwargs.update(memories = memories)

    out_seq = cat(out_seqs, dim = 1)

    assert torch.allclose(out, out_seq, atol = 1e-5)

def test_goal_flow_match():
    dim = 64
    batch_size = 2

    goal_gen = GoalGenerator(dim = dim)

    flow_matching = FlowMatching(
        model = goal_gen,
        noise_std = 10.
    )

    mock_state_latents = torch.randn(batch_size, dim)
    mock_returns = torch.randn(batch_size)

    loss = flow_matching(
        mock_state_latents,
        returns = mock_returns
    )

    assert loss.ndim == 0

    sampled_goals = flow_matching.sample(
        batch_size = batch_size,
        data_shape = (dim,),
        returns = mock_returns
    )

    assert sampled_goals.shape == (batch_size, dim)

@param('use_pope', (False, True))
@param('use_pseudo_queries', (False, True))
@param('attn_res_dim_head', (1, 64, 512))
def test_transformer(use_pope, use_pseudo_queries, attn_res_dim_head):
    model = Transformer(
        dim = 512,
        depth = 2,
        causal = True,
        use_pope = use_pope,
        use_pseudo_queries = use_pseudo_queries,
        attn_res_dim_head = attn_res_dim_head
    )

    tokens = torch.randn(2, 10, 512)
    out = model(tokens)

    assert out.shape == (2, 10, 512)

    loss = out.sum()
    loss.backward()

@param('use_pope', (False, True))
def test_transformer_sequential_vs_parallel(use_pope):
    model = Transformer(dim = 128, depth = 4, causal = True, use_pope = use_pope)
    model.eval()

    tokens = torch.randn(2, 10, 128)

    parallel_out, parallel_layer_hiddens = model(tokens, return_hiddens = True)

    sequential_out = []
    memories = None

    for step_tokens in rearrange(tokens, 'b n d -> n b 1 d'):
        (step_out, step_layer_hiddens), memories = model(
            step_tokens,
            return_hiddens = True,
            memories = memories,
            return_memories = True
        )
        sequential_out.append(step_out)

    sequential_out = cat(sequential_out, dim = -2)

    assert_close(parallel_out, sequential_out, atol = 1e-4, rtol = 1e-4)

    for parallel_hidden, sequential_hidden in zip(parallel_layer_hiddens, step_layer_hiddens):
        assert_close(parallel_hidden[:, -1:], sequential_hidden, atol = 1e-4, rtol = 1e-4)

@torch.no_grad()
def test_world_model_sequential_vs_parallel():
    model = Transformer(
        dim = 128,
        depth = 2,
        causal = True
    )

    world_model = Agent(
        model = model,
        dim_action = 4,
        state_encoder = nn.Linear(128, 128),
        action_encoder = nn.Linear(4, 128),
        action_decoder = nn.Linear(128, 4),
        transition_action_space = 'local',
        continuous_actions = True
    )

    world_model.eval()

    states = torch.randn(2, 10, 128)
    actions = torch.randn(2, 10, 4)

    parallel_out = world_model(states, actions, return_loss = False)
    parallel_embeds = parallel_out['embeds']

    sequential_embeds = []
    memories = None

    for step_states, step_actions in zip(rearrange(states, 'b n d -> n b 1 d'), rearrange(actions, 'b n d -> n b 1 d')):
        step_out, memories = world_model(
            step_states,
            step_actions,
            return_loss = False,
            memories = memories,
            return_memories = True
        )
        sequential_embeds.append(step_out['embeds'])

    sequential_embeds = cat(sequential_embeds, dim = 1)

    assert_close(parallel_embeds, sequential_embeds, atol = 1e-4, rtol = 1e-4)

@torch.no_grad()
def test_world_model_plan_with_memories():
    model = Transformer(
        dim = 128,
        depth = 2,
        causal = True
    )

    world_model = Agent(
        model = model,
        dim_action = 4,
        state_encoder = nn.Linear(128, 128),
        action_encoder = nn.Linear(4, 128),
        action_decoder = nn.Linear(128, 4),
        transition_action_space = 'local',
        continuous_actions = True
    )

    world_model.eval()

    batch_size = 2
    memories = None
    empty_actions = torch.empty(batch_size, 0, 4)

    def fitness_fn(pred_state_latents):
        return torch.randn(pred_state_latents.shape[:2]) # (b, p)

    for _ in range(5):
        step_state = torch.randn(batch_size, 1, 128)

        planned_actions, memories = world_model.plan(
            states = step_state,
            actions = empty_actions,
            fitness_fn = fitness_fn,
            horizon = 2,
            memories = memories,
            return_memories = True
        )

        assert planned_actions.shape == (batch_size, 2, 4)

        first_action = planned_actions[:, :1]
        assert first_action.shape == (batch_size, 1, 4)

@torch.no_grad()
def test_interact_with_environment():
    import gymnasium as gym

    model = Transformer(
        dim = 128,
        depth = 2,
        causal = True
    )

    world_model = Agent(
        model = model,
        dim_action = 2,
        state_encoder = nn.Linear(4, 128),
        action_encoder = nn.Linear(2, 128),
        action_decoder = nn.Linear(128, 2),
        transition_action_space = 'local',
        continuous_actions = False,
        value_loss_weight = 1.,
        discount_factor = 0.99
    )

    world_model.eval()

    env = gym.make('CartPole-v1')

    def fitness_fn(pred_state_latents):
        return torch.randn(pred_state_latents.shape[:2])

    experience = world_model.interact_with_environment(
        env = env,
        max_steps = 10,
        fitness_fn = fitness_fn,
        horizon = 2
    )

    assert experience.states.shape[1] > 0
    assert experience.actions.shape[1] == experience.states.shape[1]
    assert experience.rewards.shape[1] == experience.states.shape[1]
    assert exists(experience.returns)
    _, returns_tensor = experience.returns
    assert returns_tensor.shape[1] == experience.states.shape[1]

@torch.no_grad()
def test_interact_with_environment_commit_k_steps():
    import gymnasium as gym

    model = Transformer(
        dim = 128,
        depth = 2,
        causal = True
    )

    world_model = Agent(
        model = model,
        dim_action = 2,
        state_encoder = nn.Linear(4, 128),
        action_encoder = nn.Linear(2, 128),
        action_decoder = nn.Linear(128, 2),
        transition_action_space = 'local',
        continuous_actions = False,
        value_loss_weight = 1.,
        discount_factor = 0.99
    )

    world_model.eval()

    class TerminatingOnFirstStep(gym.Wrapper):
        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            return obs, reward, True, truncated, info

    env = gym.vector.SyncVectorEnv([
        lambda: TerminatingOnFirstStep(gym.make('CartPole-v1')),
        lambda: gym.make('CartPole-v1')
    ])

    plan_count = 0
    orig_plan = world_model.plan

    def tracking_plan(*args, **kwargs):
        nonlocal plan_count
        plan_count += 1
        return orig_plan(*args, **kwargs)

    world_model.plan = tracking_plan

    def fitness_fn(pred_state_latents):
        return torch.randn(pred_state_latents.shape[:2])

    experience = world_model.interact_with_environment(
        env = env,
        max_steps = 4,
        fitness_fn = fitness_fn,
        commit_k_steps = 2
    )

    assert experience.states.shape[1] == 4
    # With commit_k_steps = 2 for 4 steps, plan is called once per commit boundary (steps 0 and
    # 2); a dead env freezes and consumes the remaining committed plan, so there is no mid-commit
    # replan on termination
    assert plan_count == 2


@param('complex_sensory', (False, True))
def test_multimodal(complex_sensory):
    image_encoder = nn.Sequential(Rearrange('... c h w -> ... (c h w)'), nn.Linear(48, 256))
    vector_encoder = nn.Linear(128, 256)

    model = Transformer(dim = 256, depth = 2)

    actor_model = Transformer(dim = 256, depth = 2)
    if complex_sensory:
        state_encoders = [image_encoder, image_encoder, vector_encoder, vector_encoder]
    else:
        state_encoders = [image_encoder, vector_encoder]

    world_model = Agent(
        state_encoder = state_encoders,
        action_encoder = nn.Linear(64, 256),
        model = model,
        dim_action = 64,
        add_transformer_actor = True,
        actor_model = actor_model,
        pass_sensory_hiddens_to_world_model = True
    )

    if complex_sensory:
        images1 = torch.randn(2, 5, 3, 4, 4)
        images2 = torch.randn(2, 5, 3, 4, 4)
        vectors1 = torch.randn(2, 5, 128)
        vectors2 = torch.randn(2, 5, 128)
        states = [images1, images2, vectors1, vectors2]
    else:
        images = torch.randn(2, 5, 3, 4, 4)
        vectors = torch.randn(2, 5, 128)
        states = [images, vectors]

    actions = torch.randn(2, 4, 64)

    loss, losses = world_model(states, actions, behavior_clone = True)
    loss.backward()

    assert losses.actor > 0.

    if complex_sensory:
        goal_images1 = torch.randn(2, 3, 4, 4)
        goal_images2 = torch.randn(2, 3, 4, 4)
        goal_vectors1 = torch.randn(2, 128)
        goal_vectors2 = torch.randn(2, 128)
        goal_states = [goal_images1, goal_images2, goal_vectors1, goal_vectors2]
        states_plan = [s[:, :2] for s in states]
    else:
        goal_images = torch.randn(2, 3, 4, 4)
        goal_vectors = torch.randn(2, 128)
        goal_states = [goal_images, goal_vectors]
        states_plan = [images[:, :2], vectors[:, :2]]

    planned_actions = world_model.plan(
        states = states_plan,
        actions = actions[:, :1],
        goal_state = goal_states,
        horizon = 2
    )

    assert planned_actions.shape[-1] == 64

def test_vljepa_cross_sensory_alignment():
    # eyes, ears, proprioception

    eyes = nn.Linear(3, 256)
    ear = nn.Linear(5, 256)
    proprioception = nn.Linear(8, 256)

    model = Transformer(
        dim = 256,
        depth = 2
    )

    world_model = Agent(
        model = model,
        state_encoder = nn.ModuleList([eyes, ear, proprioception]),
        action_encoder = nn.Linear(4, 256),
        dim_action = 4,
        num_sensory_views = (2, 1, 1),
        cross_sensory_align_pairs = [(0, 1), (0, 2)],
        cross_sensory_align_loss_weight = 1.,
        cross_sensory_align_sigreg_weight = 1.
    )

    states = (
        (torch.randn(2, 2, 3), torch.randn(2, 2, 3)),
        torch.randn(2, 2, 5),
        torch.randn(2, 2, 8)
    )

    actions = torch.randn(2, 2, 4)

    loss, losses = world_model(
        states = states,
        actions = actions
    )

    assert len(losses.cross_sensory_align_breakdown) == 2

    loss.backward()

@torch.no_grad()
def test_interact_with_environment_multimodal():
    import gymnasium as gym

    model = Transformer(
        dim = 128,
        depth = 2,
        causal = True
    )

    world_model = Agent(
        model = model,
        dim_action = 2,
        state_encoder = nn.ModuleList([
            nn.Linear(4, 128),
            nn.Sequential(Rearrange('... c h w -> ... (c h w)'), nn.Linear(48, 128))
        ]),
        action_encoder = nn.Linear(2, 128),
        action_decoder = nn.Linear(128, 2),
        transition_action_space = 'local',
        continuous_actions = False,
        value_loss_weight = 1.,
        discount_factor = 0.99
    )

    world_model.eval()

    class MultimodalEnvWrapper(gym.ObservationWrapper):
        def observation(self, obs):
            return [torch.tensor(obs).float(), torch.randn(3, 4, 4)]

    env = MultimodalEnvWrapper(gym.make('CartPole-v1'))

    def fitness_fn(pred_state_latents):
        return torch.randn(pred_state_latents.shape[:2])

    experience = world_model.interact_with_environment(
        env = env,
        max_steps = 10,
        fitness_fn = fitness_fn,
        horizon = 2
    )

    assert experience.states[0].shape[1] > 0
    assert experience.actions.shape[1] == experience.states[0].shape[1]
    assert experience.rewards.shape[1] == experience.states[0].shape[1]
    assert exists(experience.returns)
    _, returns_tensor = experience.returns
    assert returns_tensor.shape[1] == experience.states[0].shape[1]

@param('has_actions', (False, True))
def test_world_model_with_intrinsics(has_actions):
    from copy import deepcopy
    from x_jepa.intrinsic import CoinFlipNetwork

    dim = 256
    action_dim = 4
    model = Transformer(
        dim = dim,
        depth = 2,
        causal = True
    )

    in_dim = (dim + action_dim) if has_actions else dim
    cfn_net = MLP(in_dim, 64, dim)
    coin_flip_network = CoinFlipNetwork(net = cfn_net, dim = dim, accepts_actions = has_actions)

    world_model = Agent(
        model = model,
        dim_action = action_dim,
        state_encoder = nn.Linear(128, dim),
        action_encoder = nn.Linear(action_dim, dim),
        action_decoder = nn.Linear(dim, action_dim),
        transition_action_space = 'local',
        continuous_actions = True,
        intrinsics = [coin_flip_network],
        intrinsic_loss_weight = 1.0,
        intrinsic_frac_gradient = 0.1
    )

    world_model.eval()

    states = torch.randn(2, 5, 128)
    actions = torch.randn(2, 4, action_dim)

    # Test forward pass with intrinsics loss
    loss, loss_breakdown = world_model(states, actions)
    assert loss.dim() == 0
    assert loss_breakdown.intrinsics.dim() == 0
    assert len(loss_breakdown.intrinsics_breakdown) == 1

    # Test planning pass with intrinsic bonus dependency injection
    def fitness_fn(pred_intrinsic_bonuses):
        bonus = pred_intrinsic_bonuses[0]
        return bonus.sum(dim = -1)

    planned_actions = world_model.plan(
        states,
        actions,
        fitness_fn = fitness_fn,
        horizon = 3,
        pop_size = 4,
        generations = 1
    )

    assert planned_actions.shape == (2, 3, action_dim)

def test_reflexive_actor_and_planning():
    dim = 128
    dim_action = 4
    model = Transformer(
        dim = dim,
        depth = 2,
        causal = True
    )

    world_model = Agent(
        model = model,
        dim_action = dim_action,
        state_encoder = nn.Linear(64, dim),
        action_encoder = nn.Linear(dim_action, dim),
        action_decoder = nn.Linear(dim, dim_action),
        transition_action_space = 'local',
        continuous_actions = True,
        add_reflexive_actor = True,
        actor_loss_weights = 1.
    )

    world_model.eval()

    states = torch.randn(2, 5, 64)
    actions = torch.randn(2, 4, dim_action)

    # Test forward pass with reflexive actor loss
    loss, loss_breakdown = world_model(states, actions, behavior_clone = True)
    assert loss.dim() == 0
    assert loss_breakdown.actor_losses['reflexive'] > 0.

    def fitness_fn(pred_state_latents):
        return torch.randn(pred_state_latents.shape[:2])

    planned_actions = world_model.plan(
        states = states[:, :1],
        actions = torch.empty(2, 0, dim_action),
        fitness_fn = fitness_fn,
        horizon = 3,
        pop_size = 4,
        generations = 2,
        seed_with_actor = 'reflexive',
        actor_temperature = 0.5
    )

    assert planned_actions.shape == (2, 3, dim_action)

def test_transformer_actor_sequential_vs_parallel():
    dim = 128
    dim_action = 4

    world_model = Agent(
        model = Transformer(dim = dim, depth = 2, causal = True),
        dim_action = dim_action,
        state_encoder = nn.Linear(64, dim),
        action_encoder = nn.Linear(dim_action, dim),
        action_decoder = nn.Linear(dim, dim_action),
        transition_action_space = 'local',
        continuous_actions = True,
        add_transformer_actor = True,
        actor_model = Transformer(dim = dim, depth = 2, causal = True),
        pass_world_model_hiddens_to_actor = False,
        actor_loss_weights = 1.
    )

    world_model.eval()

    transformer_actor = world_model.actors['transformer']

    states = torch.randn(2, 5, 64)
    actions = torch.randn(2, 4, dim_action)

    state_tokens, _, _ = world_model.encode_states(world_model.state_encoder, states)
    state_latents = world_model.to_state_latent(state_tokens)

    action_tokens = world_model.action_encoder(actions)
    action_cond = world_model.to_action_latent(action_tokens)

    # parallel forward - states and actions must have matching seq len for interleaving

    num_steps = actions.shape[1]

    parallel_preds, parallel_memories = transformer_actor.get_action_preds(
        state_latents = state_latents[:, :num_steps],
        state_tokens = state_tokens[:, :num_steps],
        action_cond = action_cond,
        action_tokens = action_tokens,
        return_memories = True
    )

    # sequential forward

    sequential_preds = []
    memories = None

    for i in range(num_steps):
        step_preds, memories = transformer_actor.get_action_preds(
            state_latents = state_latents[:, i:i + 1],
            state_tokens = state_tokens[:, i:i + 1],
            action_cond = action_cond[:, i:i + 1],
            action_tokens = action_tokens[:, i:i + 1],
            memories = memories,
            return_memories = True
        )

        sequential_preds.append(step_preds[:, 0])

    sequential_preds = torch.stack(sequential_preds, dim = 1)

    assert torch.allclose(parallel_preds, sequential_preds, atol = 1e-4)

class MinGRUActor(Actor):
    def __init__(self, dim_state_latent, dim_action, continuous_actions, depth = 2, action_eps = 1e-5):
        super().__init__(continuous_actions, dim_action, action_eps = action_eps)
        dim_out = dim_action * 2 if continuous_actions else dim_action
        self.gru = minGRUBlocks(dim = dim_state_latent, depth = depth)
        self.to_action_pred = nn.Linear(dim_state_latent, dim_out)

    def get_action_preds(self, state_latents, memories = None, return_memories = False, **kwargs):
        out, next_memories = self.gru(state_latents, memories = memories, return_memories = True)
        pred = self.to_action_pred(out)

        if not return_memories:
            return pred

        return pred, next_memories

def test_custom_mingru_actor():
    dim = 128
    dim_action = 4

    mingru_actor = MinGRUActor(
        dim_state_latent = dim,
        dim_action = dim_action,
        continuous_actions = True,
        depth = 2
    )

    world_model = Agent(
        model = Transformer(dim = dim, depth = 2, causal = True),
        dim_action = dim_action,
        state_encoder = nn.Linear(64, dim),
        action_encoder = nn.Linear(dim_action, dim),
        action_decoder = nn.Linear(dim, dim_action),
        transition_action_space = 'local',
        continuous_actions = True,
        actors = dict(mingru = mingru_actor),
        actor_loss_weights = 1.
    )

    states = torch.randn(2, 5, 64)
    actions = torch.randn(2, 4, dim_action)

    # behavior clone

    loss, loss_breakdown = world_model(states, actions, behavior_clone = 'mingru')
    assert loss.dim() == 0
    assert loss_breakdown.actor_losses['mingru'] > 0.

    # planning

    world_model.eval()

    planned_actions = world_model.plan(
        states = states[:, :1],
        actions = torch.empty(2, 0, dim_action),
        fitness_fn = lambda pred_state_latents: torch.randn(pred_state_latents.shape[:2]),
        horizon = 3,
        pop_size = 4,
        generations = 2,
        seed_with_actor = 'mingru',
        actor_temperature = 0.5
    )

    assert planned_actions.shape == (2, 3, dim_action)

@param('use_perception_film', [True, False])
@param('episodic_mem_len', [0, 4])
def test_world_model_ttt(use_perception_film, episodic_mem_len):
    dim = 256

    # transformer

    model = Transformer(
        dim = dim,
        depth = 2,
        causal = True
    )

    # world model

    wm = Agent(
        model = model,
        state_encoder = nn.Linear(32, dim),
        action_encoder = nn.Linear(8, dim),
        dim_action = 8,
        transition_action_space = 'raw',
        continuous_actions = True,
        use_perception_film = use_perception_film,
        episodic_mem_len = episodic_mem_len
    )

    # ttt meta learning loss

    ttt_loss = TTTMetaLearningLoss(
        dim = dim,
        num_classes = 64,
        depth = 1,
        heads = 4
    )

    # wrap with rollout wrapper

    # targeting the first Linear projection of the feedforward block in the first transformer layer
    ttt_module_paths = ('model.layers.0.2.1',)

    wrapper = AgentRolloutWrapper(
        agent = wm,
        chunk_size = 16,
        tbptt_steps = 2,
        ttt_module_paths = ttt_module_paths,
        ttt_loss_module = ttt_loss,
        ttt_update_perception_film = use_perception_film,
        ttt_lr = 1e-3
    )

    # simulate rollout

    batch_size = 2

    # step 1

    states_step1 = torch.randn(batch_size, 2, 32)
    actions_step1 = torch.randn(batch_size, 2, 8)

    out1, memories = wrapper(states_step1, actions_step1, return_memories = True, return_loss = True)

    # ensure ttt params initialized

    check_module_name = 'perception_film' if use_perception_film else 'world_models.0.model.layers.0.2.1'
    ttt_wrapper = wrapper.ttt_trainer.ttt_wrappers[check_module_name]
    assert exists(ttt_wrapper.batch_params)

    # step 2

    states_step2 = torch.randn(batch_size, 2, 32)
    actions_step2 = torch.randn(batch_size, 2, 8)

    # save params for comparison

    params_before_step2 = {k: v.clone() for k, v in ttt_wrapper.batch_params.items()}

    out2, memories = wrapper(states_step2, actions_step2, memories = memories, return_memories = True, return_loss = True)

    # ensure ttt params updated

    params_after_step2 = ttt_wrapper.batch_params

    updated = False
    for k in params_before_step2.keys():
        if not torch.allclose(params_before_step2[k], params_after_step2[k]):
            updated = True
            break

    assert updated

def test_agent_temporal_compression_tuple():
    dim = 32
    model = Transformer(dim = dim, depth = 2, heads = 4, dim_head = 8)

    agent = Agent(
        model = model,
        state_encoder = nn.Linear(32, dim),
        action_encoder = nn.Linear(8, dim),
        dim_action = 8,
        transition_action_space = 'raw',
        continuous_actions = True,
        temporal_compression = (1, 2, 4)
    )

    assert len(agent.world_models) == 3
    assert agent.world_models[0].temporal_compression == 1
    assert agent.world_models[1].temporal_compression == 2
    assert agent.world_models[2].temporal_compression == 4

def test_plan_with_gradient_descent():
    model = Transformer(
        dim = 128,
        depth = 2,
        causal = True
    )

    world_model = Agent(
        model = model,
        dim_action = 4,
        state_encoder = nn.Linear(128, 128),
        action_encoder = nn.Linear(4, 128),
        action_decoder = nn.Linear(128, 4),
        transition_action_space = 'local',
        continuous_actions = True
    )

    world_model.eval()

    batch_size = 2
    empty_actions = torch.empty(batch_size, 0, 4)
    step_state = torch.randn(batch_size, 1, 128)
    goal_state = torch.randn(batch_size, 128)

    planned_actions = world_model.plan(
        states = step_state,
        actions = empty_actions,
        goal_state = goal_state,
        horizon = 2,
        pop_size = 1,
        generations = 1,
        gradient_steps = 5,
        gradient_lr = 0.1
    )

    assert planned_actions.shape == (batch_size, 2, 4)

def test_plan_mixed_cem_and_gradient_descent():
    model = Transformer(
        dim = 128,
        depth = 2,
        causal = True
    )

    world_model = Agent(
        model = model,
        dim_action = 4,
        state_encoder = nn.Linear(128, 128),
        action_encoder = nn.Linear(4, 128),
        action_decoder = nn.Linear(128, 4),
        transition_action_space = 'local',
        continuous_actions = True
    )

    world_model.eval()

    batch_size = 2
    empty_actions = torch.empty(batch_size, 0, 4)
    step_state = torch.randn(batch_size, 1, 128)
    goal_state = torch.randn(batch_size, 128)

    planned_actions = world_model.plan(
        states = step_state,
        actions = empty_actions,
        goal_state = goal_state,
        horizon = 2,
        pop_size = 256,
        generations = 2,
        gradient_steps = 10,
        gradient_lr = 0.1
    )

    assert planned_actions.shape == (batch_size, 2, 4)

@param('sensory_dropout', (0., 0.5))
def test_sensory_dropout(sensory_dropout):
    model = Transformer(
        dim = 128,
        depth = 2,
        causal = True
    )

    world_model = Agent(
        model = model,
        dim_action = 4,
        state_encoder = nn.ModuleList([nn.Linear(64, 128), nn.Linear(64, 128)]),
        action_encoder = nn.Linear(4, 128),
        action_decoder = nn.Linear(128, 4),
        transition_action_space = 'local',
        continuous_actions = True,
        pass_sensory_hiddens_to_world_model = True,
        sensory_dropout_prob = sensory_dropout
    )

    states = [torch.randn(2, 5, 64), torch.randn(2, 5, 64)]
    actions = torch.randn(2, 4, 4)

    world_model.train()
    loss, _ = world_model(states, actions)
    loss.backward()

    world_model.eval()
    with torch.no_grad():
        out = world_model(states, actions, return_loss = False)
        assert out['embeds'].shape == (2, 10, 128)

@param('seq_len', (None, 4, 10))
def test_action_flow_matching(seq_len):
    dim_action = 8
    dim_state_latent = 16

    model = LatentActionModel(
        dim_action,
        dim_state_latent = dim_state_latent
    )
    flow_matching = model

    batch_size = 16

    # train the model on dummy continuous actions

    optimizer = torch.optim.Adam(flow_matching.parameters(), lr = 1e-2)

    # create some dummy expert actions

    if exists(seq_len):
        expert_actions = torch.rand(batch_size, seq_len, dim_action) * 2 - 1
        expert_state_latents = torch.randn(batch_size, seq_len, dim_state_latent)
    else:
        expert_actions = torch.rand(batch_size, dim_action) * 2 - 1
        expert_state_latents = torch.randn(batch_size, dim_state_latent)

    for _ in range(50):
        optimizer.zero_grad()
        loss = flow_matching(
            expert_actions,
            state_latent = expert_state_latents,
            cond_drop_prob = 0.25
        )
        loss.backward()
        optimizer.step()

    # embed real actions back to gaussian noise using invert method

    test_actions = expert_actions[:4]
    test_state_latents = expert_state_latents[:4]

    # invert the actions back to noise

    latent_noise = flow_matching.invert(
        test_actions,
        state_latent = test_state_latents,
        steps = 32
    )

    # sample back from that noise (conditioned on state_latent)

    reconstructed_actions = flow_matching.sample(
        steps = 32,
        batch_size = test_actions.shape[0],
        noise = latent_noise,
        state_latent = test_state_latents
    )

    assert reconstructed_actions.shape == test_actions.shape

    # sample back unconditionally (null state_latent)

    uncond_reconstructed_actions = flow_matching.sample(
        steps = 32,
        batch_size = test_actions.shape[0],
        noise = latent_noise
    )

    assert uncond_reconstructed_actions.shape == test_actions.shape

    # demonstrate that LatentActionModel can take in variable length sequences

    if exists(seq_len):
        seq_actions = torch.randn(batch_size, seq_len, dim_action)
        seq_times = torch.rand(batch_size)
        seq_state_latents = torch.randn(batch_size, seq_len, dim_state_latent)

        seq_out = model.model(seq_actions, seq_times, state_latent = seq_state_latents)
        assert seq_out.shape == (batch_size, seq_len, dim_action)

@param('seq_len', (4, 10))
def test_latent_action_model_prepend_and_causal(seq_len):
    dim_action = 8
    dim_state_latent = 16

    model = LatentActionModel(
        dim_action,
        dim_state_latent = dim_state_latent,
        prepend_state_latent = True,
        causal_mask = True
    )

    batch_size = 2
    actions = torch.randn(batch_size, seq_len, dim_action)
    state_latents = torch.randn(batch_size, dim_state_latent)

    loss = model(
        actions,
        state_latent = state_latents
    )

    assert loss.ndim == 0
    loss.backward()

    noise = torch.randn(batch_size, seq_len, dim_action)
    sampled_actions = model.sample(
        steps = 2,
        batch_size = batch_size,
        noise = noise,
        state_latent = state_latents
    )

    assert sampled_actions.shape == (batch_size, seq_len, dim_action)

@param('use_mrn', (True, False))
def test_mrn_world_model(use_mrn):
    model = Transformer(
        dim = 512,
        depth = 2,
        causal = True
    )

    mrn = None
    if use_mrn:
        mrn = MetricResidualNetwork(
            sym_network = MLP(512, 256, 512),
            asym_network = MLP(512, 256, 512)
        )

    world_model = Agent(
        state_encoder = nn.Linear(128, 512),
        action_encoder = nn.Linear(20, 512),
        action_decoder = nn.Linear(32, 20),
        transition_action_space = 'local',
        dim_action = 20,
        dim_action_latent = 32,
        model = model,
        quasimetric_distance_network = mrn
    )

    states = torch.randn(2, 4, 128)
    actions = torch.randn(2, 3, 20).tanh()

    loss, loss_breakdown = world_model(states, actions)
    assert loss.ndim == 0
    loss.backward()

    # plan

    planned_actions = world_model.plan(
        states[:, :2],
        actions[:, :1],
        horizon = 3,
        goal_state = torch.randn(2, 128)
    )

    assert planned_actions.shape == (2, 3, 20)

@param('rl_type', ('ppo', 'tpo'))
@param('is_vector_env', (False, True))
@param('use_var_lens', (False, True))
def test_end_to_end_bc_rl_finetune_flow(rl_type, is_vector_env, use_var_lens):
    import tempfile
    import gymnasium as gym
    from torch.optim import Adam

    obs_dim = 4
    action_dim = 2
    dim_model = 64
    horizon = 5

    env = gym.vector.SyncVectorEnv([lambda: gym.make('CartPole-v1')] * 2) if is_vector_env else gym.make('CartPole-v1')

    # instantiate world model with transformer backbone and reflexive actor

    model = Transformer(
        dim = dim_model,
        depth = 2,
        causal = True
    )

    world_model = Agent(
        state_encoder = nn.Linear(obs_dim, dim_model),
        action_encoder = nn.Embedding(action_dim, dim_model),
        action_decoder = nn.Linear(dim_model, action_dim),
        transition_action_space = 'local',
        dim_action = action_dim,
        model = model,
        add_reflexive_actor = True,
        continuous_actions = False
    )

    wm_optimizer = Adam(world_model.parameters(), lr = 1e-3)

    # system 2 exploration / trial & error rollouts using cem planning

    planning_experience = world_model.interact_with_environment(
        env,
        max_steps = horizon,
        actor_module = None, # cem planning
        fitness_fn = lambda pred_state_latents: pred_state_latents.sum(dim = (-1, -2))
    )

    assert planning_experience.states.ndim == 3

    # store in memmap replay buffer & train world model

    with tempfile.TemporaryDirectory() as tmpdir:
        buffer = store_experience_in_replay_buffer(
            planning_experience,
            1000,
            100000,
            folder = tmpdir,
            overwrite = True
        )

        reloaded_exp = coerce_experience(buffer.get_all_data())
        assert reloaded_exp.states.shape == planning_experience.states.shape

        lens = reloaded_exp.episode_len if use_var_lens else None
        wm_loss, _ = world_model(reloaded_exp.states, reloaded_exp.actions, lens = lens)
        assert wm_loss.ndim == 0
        wm_loss.backward()
        wm_optimizer.step()

    # behavior clone planned actions into reflexive actor

    actor_optimizer = Adam(world_model.actors['reflexive'].parameters(), lr = 1e-3)

    lens = planning_experience.episode_len if use_var_lens else None
    bc_loss, _ = world_model(
        planning_experience.states,
        planning_experience.actions,
        lens = lens,
        behavior_clone = True
    )
    assert bc_loss.ndim == 0
    bc_loss.backward()
    actor_optimizer.step()

    # roll out bc actor against environment, gathering log probabilities

    bc_experience = world_model.interact_with_environment(
        env,
        max_steps = horizon,
        actor_module = 'reflexive'
    )

    assert exists(bc_experience.actor_log_probs)

    # store bc rollout experience in memmap replay buffer

    with tempfile.TemporaryDirectory() as tmpdir:
        bc_buffer = store_experience_in_replay_buffer(
            bc_experience,
            1000,
            100000,
            folder = tmpdir,
            overwrite = True
        )
        if not is_vector_env:
            store_experience_in_replay_buffer(
                bc_experience,
                1000,
                100000,
                folder = tmpdir,
                buffer = bc_buffer
            )
        fetched_bc_exp = coerce_experience(bc_buffer.get_all_data())
        assert exists(fetched_bc_exp.actor_log_probs)

    if not use_var_lens:
        fetched_bc_exp = fetched_bc_exp._replace(episode_len = None)

    # rl fine-tuning step

    actor = world_model.actors['reflexive']

    if rl_type == 'ppo':
        optimizer = Adam(list(actor.parameters()) + list(world_model.value_network.parameters()), lr = 1e-3)
        loss_fn = ppo_loss
    else:
        optimizer = Adam(actor.parameters(), lr = 1e-3)
        loss_fn = tpo_loss

    optimizer.zero_grad()
    loss = loss_fn(world_model, fetched_bc_exp, actor_module = 'reflexive')
    assert loss.ndim == 0
    loss.backward()
    optimizer.step()

    # continue to do rollouts with fine-tuned actor and strengthen world model

    fine_tuned_exp = world_model.interact_with_environment(
        env,
        max_steps = horizon,
        actor_module = 'reflexive'
    )

    wm_optimizer.zero_grad()
    lens = fine_tuned_exp.episode_len if use_var_lens else None
    fine_tuned_wm_loss, _ = world_model(fine_tuned_exp.states, fine_tuned_exp.actions, lens = lens, behavior_clone = False)
    assert fine_tuned_wm_loss.ndim == 0
    fine_tuned_wm_loss.backward()
    wm_optimizer.step()

    # train latent action model on fine-tuned actions

    init_state_latent = fine_tuned_exp.state_latents[:, 0]

    latent_action_model = LatentActionModel(
        dim = dim_model,
        dim_state_latent = dim_model,
        prepend_state_latent = True,
        causal_mask = True
    )

    lam_optimizer = Adam(latent_action_model.parameters(), lr = 1e-3)

    fine_tuned_action_latents = world_model.to_action_latent(world_model.encode_actions(fine_tuned_exp.actions))

    lam_loss = latent_action_model(
        fine_tuned_action_latents,
        state_latent = init_state_latent
    )
    assert lam_loss.ndim == 0
    lam_loss.backward()
    lam_optimizer.step()

    # plan with latent action model generating candidate proposal actions

    planned_actions = world_model.plan(
        fine_tuned_exp.states[:, :2],
        fine_tuned_exp.actions[:, :1],
        horizon = horizon,
        latent_action_model = latent_action_model,
        latent_action_model_steps = 2,
        latent_action_model_chunk_size = (3, 2),
        fitness_fn = lambda pred_state_latents: pred_state_latents.sum(dim = (-1, -2))
    )

    assert planned_actions.shape == (fine_tuned_exp.states.shape[0], horizon)

@param('has_actor_log_probs', (True, False))
@param('has_rewards', (True, False))
@param('has_returns', (True, False))
@param('has_state_latents', (True, False))
def test_replay_buffer_serialization(has_actor_log_probs, has_rewards, has_returns, has_state_latents):
    import tempfile
    batch_size = 2
    horizon = 5
    states = torch.randn(batch_size, horizon, 16)
    actions = torch.randn(batch_size, horizon, 4)

    exp = Experience(
        states = states,
        actions = actions,
        actor_log_probs = torch.randn(batch_size, horizon) if has_actor_log_probs else None,
        rewards = torch.randn(batch_size, horizon) if has_rewards else None,
        returns = torch.randn(batch_size, horizon) if has_returns else None,
        state_latents = torch.randn(batch_size, horizon, 32) if has_state_latents else None,
        cumulative_rewards = torch.randn(batch_size),
        episode_len = torch.randint(1, horizon, (batch_size,))
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        buffer = store_experience_in_replay_buffer(exp, 1000, 100000, folder = tmpdir, overwrite = True)
        reloaded_exp = coerce_experience(buffer.get_all_data())

        assert torch.allclose(reloaded_exp.states, exp.states)
        assert torch.allclose(reloaded_exp.actions, exp.actions)

        if has_actor_log_probs:
            assert torch.allclose(reloaded_exp.actor_log_probs, exp.actor_log_probs)
        else:
            assert not exists(reloaded_exp.actor_log_probs)

        if has_state_latents:
            assert torch.allclose(reloaded_exp.state_latents, exp.state_latents)
        else:
            assert not exists(reloaded_exp.state_latents)

def test_replay_buffer_schema_evolution():
    import tempfile
    batch_size = 2
    horizon = 5

    states = torch.randn(batch_size, horizon, 16)
    actions = torch.randn(batch_size, horizon, 4)

    exp_phase1 = Experience(
        states = states,
        actions = actions,
        actor_log_probs = None,
        cumulative_rewards = torch.tensor([1.0, 2.0]),
        episode_len = torch.tensor([5, 5])
    )

    exp_phase2 = Experience(
        states = states,
        actions = actions,
        actor_log_probs = torch.randn(batch_size, horizon),
        cumulative_rewards = torch.tensor([3.0, 4.0]),
        episode_len = torch.tensor([5, 5])
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        buffer = store_experience_in_replay_buffer(exp_phase1, 100, 50, folder = tmpdir, overwrite = True)
        assert 'actor_log_probs' not in buffer.fieldnames

        buffer = store_experience_in_replay_buffer(exp_phase2, 100, 50, folder = tmpdir, buffer = buffer)
        reloaded_exp = coerce_experience(buffer.get_all_data())

        assert reloaded_exp.states.shape[0] == 4
        assert not exists(reloaded_exp.actor_log_probs)

def test_actor_residual_state_encoder():
    dim = 64
    obs_dim = 8
    action_dim = 2

    wm = Agent(
        state_encoder = nn.Linear(obs_dim, dim),
        action_encoder = nn.Linear(action_dim, dim),
        model = Transformer(dim = dim, depth = 2, causal = True),
        dim_action = action_dim,
        continuous_actions = True,
        transition_action_space = 'raw',
        add_reflexive_actor = True
    )

    states = torch.randn(2, 5, obs_dim)
    actions = torch.randn(2, 5, action_dim)

    params = wm.get_actor_parameters('reflexive')
    assert len(params) > 0
    assert 'reflexive' in wm.actor_state_encoders
    assert exists(wm.value_state_encoder)

    exp = Experience(
        states = states,
        actions = actions,
        cumulative_rewards = torch.tensor([1.0, 2.0]),
        episode_len = torch.tensor([5, 5])
    )

    loss = tpo_loss(wm, exp, actor_module = 'reflexive')
    assert loss.ndim == 0 and not torch.isnan(loss)
    loss.backward()

def test_detach_base_state_encoder():
    dim = 64
    obs_dim = 8
    action_dim = 2

    wm = Agent(
        state_encoder = nn.Linear(obs_dim, dim),
        action_encoder = nn.Linear(action_dim, dim),
        model = Transformer(dim = dim, depth = 2, causal = True),
        dim_action = action_dim,
        continuous_actions = True,
        transition_action_space = 'raw',
        add_reflexive_actor = True
    )

    states = torch.randn(2, 5, obs_dim)
    actions = torch.randn(2, 5, action_dim)

    exp = Experience(
        states = states,
        actions = actions,
        cumulative_rewards = torch.tensor([1.0, 2.0]),
        episode_len = torch.tensor([5, 5])
    )

    for mode in ('mean', 'sum', 'log_timestep', 'sqrt_timestep'):
        l = tpo_loss(wm, exp, actor_module = 'reflexive', log_score_average_mode = mode)
        assert torch.isfinite(l)

    loss = tpo_loss(wm, exp, actor_module = 'reflexive', detach_base_state_encoder = True)
    loss.backward()

    for p in wm.state_encoder.parameters():
        assert not exists(p.grad)

    for p in wm.actors['reflexive'].parameters():
        assert exists(p.grad)

    for p in wm.actor_state_encoders['reflexive'].parameters():
        assert exists(p.grad)

def test_tpo_batch_size_warning():
    dim = 64
    obs_dim = 8
    action_dim = 2

    wm = Agent(
        state_encoder = nn.Linear(obs_dim, dim),
        action_encoder = nn.Linear(action_dim, dim),
        model = Transformer(dim = dim, depth = 2, causal = True),
        dim_action = action_dim,
        continuous_actions = True,
        transition_action_space = 'raw',
        add_reflexive_actor = True
    )

    states = torch.randn(4, 5, obs_dim)
    actions = torch.randn(4, 5, action_dim)

    exp = Experience(
        states = states,
        actions = actions,
        cumulative_rewards = torch.tensor([1.0, 2.0, 3.0, 4.0]),
        episode_len = torch.tensor([5, 5, 5, 5])
    )

    loss = tpo_loss(wm, exp, actor_module = 'reflexive')
    assert loss.ndim == 0

def test_ppo_value_state_encoder_gradients():
    dim = 64
    obs_dim = 8
    action_dim = 2

    wm = Agent(
        state_encoder = nn.Linear(obs_dim, dim),
        action_encoder = nn.Linear(action_dim, dim),
        model = Transformer(dim = dim, depth = 2, causal = True),
        dim_action = action_dim,
        continuous_actions = True,
        transition_action_space = 'raw',
        add_reflexive_actor = True
    )

    states = torch.randn(2, 5, obs_dim)
    actions = torch.randn(2, 5, action_dim)

    exp = Experience(
        states = states,
        actions = actions,
        returns = torch.randn(2, 5),
        episode_len = torch.tensor([5, 5])
    )

    loss = ppo_loss(wm, exp, actor_module = 'reflexive', detach_base_state_encoder = True)
    loss.backward()

    for p in wm.state_encoder.parameters():
        assert not exists(p.grad)

    for p in wm.actor_state_encoders['reflexive'].parameters():
        assert exists(p.grad)

    for p in wm.value_state_encoder.parameters():
        assert exists(p.grad)

def test_residual_state_encoder_sum():
    dim = 64
    obs_dim = 8
    action_dim = 2

    wm_res = Agent(
        state_encoder = nn.Linear(obs_dim, dim),
        action_encoder = nn.Linear(action_dim, dim),
        model = Transformer(dim = dim, depth = 2, causal = True),
        dim_action = action_dim,
        continuous_actions = True,
        transition_action_space = 'raw',
        add_reflexive_actor = True,
        actor_residual_state_encoder = True
    )

    states = torch.randn(2, 5, obs_dim)

    base_tokens, _, _ = wm_res.encode_states(wm_res.state_encoder, states)
    actor_res_tokens, _, _ = wm_res.encode_states(wm_res.actor_state_encoders['reflexive'], states)
    val_res_tokens, _, _ = wm_res.encode_states(wm_res.value_state_encoder, states)

    actor_state_tokens, _ = wm_res.get_actor_state_tokens('reflexive', states)
    val_state_tokens, _ = wm_res.get_value_state_tokens(states)

    assert_close(actor_state_tokens, base_tokens + actor_res_tokens)
    assert_close(val_state_tokens, base_tokens + val_res_tokens)

    wm_sep = Agent(
        state_encoder = nn.Linear(obs_dim, dim),
        action_encoder = nn.Linear(action_dim, dim),
        model = Transformer(dim = dim, depth = 2, causal = True),
        dim_action = action_dim,
        continuous_actions = True,
        transition_action_space = 'raw',
        add_reflexive_actor = True,
        actor_residual_state_encoder = False
    )

    actor_sep_tokens, _, _ = wm_sep.encode_states(wm_sep.actor_state_encoders['reflexive'], states)
    val_sep_tokens, _, _ = wm_sep.encode_states(wm_sep.value_state_encoder, states)

    actor_state_tokens, _ = wm_sep.get_actor_state_tokens('reflexive', states)
    val_state_tokens, _ = wm_sep.get_value_state_tokens(states)

    assert_close(actor_state_tokens, actor_sep_tokens)
    assert_close(val_state_tokens, val_sep_tokens)

def test_multiple_world_models():
    dim = 256
    model = Transformer(dim = dim, depth = 2, causal = True)

    agent = Agent(
        model = model,
        num_world_models = 3,
        state_encoder = nn.Linear(32, dim),
        action_encoder = nn.Linear(8, dim),
        dim_action = 8,
        transition_action_space = 'raw',
        continuous_actions = True
    )

    assert len(agent.world_models) == 3

    states = torch.randn(2, 5, 32)
    actions = torch.randn(2, 4, 8)

    loss0, _ = agent(states, actions, world_model_index = 0)
    loss1, _ = agent(states, actions, world_model_index = 1)
    loss2, _ = agent(states, actions, world_model_index = 2)

    assert exists(loss0) and exists(loss1) and exists(loss2)

    planned0 = agent.plan(states, actions, goal_state = torch.randn(2, 32), horizon = 1, world_model_index = 0)
    planned2 = agent.plan(states, actions, goal_state = torch.randn(2, 32), horizon = 1, world_model_index = 2)

    assert planned0.shape == (2, 1, 8)
    assert planned2.shape == (2, 1, 8)

    wrapper = AgentRolloutWrapper(agent = agent, world_model_index = 1)
    assert wrapper.world_model_index == 1

    with pytest.raises(AssertionError):
        agent.get_world_model(world_model_index = 5)

    with pytest.raises(AssertionError):
        AgentRolloutWrapper(agent = agent, world_model_index = 5)

@param('temporal_compression', (1, 2, 4))
def test_world_model_temporal_compression(temporal_compression):
    dim = 64
    dim_action = 8
    model = Transformer(dim = dim, depth = 2, causal = True)

    agent = Agent(
        state_encoder = nn.Linear(32, dim),
        action_encoder = nn.Linear(dim_action, dim),
        model = model,
        dim_action = dim_action,
        continuous_actions = True,
        transition_action_space = 'raw',
        temporal_compression = temporal_compression
    )

    seq_len = temporal_compression * 3 + 1
    action_len = seq_len - 1

    states = torch.randn(2, seq_len, 32)
    actions = torch.randn(2, action_len, dim_action)

    # test forward with temporal compression
    loss, losses = agent(states, actions)
    assert loss.ndim == 0
    loss.backward()

    # test plan with temporal compression
    horizon = temporal_compression * 2
    planned_actions = agent.plan(
        states,
        actions,
        goal_state = states[:, -1],
        horizon = horizon,
        pop_size = 4,
        generations = 2
    )

    assert planned_actions.shape == (2, horizon, dim_action)

@param('temporal_compression', (2, 4))
def test_compressed_lane_bounds_keep_final_chunk(temporal_compression):
    num_chunks = 4
    seq_len = num_chunks * temporal_compression + 1
    compressed_len, stop_offset = compressed_lane_bounds(seq_len, temporal_compression)
    assert (compressed_len, stop_offset) == (num_chunks, num_chunks * temporal_compression)

@param('is_vectorized_env', (False, True))
@param('diversity_skill_loss_weight', (0., 0.1))
def test_interact_with_env_with_skills(is_vectorized_env, diversity_skill_loss_weight):
    dim = 32
    num_skills = 4
    dim_skill = 16

    class ConvEncoder(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding = 1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(16, dim)
            )

        def forward(self, x):
            if x.ndim > 4:
                *batch_shape, c, h, w = x.shape
                x = x.reshape(-1, c, h, w)
                out = self.net(x)
                return out.reshape(*batch_shape, -1)

            return self.net(x)

    model = Transformer(
        dim = dim,
        depth = 2,
        heads = 4,
        dim_head = 8
    )

    agent = Agent(
        model = model,
        state_encoder = [ConvEncoder(dim), nn.Linear(4, dim)],
        action_encoder = nn.Linear(8, dim),
        dim_action = 8,
        transition_action_space = 'raw',
        continuous_actions = True,
        use_diayn = True,
        num_skills = num_skills,
        dim_skill = dim_skill,
        add_reflexive_actor = True
    )

    class MultimodalMockEnv:
        def __init__(self, is_vector = False):
            self.step_count = 0
            self.is_vector = is_vector
            if is_vector:
                self.is_vector_env = True
                self.num_envs = 2

        def reset(self):
            self.step_count = 0
            if self.is_vector:
                return [torch.randn(2, 3, 32, 32), torch.randn(2, 4)], {}
            return [torch.randn(3, 32, 32), torch.randn(4)], {}

        def step(self, action):
            self.step_count += 1
            done = self.step_count >= 10
            if self.is_vector:
                img = torch.randn(2, 3, 32, 32)
                prop = torch.randn(2, 4)
                reward = torch.ones(2)
                terminated = torch.full((2,), done)
                truncated = torch.full((2,), False)
            else:
                img = torch.randn(3, 32, 32)
                prop = torch.randn(4)
                reward = 1.0
                terminated = done
                truncated = False
            return [img, prop], reward, terminated, truncated, {}

    env = MultimodalMockEnv(is_vector = is_vectorized_env)
    opt = torch.optim.Adam(agent.skill_discriminator.parameters(), lr = 1e-3)

    # interact with env across skills

    experiences = []

    for active_skill in range(num_skills):
        exp = agent.interact_with_environment(
            env,
            actor_module = 'reflexive',
            max_steps = 10,
            skill_id = active_skill,
            diversity_skill_loss_weight = diversity_skill_loss_weight,
            return_cpu = False
        )

        assert exists(exp.skill_ids)
        assert (exp.skill_ids == active_skill).all()
        if diversity_skill_loss_weight == 0.:
            assert torch.allclose(exp.rewards, torch.full_like(exp.rewards, 1.0))

        experiences.append(exp)

    # learn on experience

    batch_states = torch.cat([e.state_latents for e in experiences], dim = 0)
    batch_actions = torch.cat([e.actions for e in experiences], dim = 0)
    batch_skill_ids = torch.cat([e.skill_ids for e in experiences], dim = 0)

    loss = agent.skill_discriminator.compute_loss(
        batch_states,
        batch_skill_ids,
        actions = batch_actions
    )

    loss.backward()
    opt.step()
    opt.zero_grad()

    # plan with a skill
    img_states = torch.randn(2, 4, 3, 32, 32)
    prop_states = torch.randn(2, 4, 4)
    actions = torch.randn(2, 3, 8)
    img_goal = torch.randn(2, 3, 32, 32)
    prop_goal = torch.randn(2, 4)

    planned_actions = agent.plan(
        [img_states, prop_states],
        actions,
        goal_state = [img_goal, prop_goal],
        horizon = 4,
        pop_size = 8,
        generations = 2,
        skill_id = 2,
        seed_with_actor = 'reflexive'
    )
    assert planned_actions.shape == (2, 4, 8)

    assert planned_actions.shape == (2, 4, 8)

    # custom goal function evaluating DIAYN intrinsic rewards via skill_discriminator
    def custom_fitness_fn(pred_state_latents, skill_discriminator):
        pop_skill_ids = torch.full((2, 8, 4), fill_value = 2, device = pred_state_latents.device)
        return skill_discriminator.compute_intrinsic_reward(pred_state_latents, pop_skill_ids).sum(dim = -1)

    planned_actions_custom = agent.plan(
        [img_states, prop_states],
        actions,
        fitness_fn = custom_fitness_fn,
        horizon = 4,
        pop_size = 8,
        generations = 2,
        skill_id = 2
    )

    assert planned_actions_custom.shape == (2, 4, 8)

    # edge case test: batched per-instance tensor skill_id in vector environment rollout
    class MultimodalBatchMockEnv:
        def __init__(self):
            self.step_count = 0
            self.num_envs = 2

        def reset(self):
            self.step_count = 0
            return [torch.randn(2, 3, 32, 32), torch.randn(2, 4)], {}

        def step(self, action):
            self.step_count += 1
            done = self.step_count >= 5
            img = torch.randn(2, 3, 32, 32)
            prop = torch.randn(2, 4)
            return [img, prop], torch.tensor([1.0, 1.0]), torch.tensor([done, done]), torch.tensor([False, False]), {}

    batch_env = MultimodalBatchMockEnv()
    tensor_skills = torch.tensor([0, 3])

    exp_batched = agent.interact_with_environment(
        batch_env,
        actor_module = 'reflexive',
        max_steps = 5,
        skill_id = tensor_skills,
        return_cpu = False
    )

    assert exists(exp_batched.skill_ids)
    assert exp_batched.skill_ids.shape == (2, 5)
    assert (exp_batched.skill_ids[0] == 0).all()
    assert (exp_batched.skill_ids[1] == 3).all()

    # multimodal observation environment rollout with DIAYN skills
    class ConvEncoder(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, 4, stride = 2, padding = 1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(16 * 16 * 16, dim)
            )

        def forward(self, x):
            if x.ndim > 4:
                shape = x.shape
                return self.net(x.reshape(-1, *shape[-3:])).reshape(*shape[:-3], -1)
            return self.net(x)

    class MultimodalMockEnv:
        def __init__(self):
            self.step_count = 0
            self.is_vector = True

        def reset(self):
            self.step_count = 0
            return (torch.randn(1, 3, 32, 32), torch.randn(1, 4)), {}

        def step(self, action):
            self.step_count += 1
            done = self.step_count >= 5
            return (torch.randn(1, 3, 32, 32), torch.randn(1, 4)), torch.tensor([1.0]), torch.tensor([done]), torch.tensor([False]), {}

    multimodal_agent = Agent(
        model = model,
        state_encoder = [ConvEncoder(dim), nn.Linear(4, dim)],
        action_encoder = nn.Linear(8, dim),
        dim_action = 8,
        transition_action_space = 'raw',
        continuous_actions = True,
        use_diayn = True,
        num_skills = num_skills,
        dim_skill = dim_skill,
        add_reflexive_actor = True
    )

    mm_env = MultimodalMockEnv()
    exp_mm = multimodal_agent.interact_with_environment(
        mm_env,
        actor_module = 'reflexive',
        max_steps = 5,
        skill_id = 2,
        return_cpu = False
    )

    assert exists(exp_mm.skill_ids)
    assert (exp_mm.skill_ids == 2).all()

    mm_loss = multimodal_agent.skill_discriminator.compute_loss(
        exp_mm.state_latents,
        exp_mm.skill_ids,
        actions = exp_mm.actions
    )
    assert not torch.isnan(mm_loss)

def test_transformer_prepend_embeds():
    model = Transformer(
        dim = 32,
        depth = 2,
        causal = True
    )

    tokens = torch.randn(2, 10, 32)
    prepend = torch.randn(2, 4, 32)

    # test without excision
    out = model(tokens, prepend_embeds = prepend, excise_prepend_embeds = False)
    assert out.shape == (2, 14, 32)

    # test with excision
    out_excised = model(tokens, prepend_embeds = prepend, excise_prepend_embeds = True)
    assert out_excised.shape == (2, 10, 32)

    # test return_hiddens with excision
    out_hiddens, hiddens = model(tokens, prepend_embeds = prepend, excise_prepend_embeds = True, return_hiddens = True)
    assert out_hiddens.shape == (2, 10, 32)
    for h in hiddens:
        assert h.shape == (2, 10, 32)

def test_risk_conditioning_e2e_rollout_and_experience():
    from x_jepa.rl import ppo_loss

    class DummyVectorEnv:
        def __init__(self, batch = 2, state_dim = 3):
            self.batch = batch
            self.state_dim = state_dim
            self.is_vector_env = True
            self.num_envs = batch

        def reset(self):
            return torch.randn(self.batch, self.state_dim), {}

        def step(self, action):
            state = torch.randn(self.batch, self.state_dim)
            reward = torch.ones(self.batch)
            terminated = torch.zeros(self.batch, dtype = torch.bool)
            truncated = torch.zeros(self.batch, dtype = torch.bool)
            return state, reward, terminated, truncated, {}

    dim = 32
    state_encoder = nn.Linear(3, dim)
    action_encoder = nn.Linear(2, dim)
    model = Transformer(dim = dim, depth = 2, dim_head = 16, heads = 2)

    agent = Agent(
        state_encoder = state_encoder,
        action_encoder = action_encoder,
        model = model,
        dim_action = 2,
        add_reflexive_actor = True,
        transition_risk_factor = 0.0,
        probabilistic_state_transition = True
    )

    env_optimistic = DummyVectorEnv()
    env_pessimistic = DummyVectorEnv()

    exp_optimistic = agent.interact_with_environment(
        env_optimistic,
        max_steps = 4,
        actor_module = 'reflexive',
        transition_risk_factor = 1.0,
        risk_entropy_bonus_weight = 1.0
    )

    exp_pessimistic = agent.interact_with_environment(
        env_pessimistic,
        max_steps = 4,
        actor_module = 'reflexive',
        transition_risk_factor = -2.0,
        risk_entropy_bonus_weight = 1.0
    )

    assert exists(exp_optimistic.base_reward)
    assert exists(exp_optimistic.transition_risk_reward)

    loss_opt, _ = ppo_loss(agent, exp_optimistic, actor_module = 'reflexive', return_breakdown = True)
    loss_pess, _ = ppo_loss(agent, exp_pessimistic, actor_module = 'reflexive', return_breakdown = True)

    (loss_opt + loss_pess).backward()

def test_risk_embedder_takes_transition_and_value_risk_separately():
    from x_jepa.value_networks import (
        TransitionRiskEmbedder, ValueRiskEmbedder, RiskEmbedder, EnsembleValue
    )

    dim = 32
    batch, seq_len = 2, 4

    # separate bin ranges - transition risk is log-odds-ish (can be negative), value
    # risk is an uncertainty std (>= 0), so the two channels scale differently

    transition_embedder = TransitionRiskEmbedder(dim)
    value_embedder = ValueRiskEmbedder(dim)

    assert transition_embedder.bin_centers.min() < 0.
    assert value_embedder.bin_centers.min() >= 0.

    risk_embedder = RiskEmbedder(dim)

    transition_risk = torch.tensor([1.0, -1.0])
    value_risk = torch.tensor([0.3, 1.5])

    transition_embed = risk_embedder.embed_transition_risk(transition_risk)
    value_embed = risk_embedder.embed_value_risk(value_risk)

    assert transition_embed.shape == (2, dim)
    assert value_embed.shape == (2, dim)

    combined = risk_embedder(transition_risk, value_risk)

    # both risks are taken into account, each through its own channel - the combined
    # embedding is exactly the sum of the separate transition and value embeddings

    assert_close(combined, transition_embed + value_embed, atol = 1e-6, rtol = 1e-6)

    # each risk alone still conditions the embedding

    assert_close(risk_embedder(transition_risk, None), transition_embed, atol = 1e-6, rtol = 1e-6)
    assert_close(risk_embedder(None, value_risk), value_embed, atol = 1e-6, rtol = 1e-6)

    # changing only the transition risk changes the embedding

    different_transition = risk_embedder(transition_risk * 0.5, value_risk)
    assert not torch.allclose(combined, different_transition, atol = 1e-6)

    # changing only the value risk changes the embedding

    different_value = risk_embedder(transition_risk, value_risk * 0.5)
    assert not torch.allclose(combined, different_value, atol = 1e-6)

    # different values for both risks produce distinct conditioning

    embeds = [
        risk_embedder(transition_risk, value_risk),
        risk_embedder(transition_risk, value_risk * 0.5),
        risk_embedder(transition_risk * 0.5, value_risk),
        risk_embedder(transition_risk * 0.5, value_risk * 0.5)
    ]

    for i in range(len(embeds)):
        for j in range(i + 1, len(embeds)):
            assert not torch.allclose(embeds[i], embeds[j], atol = 1e-6), f'embedding pair ({i}, {j}) must differ'

    # gradients flow into both embedders independently

    combined.sum().backward()
    assert exists(risk_embedder.transition_risk_embedder.net[0].weight.grad)
    assert exists(risk_embedder.value_risk_embedder.net[0].weight.grad)
    assert risk_embedder.transition_risk_embedder.net[0].weight.grad.abs().sum() > 0.
    assert risk_embedder.value_risk_embedder.net[0].weight.grad.abs().sum() > 0.

    # a value network built with the combined embedder conditions on both risks, keeps
    # separate EMA copies, and responds to different values of either risk

    tokens = torch.randn(batch, seq_len, dim)
    latents = torch.randn(batch, seq_len, dim)

    value_net = EnsembleValue(
        dim = dim,
        dim_state_latent = dim,
        num_heads = 3,
        transition_risk_factor = 0.0,
        value_risk_factor = 0.0,
        risk_embedder = risk_embedder
    )

    assert value_net.transition_risk_embedder is risk_embedder.transition_risk_embedder
    assert value_net.value_risk_embedder is risk_embedder.value_risk_embedder

    value_base = value_net(tokens, latents, transition_risk = 1.0, value_risk = 0.2)
    value_value_risk_swapped = value_net(tokens, latents, transition_risk = 1.0, value_risk = 1.4)
    value_transition_risk_swapped = value_net(tokens, latents, transition_risk = 0.5, value_risk = 0.2)

    assert value_base.shape == (batch, seq_len, 3)

    # different value risk, same transition risk
    assert not torch.allclose(value_base, value_value_risk_swapped)

    # same value risk, different transition risk
    assert not torch.allclose(value_base, value_transition_risk_swapped)

    # runtime defaults fall back to the configured factors

    assert_close(value_net(tokens, latents), value_net(tokens, latents))

    value_net.update()
    assert exists(value_net.ema_value_risk_embedder)

    value_ema = value_net.forward_ema(tokens, latents, transition_risk = 1.0, value_risk = 0.2)
    assert value_ema.shape == (batch, seq_len, 3)

def test_risk_conditioning_e2e_both_risks_separate_values():
    from x_jepa.rl import ppo_loss
    from x_jepa.value_networks import TransitionRiskEmbedder, ValueRiskEmbedder

    class DummyVectorEnv:
        def __init__(self, batch = 2, state_dim = 3):
            self.batch = batch
            self.state_dim = state_dim
            self.is_vector_env = True
            self.num_envs = batch

        def reset(self):
            return torch.randn(self.batch, self.state_dim), {}

        def step(self, action):
            state = torch.randn(self.batch, self.state_dim)
            reward = torch.ones(self.batch)
            terminated = torch.zeros(self.batch, dtype = torch.bool)
            truncated = torch.zeros(self.batch, dtype = torch.bool)
            return state, reward, terminated, truncated, {}

    dim = 32
    state_encoder = nn.Linear(3, dim)
    action_encoder = nn.Linear(2, dim)
    model = Transformer(dim = dim, depth = 2, dim_head = 16, heads = 2)

    agent = Agent(
        state_encoder = state_encoder,
        action_encoder = action_encoder,
        model = model,
        dim_action = 2,
        add_reflexive_actor = True,
        transition_risk_factor = 0.0,
        value_risk_factor = 0.0,
        probabilistic_state_transition = True
    )

    # both embedders are constructed and are separate modules

    actor = agent.actors['reflexive']
    assert isinstance(actor.transition_risk_embedder, TransitionRiskEmbedder)
    assert isinstance(actor.value_risk_embedder, ValueRiskEmbedder)
    assert actor.transition_risk_embedder is not actor.value_risk_embedder

    value_net = agent.value_network
    assert isinstance(value_net.transition_risk_embedder, TransitionRiskEmbedder)
    assert isinstance(value_net.value_risk_embedder, ValueRiskEmbedder)
    assert value_net.transition_risk_embedder is not value_net.value_risk_embedder

    # the reflexive actor responds to each risk separately - different values for either
    # risk change the action predictions

    state_latents = torch.randn(2, 4, dim)

    pred_base = actor.get_action_preds(state_latents, transition_risk = 1.0, value_risk = 0.2)
    pred_value_risk_swapped = actor.get_action_preds(state_latents, transition_risk = 1.0, value_risk = 1.5)
    pred_transition_risk_swapped = actor.get_action_preds(state_latents, transition_risk = -2.0, value_risk = 0.2)

    assert pred_base.shape == (2, 4, 4) # beta mean + concentration
    assert not torch.allclose(pred_base, pred_value_risk_swapped)
    assert not torch.allclose(pred_base, pred_transition_risk_swapped)

    # the value network responds to each risk separately

    tokens = torch.randn(2, 4, dim)
    latents = torch.randn(2, 4, dim)

    value_base = value_net(tokens, latents, transition_risk = 1.0, value_risk = 0.2)
    value_value_risk_swapped = value_net(tokens, latents, transition_risk = 1.0, value_risk = 1.5)
    value_transition_risk_swapped = value_net(tokens, latents, transition_risk = -2.0, value_risk = 0.2)

    assert not torch.allclose(value_base, value_value_risk_swapped)
    assert not torch.allclose(value_base, value_transition_risk_swapped)

    # rollout with different values for BOTH risks at once - both are recorded and both
    # affect the ppo loss

    exp_pair_a = agent.interact_with_environment(
        DummyVectorEnv(),
        max_steps = 4,
        actor_module = 'reflexive',
        transition_risk_factor = 1.0,
        value_risk_factor = 0.2,
        risk_entropy_bonus_weight = 1.0
    )

    exp_pair_b = agent.interact_with_environment(
        DummyVectorEnv(),
        max_steps = 4,
        actor_module = 'reflexive',
        transition_risk_factor = -2.0,
        value_risk_factor = 1.5,
        risk_entropy_bonus_weight = 1.0
    )

    assert exp_pair_a.transition_risk_factor == 1.0
    assert exp_pair_a.value_risk_factor == 0.2
    assert exp_pair_b.transition_risk_factor == -2.0
    assert exp_pair_b.value_risk_factor == 1.5
    assert exists(exp_pair_a.transition_risk_reward)
    assert exists(exp_pair_b.transition_risk_reward)

    loss_pair_a, _ = ppo_loss(agent, exp_pair_a, actor_module = 'reflexive', return_breakdown = True)
    loss_pair_b, _ = ppo_loss(agent, exp_pair_b, actor_module = 'reflexive', return_breakdown = True)

    # value risk flows through ppo's value-network conditioning - overriding the factor
    # with a different value changes the loss

    loss_pair_a_override = ppo_loss(agent, exp_pair_a, actor_module = 'reflexive', value_risk_factor = 1.5)

    (loss_pair_a + loss_pair_b + loss_pair_a_override).backward()

    assert not torch.allclose(loss_pair_a, loss_pair_b)
    assert not torch.allclose(loss_pair_a, loss_pair_a_override)

def test_plan_with_seed_action():
    dim = 32
    dim_action = 4
    model = Transformer(dim = dim, depth = 2, causal = True)

    agent = Agent(
        state_encoder = nn.Linear(16, dim),
        action_encoder = nn.Linear(dim_action, dim),
        model = model,
        dim_action = dim_action,
        continuous_actions = True,
        transition_action_space = 'raw'
    )

    batch, seq_len = 2, 4
    states = torch.randn(batch, seq_len, 16)
    actions = torch.randn(batch, seq_len - 1, dim_action)
    goal_state = torch.randn(batch, 16)
    horizon = 4

    seed_act = torch.randn(batch, horizon, dim_action)

    planned = agent.plan(
        states,
        actions,
        goal_state = goal_state,
        horizon = horizon,
        seed_action = seed_act,
        include_unperturbed_seed = True,
        pop_size = 8,
        generations = 2
    )

    assert planned.shape == (batch, horizon, dim_action)

    planned_no_unperturbed = agent.plan(
        states,
        actions,
        goal_state = goal_state,
        horizon = horizon,
        seed_action = seed_act,
        include_unperturbed_seed = False,
        pop_size = 8,
        generations = 2
    )

    assert planned_no_unperturbed.shape == (batch, horizon, dim_action)

def test_hierarchical_temporal_refinement_plan():
    dim = 32
    dim_action = 4
    model = Transformer(dim = dim, depth = 2, causal = True)

    agent = Agent(
        state_encoder = nn.Linear(16, dim),
        action_encoder = nn.Linear(dim_action, dim),
        model = model,
        dim_action = dim_action,
        continuous_actions = True,
        transition_action_space = 'raw',
        temporal_compression = (1, 2, 4)
    )

    batch, seq_len = 2, 5
    states = torch.randn(batch, seq_len, 16)
    actions = torch.randn(batch, seq_len - 1, dim_action)
    goal_state = torch.randn(batch, 16)
    horizon = 8  # divisible by highest compression 4

    # 4x

    actions_4x = agent.plan(
        states,
        actions,
        world_model_index = 2,
        discount_factor = 0.999,
        goal_state = goal_state,
        horizon = horizon,
        pop_size = 16,
        generations = 2
    )
    assert actions_4x.shape == (batch, horizon, dim_action)

    # 2x

    actions_2x = agent.plan(
        states,
        actions,
        world_model_index = 1,
        discount_factor = 0.99,
        seed_action = actions_4x,
        goal_state = goal_state,
        horizon = horizon,
        pop_size = 16,
        generations = 2
    )
    assert actions_2x.shape == (batch, horizon, dim_action)

    # 1x

    actions_1x = agent.plan(
        states,
        actions,
        world_model_index = 0,
        discount_factor = 0.95,
        seed_action = actions_2x,
        goal_state = goal_state,
        horizon = horizon,
        pop_size = 16,
        generations = 2
    )
    assert actions_1x.shape == (batch, horizon, dim_action)

# coin-flip network: a frequent state's bonus should collapse, a novel state's stay high
# (Lobel et al. 2023)

def test_coin_flip_network_novelty():
    from x_jepa.intrinsic import CoinFlipNetwork

    torch.manual_seed(42)

    state_dim, dim = 10, 64
    cfn = CoinFlipNetwork(
        net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, dim)),
        dim = dim
    )
    optimizer = torch.optim.Adam(cfn.parameters(), lr = 1e-3, weight_decay = 1e-4)

    state_a = torch.randn(1, state_dim)
    state_b = torch.randn(1, state_dim)

    # warm up the prior running stats
    cfn.train()
    for _ in range(100):
        cfn(torch.randn(16, state_dim))

    # overfit state A only
    cfn.train()
    for _ in range(200):
        optimizer.zero_grad()
        loss = cfn.compute_loss(state_a, coin_flips = cfn.sample_coin_flips(1))
        loss.backward()
        optimizer.step()

    cfn.eval()
    bonus_a = cfn.compute_bonus(state_a).item()
    bonus_b = cfn.compute_bonus(state_b).item()

    assert bonus_a < 0.2, f'bonus for frequent state should drop, got {bonus_a:.4f}'
    assert bonus_b > 0.5, f'bonus for novel state should stay high, got {bonus_b:.4f}'

# probabilistic value networks

def test_value_networks_shapes_and_uncertainty():
    from x_jepa.value_networks import (
        Value, EnsembleValue, DistributionalValue, MeanVarianceValue,
        create_value_network, TransitionRiskEmbedder
    )

    dim, dim_state_latent = 16, 16
    batch, seq_len = 2, 8

    tokens = torch.randn(batch, seq_len, dim)
    latents = torch.randn(batch, seq_len, dim_state_latent)

    value = Value(dim = dim, dim_state_latent = dim_state_latent)
    out = value(tokens, latents)
    assert out.shape == (batch, seq_len)

    ensemble = EnsembleValue(dim = dim, dim_state_latent = dim_state_latent, num_heads = 5)
    out = ensemble(tokens, latents)
    assert out.shape == (batch, seq_len, 5)

    mean, std = ensemble.mean_and_uncertainty(out)
    assert mean.shape == (batch, seq_len)
    assert std.shape == (batch, seq_len)
    assert ensemble.uncertainty(out).shape == (batch, seq_len)

    # ema bootstrap returns per head
    ema_out = ensemble.forward_ema(tokens, latents)
    assert ema_out.shape == (batch, seq_len, 5)

    distributional = DistributionalValue(dim = dim, dim_state_latent = dim_state_latent, num_quantiles = 16)
    out = distributional(tokens, latents)
    assert out.shape == (batch, seq_len, 16)
    assert distributional.value_at_risk(out, alpha = 0.2).shape == (batch, seq_len)

    mean_variance = MeanVarianceValue(dim = dim, dim_state_latent = dim_state_latent)
    out = mean_variance(tokens, latents)
    assert out.shape == (batch, seq_len, 2)

    mean, std = mean_variance.mean_and_uncertainty(out)
    assert mean.shape == (batch, seq_len)
    assert std.shape == (batch, seq_len)
    assert (std > 0).all()

    # factory dispatch
    for vtype, expected in [
        ('value', Value),
        ('ensemble', EnsembleValue),
        ('distributional', DistributionalValue),
        ('mean_variance', MeanVarianceValue)
    ]:
        assert isinstance(create_value_network(vtype, dim = dim, dim_state_latent = dim_state_latent), expected)

def test_value_network_losses():
    from x_jepa.value_networks import EnsembleValue, DistributionalValue, MeanVarianceValue

    dim, dim_state_latent = 16, 16
    batch, seq_len = 2, 8

    tokens = torch.randn(batch, seq_len, dim)
    latents = torch.randn(batch, seq_len, dim_state_latent)
    returns = torch.randn(batch, seq_len)

    ensemble = EnsembleValue(dim = dim, dim_state_latent = dim_state_latent, num_heads = 5)
    pred = ensemble(tokens, latents)
    loss = ensemble.compute_loss(pred, returns)
    assert loss.ndim == 0
    assert loss.item() > 0.

    # per-head returns targets
    per_head_returns = returns.unsqueeze(-1).expand(pred.shape)
    loss_per_head = ensemble.compute_loss(pred, per_head_returns)
    assert loss_per_head.ndim == 0

    # bootstrap masks produce a valid loss in training mode
    ensemble.train()
    pred_masked = ensemble(tokens, latents)
    loss_masked = ensemble.compute_loss(pred_masked, returns)
    assert loss_masked.ndim == 0
    assert not torch.isnan(loss_masked)
    ensemble.eval()

    distributional = DistributionalValue(dim = dim, dim_state_latent = dim_state_latent, num_quantiles = 16)
    pred = distributional(tokens, latents)
    loss = distributional.compute_loss(pred, returns)
    assert loss.ndim == 0
    assert loss.item() > 0.

    mean_variance = MeanVarianceValue(dim = dim, dim_state_latent = dim_state_latent)
    pred = mean_variance(tokens, latents)
    loss = mean_variance.compute_loss(pred, returns)
    assert loss.ndim == 0
    assert not torch.isnan(loss)

    # mean-variance nll prefers a low variance when returns are easy to predict
    torch.manual_seed(0)
    easy_returns = torch.zeros(batch, seq_len)
    optimizer = torch.optim.Adam(mean_variance.parameters(), lr = 1e-2)
    for _ in range(200):
        optimizer.zero_grad()
        pred = mean_variance(tokens, latents)
        loss = mean_variance.compute_loss(pred, easy_returns)
        loss.backward()
        optimizer.step()

    pred = mean_variance(tokens, latents)
    _, std = mean_variance.mean_and_uncertainty(pred)
    assert std.mean().item() < 1.0, f'predicted std should shrink when returns are deterministic, got {std.mean().item():.4f}'

def test_calc_returns_per_statistic_bootstrap():
    from x_jepa.x_jepa import calc_returns

    batch, seq_len, num_stats = 2, 5, 4

    rewards = torch.randn(batch, seq_len)
    is_done = torch.zeros(batch, seq_len, dtype = torch.bool)
    next_values = torch.randn(batch, num_stats)
    discount = 0.9

    returns = calc_returns(rewards, is_done, next_values, discount)
    assert returns.shape == (batch, seq_len, num_stats)

    # scalar path unchanged
    returns_scalar = calc_returns(rewards, is_done, torch.randn(batch), discount)
    assert returns_scalar.shape == (batch, seq_len)

    # per-statistic returns must be the discounted-sum with the per-stat bootstrap
    expected_last = rewards[:, -1:] + discount * next_values
    assert torch.allclose(returns[:, -1], expected_last, atol = 1e-5)

    # with mask, padded entries are zeroed
    mask = torch.zeros(batch, seq_len, dtype = torch.bool)
    mask[:, :2] = True
    returns_masked = calc_returns(rewards, is_done, next_values, discount, mask = mask)
    assert returns_masked.shape == (batch, seq_len, num_stats)
    assert (returns_masked[:, 3:] == 0).all()

def test_flow_value_network():
    from x_jepa.value_networks import FlowValue

    torch.manual_seed(0)

    dim, dim_state_latent = 16, 16
    batch, seq_len = 2, 8

    flow_value = FlowValue(dim = dim, dim_state_latent = dim_state_latent, num_steps = 4, num_samples = 8)

    tokens = torch.randn(batch, seq_len, dim)
    latents = torch.randn(batch, seq_len, dim_state_latent)
    returns = torch.randn(batch, seq_len)

    pred = flow_value(tokens, latents)
    assert pred.shape == (batch, seq_len, 8)

    mean, std = flow_value.mean_and_uncertainty(pred)
    assert mean.shape == (batch, seq_len)
    assert std.shape == (batch, seq_len)

    loss = flow_value.compute_loss(pred, returns, cond = (tokens, latents))
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0.

    # after overfitting the flow on a fixed (state, return) pair, sampling that state
    # must produce return samples centered on the target
    torch.manual_seed(1)
    state_tokens = torch.randn(2, 4, dim)
    state_latents = torch.randn(2, 4, dim_state_latent)
    target = torch.full((2, 4), 2.5)

    optimizer = torch.optim.Adam(flow_value.parameters(), lr = 1e-2)
    for _ in range(150):
        optimizer.zero_grad()
        loss = flow_value.compute_loss(None, target, cond = (state_tokens, state_latents))
        loss.backward()
        optimizer.step()

    flow_value.eval()
    with torch.no_grad():
        samples = flow_value(state_tokens, state_latents)

    mean = samples.mean(dim = -1)
    assert (mean - target).abs().mean().item() < 0.6, f'flow mean should track the target return, got {(mean - target).abs().mean().item():.3f}'
