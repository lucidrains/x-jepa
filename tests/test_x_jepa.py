import pytest
param = pytest.mark.parametrize

import torch
from torch import nn, tensor, cat
from torch.testing import assert_close

from einops import reduce, rearrange
from einops.layers.torch import Rearrange

from x_jepa.x_jepa import WorldModel, Transformer
from x_jepa.regularizers import SigReg, VISReg, uniform_wasserstein_loss
from x_jepa.goals import FlowMatching, GoalGenerator

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
        dim = 512,
        depth = 4,
        causal = True
    )

    transition_action_is_raw = transition_action_space == 'raw'

    reg = SigReg() if reg_type == 'sigreg' else VISReg()

    world_model = WorldModel(
        state_encoder = nn.Linear(128, 512),
        action_encoder = nn.Linear(20, 512),
        action_decoder = nn.Linear(32, 20) if not transition_action_is_raw else None,
        transition_action_space = transition_action_space,
        dim_action = 20,
        dim_action_latent = 32,
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

    states = torch.randn(2, 10, 128)
    actions = torch.randn(2, 9, 20).tanh()
    returns = torch.randn(2, 10)

    loss, loss_breakdown = world_model(states, actions, returns = returns)

    assert len(loss_breakdown) == 13
    assert loss.ndim == 0
    loss.backward()

    # optimizer code

    world_model.update() # maybe update ema

    # planning

    if plan_type == 'goal':
        goal_state = torch.randn(2, 128)
        plan_kwargs = dict(goal_state = goal_state)

    elif plan_type == 'custom_goal':
        goal_state = torch.randn(2, 128)

        def custom_fitness_fn(pred_values, pred_next_encoded_states, encoded_goal):
            dist = torch.nn.functional.mse_loss(pred_next_encoded_states, encoded_goal, reduction = 'none')
            dist = reduce(dist, 'b p h d -> b p', 'sum')
            values = reduce(pred_values, 'b p h -> b p', 'sum')
            return values - dist

        plan_kwargs = dict(goal_state = goal_state, fitness_fn = custom_fitness_fn)

    else:
        fitness_fn = lambda pred_state_latents: reduce(pred_state_latents, 'b p ... -> b p', 'sum')
        plan_kwargs = dict(fitness_fn = fitness_fn)

    planned_actions = world_model.plan(states[:, :2], actions[:, :1], horizon = 5, **plan_kwargs)

    assert planned_actions.shape == (2, 5, 20)

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
        dim = 512,
        depth = 4,
        causal = True
    )

    transition_action_is_raw = transition_action_space == 'raw'

    world_model = WorldModel(
        state_encoder = nn.Linear(128, 512),
        action_encoder = nn.Linear(20, 512),
        action_decoder = nn.Linear(32, 20) if not transition_action_is_raw else None,
        transition_action_space = transition_action_space,
        dim_action = 20,
        dim_action_latent = 32,
        model = model,
    )

    states = torch.randn(2, 10, 128)
    actions = torch.randn(2, 9, 20).tanh()

    planned_actions = world_model.plan(
        states[:, :2],
        actions[:, :1],
        horizon = 5,
        search_space = search_space,
        fitness_fn = lambda pred_state_latents: reduce(pred_state_latents, 'b p ... -> b p', 'sum')
    )

    assert planned_actions.shape == (2, 5, 20)

@param('continuous_actions', (True, False))
@param('action_len', (9, 10))
@param('transition_action_space', ('raw', 'local', 'global'))
@param('pass_world_model_hiddens_to_actor', (True, False))
def test_behavior_cloning(
    continuous_actions,
    action_len,
    transition_action_space,
    pass_world_model_hiddens_to_actor
):
    if transition_action_space == 'raw' and not continuous_actions:
        pytest.skip('raw state transition action space requires continuous actions')

    model = Transformer(
        dim = 512,
        depth = 2,
        causal = True
    )

    actor_model = Transformer(
        dim = 512,
        depth = 2,
        causal = True
    )

    dim_action = 20 if continuous_actions else 4

    world_model = WorldModel(
        state_encoder = nn.Linear(128, 512),
        action_encoder = nn.Linear(dim_action, 512) if continuous_actions else nn.Embedding(dim_action, 512),
        action_decoder = nn.Linear(32, dim_action) if transition_action_space != 'raw' else None,
        transition_action_space = transition_action_space,
        dim_action_latent = 32,
        model = model,
        actor_model = actor_model,
        pass_world_model_hiddens_to_actor = pass_world_model_hiddens_to_actor,
        dim_action = dim_action,
        continuous_actions = continuous_actions,
        actor_loss_weight = 1.
    )

    states = torch.randn(2, 10, 128)

    if continuous_actions:
        actions = torch.randn(2, action_len, dim_action).tanh()
    else:
        actions = torch.randint(0, dim_action, (2, action_len))

    loss, _ = world_model(states, actions)

    assert loss.ndim == 0
    loss.backward()

@param('reg_type', ('sigreg', 'visreg'))
def test_reg_loss(reg_type):
    from x_jepa.regularizers import sigreg_loss, visreg_loss

    loss_fn = sigreg_loss if reg_type == 'sigreg' else visreg_loss

    x = torch.randn(256, 64).requires_grad_()

    loss = loss_fn(x)

    assert loss.ndim == 0
    loss.backward()

@pytest.mark.parametrize('samples', (tensor([[0.]]), tensor([[-0.5], [0.5]])))
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
def test_transformer(use_pope):
    model = Transformer(
        dim = 512,
        depth = 2,
        causal = True,
        use_pope = use_pope
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

    world_model = WorldModel(
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

    world_model = WorldModel(
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

    world_model = WorldModel(
        model = model,
        dim_action = 2,
        state_encoder = nn.Linear(4, 128),
        action_encoder = nn.Linear(2, 128),
        action_decoder = nn.Linear(128, 2),
        transition_action_space = 'local',
        continuous_actions = False
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

    assert len(experience.states) > 0
    assert len(experience.actions) == len(experience.states)
    assert len(experience.rewards) == len(experience.states)

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

    world_model = WorldModel(
        state_encoder = state_encoders,
        action_encoder = nn.Linear(64, 256),
        model = model,
        dim_action = 64,
        actor_model = actor_model,
        pass_sensory_hiddens_to_world_model = True,
        pass_sensory_hiddens_to_actor = True
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
