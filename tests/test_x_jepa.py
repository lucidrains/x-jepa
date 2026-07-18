import pytest
param = pytest.mark.parametrize

import torch
from torch import nn, tensor, cat
from torch.testing import assert_close

from einops import reduce, rearrange
from einops.layers.torch import Rearrange

from x_mlps_pytorch import MLP

from x_jepa.x_jepa import WorldModel, Transformer, Actor, exists, WorldModelRolloutWrapper, TTTMetaLearningLoss
from x_jepa.min_gru import minGRUBlocks
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
            dist = torch.nn.functional.mse_loss(pred_next_encoded_states, encoded_goal.expand_as(pred_next_encoded_states), reduction = 'none')
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

    planned_actions_mppi = world_model.plan(
        states[:, :2],
        actions[:, :1],
        horizon = 5,
        cem_temperature = 1.,
        search_space = search_space,
        fitness_fn = lambda pred_state_latents: reduce(pred_state_latents, 'b p ... -> b p', 'sum')
    )

    assert planned_actions_mppi.shape == (2, 5, 20)

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
    transition_action_is_raw = transition_action_space == 'raw'

    if transition_action_is_raw and not continuous_actions:
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
    dim_action_latent = 32

    world_model = WorldModel(
        state_encoder = nn.Linear(128, 512),
        action_encoder = nn.Linear(dim_action, 512) if continuous_actions else nn.Embedding(dim_action, 512),
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

    world_model = WorldModel(
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

    world_model = WorldModel(
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

def test_world_model_with_intrinsics():
    from copy import deepcopy
    from x_jepa.intrinsic import CoinFlipNetwork

    dim = 256
    model = Transformer(
        dim = dim,
        depth = 2,
        causal = True
    )

    cfn_net = MLP(dim, 64, dim)
    coin_flip_network = CoinFlipNetwork(net = cfn_net, dim = dim)

    world_model = WorldModel(
        model = model,
        dim_action = 4,
        state_encoder = nn.Linear(128, dim),
        action_encoder = nn.Linear(4, dim),
        action_decoder = nn.Linear(dim, 4),
        transition_action_space = 'local',
        continuous_actions = True,
        intrinsics = [coin_flip_network],
        intrinsic_loss_weight = 1.0,
        intrinsic_frac_gradient = 0.1
    )

    world_model.eval()

    states = torch.randn(2, 5, 128)
    actions = torch.randn(2, 4, 4)

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

    assert planned_actions.shape == (2, 3, 4)

def test_reflexive_actor_and_planning():
    dim = 128
    dim_action = 4
    model = Transformer(
        dim = dim,
        depth = 2,
        causal = True
    )

    world_model = WorldModel(
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

    world_model = WorldModel(
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

    world_model = WorldModel(
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

@pytest.mark.parametrize('use_perception_film', [True, False])
@pytest.mark.parametrize('episodic_mem_len', [0, 4])
def test_world_model_ttt(use_perception_film, episodic_mem_len):
    dim = 256

    # transformer

    model = Transformer(
        dim = dim,
        depth = 2,
        causal = True
    )

    # world model

    wm = WorldModel(
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

    wrapper = WorldModelRolloutWrapper(
        world_model = wm,
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

    states_step1 = torch.randn(batch_size, 1, 32)
    actions_step1 = torch.randn(batch_size, 1, 8)

    out1, memories = wrapper(states_step1, actions_step1, return_memories = True, return_loss = True)

    # ensure ttt params initialized

    check_module_name = 'perception_film' if use_perception_film else 'model.layers.0.2.1'
    ttt_wrapper = wrapper.ttt_trainer.ttt_wrappers[check_module_name]
    assert exists(ttt_wrapper.batch_params)

    # step 2

    states_step2 = torch.randn(batch_size, 1, 32)
    actions_step2 = torch.randn(batch_size, 1, 8)

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

def test_plan_with_gradient_descent():
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
