import pytest
param = pytest.mark.parametrize

import torch
from torch import nn, tensor
from torch.testing import assert_close

from einops import reduce

from x_jepa.x_jepa import WorldModel, Transformer
from x_jepa.regularizers import SigReg, VISReg, uniform_wasserstein_loss

@param('plan_type', ('no_goal', 'goal', 'custom_goal'))
@param('transition_action_space', ('raw', 'encoded', 'latent'))
@param('use_reg', (False, True))
@param('reg_type', ('sigreg', 'visreg'))
def test_world_model(
    plan_type,
    transition_action_space,
    use_reg,
    reg_type
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
        action_latent_wasserstein_loss_weight = float(use_reg and not transition_action_is_raw)
    )

    states = torch.randn(2, 10, 128)
    actions = torch.randn(2, 9, 20).tanh()
    returns = torch.randn(2, 9)

    loss, loss_breakdown = world_model(states, actions, returns = returns)

    assert len(loss_breakdown) == 8
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

@param('transition_action_space', ('raw', 'encoded', 'latent'))
@param('search_space', ('raw', 'encoded_latent', None))
def test_plan_search_spaces(
    transition_action_space,
    search_space
):
    if transition_action_space == 'raw' and search_space == 'encoded_latent':
        pytest.skip('raw transition action space requires raw search space')

    if transition_action_space == 'latent' and search_space == 'raw':
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

@param('transition_action_space', ('raw', 'encoded', 'latent'))
@param('search_space', ('raw', None))
def test_plan_scores_representation_of_returned_action(
    transition_action_space,
    search_space
):
    # https://github.com/lucidrains/x-jepa/issues/14 - the action representation used to score a candidate must correspond to the action returned for execution

    if transition_action_space == 'latent':
        pytest.skip('latent search candidates cannot yet be projected onto the action encoder image, so the scored representation is not guaranteed to survive decode -> re-encode')

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

    # record the action representation each candidate is scored with

    scored_action_reps = []

    def record_transition_inputs(module, args, output):
        _, step_action_cond = args[0]
        scored_action_reps.append(step_action_cond)

    world_model.ema_state_transition.ema_model.register_forward_hook(record_transition_inputs)

    states = torch.randn(2, 2, 128)
    actions = torch.randn(2, 1, 20).tanh()

    horizon = 3

    planned_actions = world_model.plan(
        states,
        actions,
        horizon = horizon,
        pop_size = 32,
        generations = 2,
        search_space = search_space,
        fitness_fn = lambda pred_state_latents: reduce(pred_state_latents, 'b p ... -> b p', 'sum')
    )

    # the representation the returned action maps back to, computed as during training

    if transition_action_is_raw:
        executed_action_reps = planned_actions
    else:
        executed_action_reps = world_model.to_action_latent(world_model.action_encoder(planned_actions))

    # the returned plan must be among the candidate representations scored in the final generation

    final_generation = torch.stack(scored_action_reps[-horizon:], dim = 2) # (b, p, h, d)

    dists = (final_generation - executed_action_reps.unsqueeze(1)).norm(dim = -1) # (b, p, h)

    closest_candidate_dist = dists.amax(dim = -1).amin(dim = -1) # worst step of the best candidate

    assert (closest_candidate_dist < 1e-4).all()

@param('continuous_actions', (True, False))
@param('action_len', (9, 10))
@param('transition_action_space', ('raw', 'encoded', 'latent'))
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

    bc_model = Transformer(
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
        bc_model = bc_model,
        pass_world_model_hiddens_to_actor = pass_world_model_hiddens_to_actor,
        dim_action = dim_action,
        continuous_actions = continuous_actions,
        bc_loss_weight = 1.
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
