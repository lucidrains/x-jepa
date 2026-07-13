import pytest
param = pytest.mark.parametrize

import torch
from torch import nn
from x_jepa.x_jepa import WorldModel, Transformer

from einops import reduce

@param('plan_type', ('no_goal', 'goal', 'custom_goal'))
@param('transition_action_space', ('raw', 'encoded', 'latent'))
@param('use_sigreg', (False, True))
def test_world_model(
    plan_type,
    transition_action_space,
    use_sigreg
):
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
        sigreg_next_state_weight = float(use_sigreg),
        sigreg_next_encoded_weight = float(use_sigreg),
        sigreg_action_weight = float(use_sigreg)
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
            dist = torch.nn.functional.mse_loss(pred_next_encoded_states, encoded_goal, reduction='none')
            dist = reduce(dist, 'b p h d -> b p', 'sum')
            values = reduce(pred_values, 'b p h -> b p', 'sum')
            return values - dist

        plan_kwargs = dict(goal_state = goal_state, fitness_fn = custom_fitness_fn)

    else:
        fitness_fn = lambda pred_state_latents: reduce(pred_state_latents, 'b p ... -> b p', 'sum')
        plan_kwargs = dict(fitness_fn = fitness_fn)

    planned_actions = world_model.plan(states[:, :2], actions[:, :1], horizon = 5, **plan_kwargs)

    assert planned_actions.shape == (2, 5, 20)

@param('continuous_actions', (True, False))
@param('action_len', (9, 10))
@param('transition_action_space', ('raw', 'encoded', 'latent'))
def test_behavior_cloning(
    continuous_actions,
    action_len,
    transition_action_space
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
