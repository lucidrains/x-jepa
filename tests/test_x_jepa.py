import pytest
param = pytest.mark.parametrize

import torch
from torch import nn
from x_jepa.x_jepa import WorldModel, Transformer

from einops import reduce

@param('use_goal', (False, True))
def test_world_model(
    use_goal
):
    model = Transformer(
        dim = 512,
        depth = 4,
        causal = True
    )

    world_model = WorldModel(
        state_encoder = nn.Linear(128, 512),
        action_encoder = nn.Linear(20, 512),
        action_decoder = nn.Linear(32, 20),
        dim_action_latent = 32,
        model = model
    )

    states = torch.randn(2, 10, 128)
    actions = torch.randn(2, 9, 20)

    loss, _ = world_model(states, actions)

    assert loss.ndim == 0
    loss.backward()

    # optimizer code

    world_model.update() # maybe update ema

    # planning

    if use_goal:
        goal_state = torch.randn(2, 128)
        plan_kwargs = dict(goal_state = goal_state)
    else:
        fitness_fn = lambda pred_state_latents: reduce(pred_state_latents, 'b p ... -> b p', 'sum')
        plan_kwargs = dict(fitness_fn = fitness_fn)

    planned_actions = world_model.plan(states[:, :2], actions[:, :1], horizon = 5, **plan_kwargs)

    assert planned_actions.shape == (2, 5, 20)

@param('continuous_actions', (True, False))
@param('action_len', (9, 10))
def test_behavior_cloning(
    continuous_actions,
    action_len
):
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
        action_decoder = nn.Linear(32, dim_action),
        dim_action_latent = 32,
        model = model,
        bc_model = bc_model,
        dim_action = dim_action,
        continuous_actions = continuous_actions,
        bc_loss_weight = 1.
    )

    states = torch.randn(2, 10, 128)

    if continuous_actions:
        actions = torch.rand(2, action_len, dim_action)
    else:
        actions = torch.randint(0, dim_action, (2, action_len))

    loss, _ = world_model(states, actions)

    assert loss.ndim == 0
    loss.backward()
