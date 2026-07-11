import pytest
import torch
from x_jepa.x_jepa import WorldModel, Transformer

from einops import reduce

def test_world_model():
    model = Transformer(
        dim = 512,
        depth = 4,
        causal = True
    )

    world_model = WorldModel(
        state_encoder = torch.nn.Linear(128, 512),
        action_encoder = torch.nn.Linear(64, 512),
        model = model
    )

    states = torch.randn(2, 10, 128)
    actions = torch.randn(2, 9, 64)

    loss = world_model(states, actions)

    assert loss.ndim == 0
    loss.backward()

    # optimizer code

    world_model.update() # maybe update ema

    # planning

    fitness_fn = lambda pred_state_latents: reduce(pred_state_latents, 'b p ... -> b p', 'sum')

    planned_actions = world_model.plan(states[:, :2], actions[:, :1], fitness_fn, horizon = 5)

    assert planned_actions.shape == (2, 5, 512)
