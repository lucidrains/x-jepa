import pytest
import torch
from x_jepa.x_jepa import WorldModel, Transformer

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
    actions = torch.randn(2, 10, 64)

    loss = world_model(states, actions)

    assert loss.ndim == 0
    loss.backward()

    # optimizer code

    world_model.update() # maybe update ema
