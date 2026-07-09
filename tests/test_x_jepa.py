import pytest
import torch
param = pytest.mark.parametrize

def test_x_jepa():
    from x_jepa.x_jepa import WorldModel

    state = torch.randn(10)

    model = WorldModel()

    pred = model(state)
