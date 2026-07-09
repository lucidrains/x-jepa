from __future__ import annotations

import torch
from torch import nn
from torch.nn import Module, ModuleList

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class WorldModel(Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, state):
        return state
