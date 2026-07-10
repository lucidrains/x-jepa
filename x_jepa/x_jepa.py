from __future__ import annotations
from functools import partial

import torch
from torch import nn, stack
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear

from einops import einsum, rearrange
from einops.layers.torch import Rearrange

from ema_pytorch import EMA

from x_mlps_pytorch import MLP

# constants

LinearNoBias = partial(Linear, bias = False)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        causal = True
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads
        self.causal = causal

        self.to_q = LinearNoBias(dim, dim_inner)
        self.to_kv = LinearNoBias(dim, dim_inner * 2)

        self.split_heads = Rearrange('b n (h d) -> b h n d', d = dim_head)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = LinearNoBias(dim_inner, dim)

    def forward(
        self,
        tokens # (b n d)
    ):
        device = tokens.device

        q, k, v = (self.to_q(tokens), *self.to_kv(tokens).chunk(2, dim = -1))
        q, k, v = (self.split_heads(t) for t in (q, k, v))

        q = q * self.scale

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)
        return self.to_out(out)

# feedforward

class SwiGLUGate(Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return F.silu(gates) * x

def SwiGLUFeedForward(dim, expand_factor = 4.):
    dim_inner = int(dim * expand_factor * 2 / 3)
    return nn.Sequential(
        Linear(dim, dim_inner * 2),
        SwiGLUGate(),
        Linear(dim_inner, dim)
    )

# transformer

class Transformer(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        causal = True,
        dim_head = 64,
        heads = 8,
        ff_expand_factor = 4.
    ):
        super().__init__()
        self.dim = dim

        layers = ModuleList([])

        for _ in range(depth):
            attn = Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal)

            ff = SwiGLUFeedForward(dim = dim, expand_factor = ff_expand_factor)

            layers.append(ModuleList([attn, ff]))

        self.layers = layers

        self.norm = nn.RMSNorm(dim)

    def forward(
        self,
        tokens
    ):

        for attn, ff in self.layers:
            tokens = attn(tokens) + tokens
            tokens = ff(tokens) + tokens

        return self.norm(tokens)

# classes

class WorldModel(Module):
    def __init__(
        self,
        *,
        state_encoder: Module,
        action_encoder: Module,
        model: Module,
        state_transition: Module | None = None,
        ema_beta = 0.95
    ):
        super().__init__()

        self.state_encoder = state_encoder
        self.action_encoder = action_encoder

        if not exists(state_transition):
            # following Teoh, learn the residual with a 3 layer mlp

            dim = model.dim

            state_transition = MLP(dim * 2, *((dim * 4,) * 2), dim)

        self.state_transition = state_transition
        self.model = model

        # in my experiments, EMA model still outperforms this hyped sigreg regularization.. but i may be seeing an improvement with EMA + sigreg, so lets just allow for all possibilities.

        self.ema_model = EMA(model, beta = ema_beta)

    def update(self):
        self.ema_model.update()

    def forward(
        self,
        states,
        actions
    ):

        state_tokens = self.state_encoder(states)
        action_tokens = self.action_encoder(actions)

        # now we interleave the states and actions

        tokens = rearrange([state_tokens, action_tokens], 'sa b n d -> b (n sa) d')

        # attention + eventually rnns - yes we need recurrence, i concede that.

        # we will follow Teoh et al's lead and use the post-norm space as the latent

        latents = self.model(tokens)

        target_latents = self.ema_model(tokens)

        # split out the state and action latents

        target_state_latents, _ = rearrange(target_latents, 'b (n sa) d -> sa b n d', sa = 2)

        next_target_state_latents = target_state_latents[:, 1:]

        # now we predict the next latent from (s_t, a_t) -> s_t+1

        state_action_latents = rearrange(latents, 'b (n sa) d -> b n (sa d)', sa = 2)

        next_state_pred = self.state_transition(state_action_latents[:, :-1])

        loss = F.smooth_l1_loss(next_state_pred, next_target_state_latents)

        return loss
