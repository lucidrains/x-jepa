from __future__ import annotations
from typing import Callable
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, cat, stack, tensor, is_tensor, Tensor
from torch.nn import Module, ModuleList, Linear, RMSNorm, Sequential

from einops import einsum, rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from ema_pytorch import EMA

from x_mlps_pytorch import MLP

from torch_einops_utils import (
    pad_right_at_dim_to,
    temp_eval,
    batched_index_select
)

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
        causal = True,
        prenorm = True
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads
        self.causal = causal

        self.norm = RMSNorm(dim) if prenorm else nn.Identity()

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

        tokens = self.norm(tokens)

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

def SwiGLUFeedForward(
    dim,
    expand_factor = 4.,
    prenorm = True
):
    dim_inner = int(dim * expand_factor * 2 / 3)
    return nn.Sequential(
        RMSNorm(dim) if prenorm else nn.Identity(),
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

    def forward(
        self,
        tokens
    ):

        for attn, ff in self.layers:
            tokens = attn(tokens) + tokens
            tokens = ff(tokens) + tokens

        return tokens

# classes

class WorldModel(Module):
    def __init__(
        self,
        *,
        state_encoder: Module,
        action_encoder: Module,
        action_decoder: Module,
        model: Module,
        state_transition: Module | None = None,
        dim_action_latent = None,
        dim_state_latent = None,
        ema_beta = 0.95,
        action_recon_loss_weight = 1.
    ):
        super().__init__()

        # dimensions

        dim = model.dim
        dim_state_latent = default(dim_state_latent, dim)
        dim_action_latent = default(dim_action_latent, dim)

        self.dim_state_latent = dim_state_latent
        self.dim_action_latent = dim_action_latent

        # state and action encoder / decoder

        self.state_encoder = state_encoder

        self.action_encoder = action_encoder
        self.action_decoder = action_decoder

        if not exists(state_transition):
            # following Teoh, learn the residual with a 3 layer mlp

            state_transition = MLP(dim_state_latent + dim_action_latent, *((dim * 2,) * 2), dim)

        self.state_transition = state_transition
        self.model = model

        # projection to latents

        self.to_state_latent = Sequential(RMSNorm(dim), LinearNoBias(dim, dim_state_latent), nn.Tanh())
        self.to_action_latent = Sequential(RMSNorm(dim), LinearNoBias(dim, dim_action_latent), nn.Tanh())

        # in my experiments, EMA model still outperforms this hyped sigreg regularization.. but i may be seeing an improvement with EMA + sigreg, so lets just allow for all possibilities.

        self.ema_model = EMA(model, beta = ema_beta)

        self.ema_state_transition = EMA(state_transition, beta = ema_beta)

        # loss related

        self.action_recon_loss_weight = action_recon_loss_weight

        self.register_buffer('zero', tensor(0.), persistent = False)

    @property
    def device(self):
        self.zero.device

    def update(self):
        self.ema_model.update()
        self.ema_state_transition.update()

    @torch.no_grad()
    @temp_eval
    def plan(
        self,
        states,
        actions,
        fitness_fn: Callable[[Tensor], Tensor],
        horizon = 1,
        pop_size = 1024,
        elite_frac = 0.1,
        generations = 5,
        eps = 1e-5,
        clamp_state_latent_to_range = True,
        return_action_latent = False
    ):
        device = self.device

        batch = states.shape[0]
        state_len, action_len = states.shape[1], actions.shape[1]
        dim_action_latent = self.dim_action_latent

        assert action_len == (state_len - 1)
        assert generations > 0
        assert horizon >= 1

        actions = pad_right_at_dim_to(actions, state_len, dim = 1)

        # get the state and action latents

        state_tokens = self.state_encoder(states)
        action_tokens = self.action_encoder(actions)

        tokens = rearrange([state_tokens, action_tokens], 'sa b n d -> b (n sa) d')

        embeds = self.ema_model(tokens)

        state_embeds = embeds[:, -2] # (b d)

        state_latents = self.to_state_latent(state_embeds)

        state_latents = repeat(state_latents, 'b d -> b p d', p = pop_size)

        # start with naive cross entropy method

        # means and std

        num_elites = int(elite_frac * pop_size)
        assert num_elites >= 1

        dim_latent = state_latents.shape[-1]

        shape = (batch, 1, horizon, dim_action_latent)

        means = torch.rand(shape, device = device) * 2. - 1.
        stds = torch.rand(shape, device = device)

        # iterate

        for generation in range(generations):
            is_last = generation == generations - 1

            # instantiate population

            action_latents = means + stds * torch.randn((batch, pop_size, horizon, dim_action_latent), device = device)
            action_latents.clamp_(-1., 1.)

            pred_state_latents = []

            # step through the learnt world model

            step_state_latents = state_latents

            for step_action_latents in action_latents.unbind(dim = 2):

                step_residual = self.ema_state_transition((step_state_latents, step_action_latents))

                step_state_latents = step_state_latents + step_residual

                if clamp_state_latent_to_range:
                    step_state_latents.clamp_(-1., 1.)

                pred_state_latents.append(step_state_latents)

            pred_state_latents = rearrange(pred_state_latents, 'h b p d -> b p h d')

            # evaluate

            fitnesses = fitness_fn(pred_state_latents) # (b p)

            # select the fittest

            topk = num_elites if not is_last else 1

            elite_indices = fitnesses.topk(topk, largest = True, dim = -1).indices

            # select the elites

            fittest_action_latents = batched_index_select(action_latents, elite_indices, dim = 1)

            if not is_last:
                means = reduce(fittest_action_latents, 'b p h d -> b 1 h d', 'mean')

                stds = reduce(fittest_action_latents, 'b p h d -> b 1 h d', partial(torch.std, unbiased = False))
                stds = stds.clamp_min(eps)

        # return the top winner

        winning_action_latents = fittest_action_latents[:, 0]

        decoded_actions = self.action_decoder(winning_action_latents)

        # return

        if not return_action_latent:
            return decoded_actions

        return decoded_actions, winning_action_latents

    def forward(
        self,
        states,
        actions
    ):
        state_len, action_len = states.shape[1], actions.shape[1]
        assert action_len in {state_len, state_len - 1}

        orig_actions = actions

        actions = pad_right_at_dim_to(actions, state_len, dim = 1)

        # handle last action not being given

        # encode the states and actions, todo: do mixture of transformers, a la VLAs

        state_tokens = self.state_encoder(states)
        action_tokens = self.action_encoder(actions)

        # now we interleave the states and actions

        tokens = rearrange([state_tokens, action_tokens], 'sa b n d -> b (n sa) d')

        # attention + eventually rnns - yes we need recurrence, i concede that.

        # we will follow Teoh et al's lead and use the post-norm space as the latent

        embeds = self.model(tokens)

        target_embeds = self.ema_model(tokens)

        # split out the state and action embeds

        target_state_embeds, _ = rearrange(target_embeds, 'b (n sa) d -> sa b n d', sa = 2)

        next_target_state_embeds = target_state_embeds[:, 1:]

        next_target_state_latents = self.to_state_latent(next_target_state_embeds)

        # now we predict the next latent from (s_t, a_t) -> s_t+1

        state_embeds, action_embeds = rearrange(embeds[:, :-2], 'b (n sa) d -> sa b n d', sa = 2)

        state_latents = self.to_state_latent(state_embeds)

        action_latents = self.to_action_latent(action_embeds)

        # prediction

        pred_residual = self.state_transition((state_latents, action_latents))

        next_state_pred = state_latents + pred_residual

        loss = F.smooth_l1_loss(next_state_pred, next_target_state_latents.detach())

        # action decoder

        decoded_actions = self.action_decoder(action_latents)

        action_recon_loss = F.mse_loss(
            orig_actions,
            decoded_actions[:, :action_len]
        )

        # losses

        total_loss = (
            loss +
            action_recon_loss * self.action_recon_loss_weight
        )

        loss_breakdown = (loss, action_recon_loss)

        return total_loss, loss_breakdown
