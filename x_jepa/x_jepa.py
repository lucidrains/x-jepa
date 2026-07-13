from __future__ import annotations
from typing import Callable, Literal

import inspect
from collections import namedtuple
from functools import partial

import torch
import torch.nn.functional as F
from torch.distributions import Beta
from torch import nn, cat, stack, tensor, is_tensor, Tensor
from torch.nn import Module, ModuleList, Linear, RMSNorm, Sequential

from einops import einsum, rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from ema_pytorch import EMA

from x_mlps_pytorch import MLP

from torch_einops_utils import (
    pad_right_at_dim_to,
    temp_eval,
    batched_index_select,
    tree_map_tensor,
    pack_with_inverse
)

from x_jepa.regularizers import sigreg_loss, SigReg, uniform_wasserstein_loss

# constants

LinearNoBias = partial(Linear, bias = False)

Losses = namedtuple('Losses', [
    'next_state_latent_pred',
    'action_recon',
    'next_encoded_state_pred',
    'bc',
    'value',
    'reg_next_state',
    'reg_next_encoded',
    'reg_action',
    'action_wasserstein'
])

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def xnor(x, y):
    return x == y

def identity(t):
    return t

def is_empty(t):
    return len(t) == 0

# helper modules

class FracGradient(Module):
    def __init__(
        self,
        frac = 0.
    ):
        super().__init__()
        assert 0. <= frac <= 1.
        self.frac = frac

    def forward(self, x):
        return x.detach().lerp(x, self.frac)

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

        self.to_gates = LinearNoBias(dim, heads)

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

        gates = self.to_gates(tokens)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = self.merge_heads(out)
        return self.to_out(out)

class AttentionResidual(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.scale = dim ** -0.5

        self.to_query = LinearNoBias(dim, dim)
        self.norm_keys = RMSNorm(dim)

    def forward(
        self,
        hiddens
    ):
        if isinstance(hiddens, list):
            hiddens = stack(hiddens)

        hiddens, unpack = pack_with_inverse(hiddens, 'l * d')

        last_layer_output = hiddens[-1]

        # cross attention

        queries = self.to_query(last_layer_output)
        keys = self.norm_keys(hiddens)
        values = hiddens

        sim = einsum(queries, keys, 'm d, l m d -> m l') * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum(attn, values, 'm l, l m d -> m d')

        return unpack(out, '* d')

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
            attn_res = AttentionResidual(dim = dim)
            ff = SwiGLUFeedForward(dim = dim, expand_factor = ff_expand_factor)

            layers.append(ModuleList([attn_res, attn, ff]))

        self.layers = layers

    def forward(
        self,
        tokens,
        prev_hiddens = None,
        return_hiddens = False
    ):
        layer_hiddens = list(default(prev_hiddens, []))
        layer_hiddens.append(tokens)

        for attn_res, attn, ff in self.layers:
            tokens = attn(tokens) + tokens
            tokens = ff(tokens) + tokens

            layer_hiddens.append(tokens)

            tokens = attn_res(layer_hiddens)

        if not return_hiddens:
            return tokens

        return tokens, layer_hiddens

# classes

class WorldModel(Module):
    def __init__(
        self,
        *,
        state_encoder: Module,
        action_encoder: Module,
        model: Module,
        action_decoder: Module | None = None,
        state_transition: Module | None = None,
        transition_action_space: Literal['raw', 'encoded', 'latent'] = 'raw',
        dim_action_latent = None,
        dim_state_latent = None,
        ema_beta = 0.95,
        bc_model: Module | None = None,
        pass_world_model_hiddens_to_actor = True,
        dim_action = None,
        continuous_actions = True,
        action_eps = 1e-5,
        action_recon_loss_weight = 1.,
        next_encoded_state_pred_loss_weight = 1.,
        bc_loss_weight = 1.,
        value_loss_weight = 1.,
        frac_gradients = 0.,
        reg_next_state_weight = 0.,
        reg_next_encoded_weight = 0.,
        reg_action_weight = 0.,
        action_latent_wasserstein_loss_weight = 0.,
        reg: Module | None = None,
        reg_loss_kwargs: dict | None = None
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

        # the learnt state transition that is popularly used for learning the so called world model, but may not be the only prediction needed

        # different types of action spaces for the transition

        if transition_action_space == 'raw': # raw actions from -1. to 1., only for continuous actions
            assert continuous_actions and exists(dim_action)

            dim_transition_action_input = dim_action
            need_learned_action_decoder = False

        elif transition_action_space == 'encoded':  # encoded, but uncontextualized
            dim_transition_action_input = dim_action_latent
            need_learned_action_decoder = True

        elif transition_action_space == 'latent': # the contextualized action
            dim_transition_action_input = dim_action_latent
            need_learned_action_decoder = True

        else:
            raise ValueError('unknown state transition action space')

        self.transition_action_space = transition_action_space
        self.has_action_latents = transition_action_space != 'raw'

        assert not (not self.has_action_latents and action_latent_wasserstein_loss_weight > 0.), 'uniform wasserstein loss on action latents can only be used if there are action latents'

        assert xnor(need_learned_action_decoder, exists(action_decoder)), 'you need to pass in the action_decoder'

        if not exists(state_transition):
            # following Teoh, learn the residual with a 3 layer mlp

            state_transition = MLP(dim_state_latent + dim_transition_action_input, *((dim * 2,) * 2), dim_state_latent)

        self.dim_transition_action_input = dim_transition_action_input
        self.state_transition = state_transition

        self.action_decoder = default(action_decoder, nn.Identity())
        self.need_learned_action_decoder = need_learned_action_decoder

        # main world model

        self.model = model

        # the predictive head for goals

        self.to_next_encoded_state_pred = MLP(dim_state_latent + dim_transition_action_input, *((dim * 2,) * 2), dim)

        # projection to latents

        self.to_state_latent = Sequential(RMSNorm(dim), LinearNoBias(dim, dim_state_latent), nn.Tanh())
        self.to_action_latent = Sequential(RMSNorm(dim), LinearNoBias(dim, dim_action_latent), nn.Tanh())

        # in my experiments, EMA model still outperforms this hyped sigreg regularization.. but i may be seeing an improvement with EMA + sigreg, so lets just allow for all possibilities.

        self.ema_model = EMA(model, beta = ema_beta)

        self.ema_state_encoder = EMA(state_encoder, beta = ema_beta)

        self.ema_state_transition = EMA(state_transition, beta = ema_beta)

        # actor / behavior clone

        self.has_bc = exists(bc_model) and exists(dim_action) and bc_loss_weight > 0.

        self.bc_state_encoder = MLP(dim_state_latent + dim, dim)
        self.bc_action_encoder = MLP(dim_transition_action_input + dim, dim)

        self.bc_model = bc_model
        self.pass_world_model_hiddens_to_actor = pass_world_model_hiddens_to_actor

        self.to_next_action_pred = None

        if self.has_bc:
            dim_action_param = (dim_action * 2) if continuous_actions else dim_action
            self.to_next_action_pred = Sequential(RMSNorm(dim), LinearNoBias(dim, dim_action_param))

        self.continuous_actions = continuous_actions
        self.action_eps = action_eps

        # value head

        self.value_head = MLP(dim_state_latent + dim, *((dim * 2,) * 2), 1)
        self.frac_gradient = FracGradient(frac_gradients)

        # loss related

        self.action_recon_loss_weight = action_recon_loss_weight
        self.next_encoded_state_pred_loss_weight = next_encoded_state_pred_loss_weight
        self.bc_loss_weight = bc_loss_weight
        self.value_loss_weight = value_loss_weight

        # regularizer - defaults to sigreg, but any drop-in (ex. visreg) can be passed in

        self.reg = default(reg, SigReg(**default(reg_loss_kwargs, dict())))

        self.reg_next_state_weight = reg_next_state_weight
        self.reg_next_encoded_weight = reg_next_encoded_weight
        self.reg_action_weight = reg_action_weight

        self.action_latent_wasserstein_loss_weight = action_latent_wasserstein_loss_weight

        self.register_buffer('zero', tensor(0.), persistent = False)

    @property
    def device(self):
        return self.zero.device

    def update(self):
        self.ema_state_encoder.update()
        self.ema_model.update()
        self.ema_state_transition.update()

    @torch.no_grad()
    @temp_eval
    def plan(
        self,
        states,
        actions,
        fitness_fn: Callable[..., Tensor] | None = None,
        goal_state: Tensor | None = None,
        goal_dist_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
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
        dim_action = self.dim_transition_action_input

        assert exists(fitness_fn) or exists(goal_state), 'either fitness_fn or goal_state must be provided'
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

        shape = (batch, 1, horizon, dim_action)

        means = torch.rand(shape, device = device) * 2. - 1.
        stds = torch.rand(shape, device = device)

        # iterate

        for generation in range(generations):
            is_last = generation == generations - 1

            # instantiate population

            # actions could be in raw, encoded, or contextualized latent space

            actions = means + stds * torch.randn((batch, pop_size, horizon, dim_action), device = device)
            actions.clamp_(-1., 1.)

            pred_state_latents = []
            pred_next_encoded_states = []
            pred_values = []

            # step through the learnt world model

            step_state_latents = state_latents

            for step_action in actions.unbind(dim = 2):

                # state transition

                step_residual = self.ema_state_transition((step_state_latents, step_action))

                step_state_latents = step_state_latents + step_residual

                if clamp_state_latent_to_range:
                    step_state_latents.clamp_(-1., 1.)

                pred_state_latents.append(step_state_latents)

                # encoded state for goal

                pred_next_encoded_state = self.to_next_encoded_state_pred((step_state_latents, step_action))

                pred_next_encoded_states.append(pred_next_encoded_state)

                step_pred_value = self.value_head((pred_next_encoded_state, step_state_latents))
                pred_values.append(step_pred_value)

            pred_state_latents = rearrange(pred_state_latents, 'h b p d -> b p h d')

            pred_next_encoded_states = rearrange(pred_next_encoded_states, 'h b p d -> b p h d')
            pred_values = rearrange(pred_values, 'h b p 1 -> b p h')

            # evaluate

            if exists(goal_state):
                encoded_goal = self.ema_state_encoder(goal_state)
                encoded_goal = rearrange(encoded_goal, 'b d -> b 1 1 d')
            else:
                encoded_goal = None

            if exists(fitness_fn):
                # simple introspect and dependency inject into fitness function

                fn_params = inspect.signature(fitness_fn).parameters
                kwargs = dict()

                if 'pred_state_latents' in fn_params:
                    kwargs['pred_state_latents'] = pred_state_latents
                if 'pred_next_encoded_states' in fn_params:
                    kwargs['pred_next_encoded_states'] = pred_next_encoded_states
                if 'pred_values' in fn_params:
                    kwargs['pred_values'] = pred_values
                if 'encoded_goal' in fn_params:
                    kwargs['encoded_goal'] = encoded_goal

                allowed_params = {'pred_state_latents', 'pred_next_encoded_states', 'pred_values', 'encoded_goal'}
                unknown_params = set(fn_params.keys()) - allowed_params
                assert is_empty(unknown_params), f"fitness_fn accepts unknown parameters: {unknown_params}. Allowed parameters are: {', '.join(allowed_params)}"

                fitnesses = fitness_fn(**kwargs)
            else:
                goal_dist_fn = default(goal_dist_fn, partial(F.smooth_l1_loss, reduction = 'none'))
                distance_to_goal = goal_dist_fn(pred_next_encoded_states, encoded_goal)
                distance_to_goal = reduce(distance_to_goal, 'b p h d -> b p', 'sum')
                fitnesses = -distance_to_goal

            # select the fittest

            topk = num_elites if not is_last else 1

            elite_indices = fitnesses.topk(topk, largest = True, dim = -1).indices

            # select the elites

            fittest_actions = batched_index_select(actions, elite_indices, dim = 1)

            if not is_last:
                means = reduce(fittest_actions, 'b p h d -> b 1 h d', 'mean')

                stds = reduce(fittest_actions, 'b p h d -> b 1 h d', partial(torch.std, unbiased = False))
                stds = stds.clamp_min(eps)

        # return the top winner

        winning_actions = fittest_actions[:, 0]

        decoded_actions = self.action_decoder(winning_actions)

        if not self.continuous_actions:
            decoded_actions = decoded_actions.argmax(dim = -1)

        # return

        if not return_action_latent:
            return decoded_actions

        return decoded_actions, winning_actions

    def forward(
        self,
        states,
        actions,
        returns = None,
        behavior_clone = True
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

        embeds, world_model_hiddens = self.model(tokens, return_hiddens = True)

        target_embeds = self.ema_model(tokens)

        # split out the state and action embeds

        target_state_embeds, _ = rearrange(target_embeds, 'b (n sa) d -> sa b n d', sa = 2)

        next_target_state_embeds = target_state_embeds[:, 1:]

        next_target_state_latents = self.to_state_latent(next_target_state_embeds)

        # now we predict the next latent from (s_t, a_t) -> s_t+1

        state_embeds, action_embeds = rearrange(embeds[:, :-2], 'b (n sa) d -> sa b n d', sa = 2)

        state_latents = self.to_state_latent(state_embeds)

        # the action conditioning for the state latents that determine its transition

        if self.transition_action_space == 'raw':
            action_cond = orig_actions

        elif self.transition_action_space == 'encoded':
            action_cond = self.to_action_latent(action_tokens)

        elif self.transition_action_space == 'latent':
            action_cond = self.to_action_latent(action_embeds)

        action_cond = action_cond[:, :state_latents.shape[1]]

        # prediction

        pred_residual = self.state_transition((state_latents, action_cond))

        next_state_pred = state_latents + pred_residual

        loss = F.smooth_l1_loss(next_state_pred, next_target_state_latents.detach())

        # action decoder

        action_recon_loss = self.zero

        if self.need_learned_action_decoder:
            decoded_actions = self.action_decoder(action_cond)

            recon_orig_actions = orig_actions[:, :decoded_actions.shape[1]]

            if self.continuous_actions:
                action_recon_loss = F.mse_loss(
                    recon_orig_actions,
                    decoded_actions
                )
            else:
                action_recon_loss = F.cross_entropy(
                    rearrange(decoded_actions, 'b n c -> b c n'),
                    recon_orig_actions
                )

        # goal prediction head - cannot use the next state, as the encoded goal does not know the past sequence that led to it

        pred_next_encoded_state = self.to_next_encoded_state_pred((state_latents, action_cond))

        ema_encoded_state = self.ema_state_encoder(states)

        next_ema_encoded_state_target = ema_encoded_state[:, 1:]

        next_encoded_state_pred_loss = F.smooth_l1_loss(pred_next_encoded_state, next_ema_encoded_state_target.detach())

        # maybe behavior clone

        bc_loss = self.zero

        if self.has_bc and behavior_clone:
            bc_encoded_states = self.bc_state_encoder(cat((state_tokens[:, :-1], state_latents.detach()), dim = -1))
            bc_encoded_actions = self.bc_action_encoder(cat((action_tokens[:, :action_cond.shape[1]], action_cond.detach()), dim = -1))

            bc_tokens = rearrange([bc_encoded_states, bc_encoded_actions], 'sa b n d -> b (n sa) d')

            detached_hiddens = None

            if self.pass_world_model_hiddens_to_actor:
                seq_len = bc_tokens.shape[1]
                detached_hiddens = tree_map_tensor(lambda t: t[:, :seq_len].detach(), world_model_hiddens)

            bc_embed = self.bc_model(bc_tokens, prev_hiddens = detached_hiddens)

            bc_state_embed, _ = rearrange(bc_embed, 'b (n sa) d -> sa b n d', sa = 2)

            next_action_pred = self.to_next_action_pred(bc_state_embed)

            # log probs

            next_actions = orig_actions[:, :next_action_pred.shape[1]]

            if self.continuous_actions:
                # use unimodal beta

                alpha, beta = rearrange(next_action_pred, 'b n (alpha_beta na) -> alpha_beta b n na', alpha_beta = 2)
                alpha = F.softplus(alpha) + self.action_eps
                beta = F.softplus(beta) + self.action_eps

                distr = Beta(alpha, beta)

                next_actions_zero_one = (next_actions + 1) / 2
                next_actions_zero_one = next_actions_zero_one.clamp(self.action_eps, 1. - self.action_eps)

                log_probs = distr.log_prob(next_actions_zero_one).sum(dim = -1)

                bc_loss = -log_probs.mean()

            else:
                bc_loss = F.cross_entropy(rearrange(next_action_pred, 'b n c -> b c n'), next_actions)

        # value head

        value_loss = self.zero

        pred_values = self.value_head((state_tokens[:, :-1], self.frac_gradient(state_latents)))
        pred_values = rearrange(pred_values, '... 1 -> ...')

        if exists(returns):
            value_loss = F.mse_loss(pred_values, returns)

        # regularizer loss

        reg_next_state_loss = self.zero
        reg_next_encoded_loss = self.zero
        reg_action_loss = self.zero

        if self.reg_next_state_weight > 0.:
            reg_next_state_loss = self.reg(next_state_pred)

        if self.reg_next_encoded_weight > 0.:
            reg_next_encoded_loss = self.reg(pred_next_encoded_state)

        if self.reg_action_weight > 0.:
            reg_action_loss = self.reg(action_cond)

        # action latent uniform loss

        action_wasserstein_loss = self.zero

        if self.action_latent_wasserstein_loss_weight > 0.:
            action_wasserstein_loss = uniform_wasserstein_loss(action_cond)

        # losses

        total_loss = (
            loss +
            action_recon_loss * self.action_recon_loss_weight +
            next_encoded_state_pred_loss * self.next_encoded_state_pred_loss_weight +
            bc_loss * self.bc_loss_weight +
            value_loss * self.value_loss_weight +
            reg_next_state_loss * self.reg_next_state_weight +
            reg_next_encoded_loss * self.reg_next_encoded_weight +
            reg_action_loss * self.reg_action_weight +
            action_wasserstein_loss * self.action_latent_wasserstein_loss_weight
        )

        loss_breakdown = Losses(
            loss,
            action_recon_loss,
            next_encoded_state_pred_loss,
            bc_loss,
            value_loss,
            reg_next_state_loss,
            reg_next_encoded_loss,
            reg_action_loss,
            action_wasserstein_loss
        )

        return total_loss, loss_breakdown
