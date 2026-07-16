from __future__ import annotations
from typing import Callable, Literal

import inspect
from functools import partial
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch.distributions import Beta
from torch import nn, cat, stack, tensor, is_tensor, Tensor
from torch.nn import Module, ModuleList, Linear, RMSNorm, Sequential
from torch.utils._pytree import tree_map

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

from PoPE_pytorch import PoPE, apply_pope_to_qk

from x_jepa.utils import EnvWrapper, Experience

from x_jepa.regularizers import SigReg, uniform_wasserstein_loss, temporal_straightening_loss
from x_jepa.min_gru import minGRUBlocks

# constants

LinearNoBias = partial(Linear, bias = False)

States = Tensor | list[Tensor] | tuple[Tensor, ...]

Losses = namedtuple('Losses', [
    'next_state_latent_pred',
    'plan_state_pred',
    'action_recon',
    'next_encoded_state_pred',
    'actor',
    'value',
    'reg_next_state',
    'reg_next_encoded',
    'action_wasserstein',
    'goal',
    'temporal_straightening'
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

def first(t):
    return None if is_empty(t) else t[0]

def batch_repeat(t, r):
    return repeat(t, 'b ... -> (b r) ...', r = r)

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

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

class SoftClamp(Module):
    def __init__(self, value = 10.):
        super().__init__()
        self.value = value
        self.has_value = exists(value) and value > 0.

    def forward(self, x):
        if not self.has_value:
            return x
        return (x / self.value).tanh() * self.value

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
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
        tokens, # (b n d)
        pope_pos_emb = None,
        memories = None,
        return_memories = False
    ):
        device = tokens.device

        tokens = self.norm(tokens)

        q, k, v = (self.to_q(tokens), *self.to_kv(tokens).chunk(2, dim = -1))
        q, k, v = (self.split_heads(t) for t in (q, k, v))

        if exists(pope_pos_emb):
            q, k = apply_pope_to_qk(pope_pos_emb, q, k)

        if exists(memories):
            past_k, past_v = memories
            k = cat((past_k, k), dim = -2)
            v = cat((past_v, v), dim = -2)

        next_memories = (k, v) if return_memories else None

        q = q * self.scale

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value(sim))

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        gates = self.to_gates(tokens)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = self.merge_heads(out)
        out = self.to_out(out)

        if not return_memories:
            return out

        return out, next_memories

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
        hiddens,
        mask = None
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

        if exists(mask):
            if isinstance(mask, list):
                mask = stack(mask)

            mask, _ = pack_with_inverse(mask, 'l *')
            mask = rearrange(mask, 'l m -> m l')

            mask = pad_right_at_dim_to(mask, sim.shape[-1], value = True)

            sim = sim.masked_fill(~mask, max_neg_value(sim))

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
        ff_expand_factor = 4.,
        use_pope = False
    ):
        super().__init__()
        self.dim = dim

        self.pope = PoPE(dim_head, heads = heads) if use_pope else None

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
        past_layers = None,
        past_layers_mask = None,
        return_hiddens = False,
        memories = None,
        return_memories = False
    ):
        seq_len = tokens.shape[-2]

        past_seq_len = 0
        layer_memories = None

        if exists(memories):
            past_seq_len, layer_memories = memories

        pope_pos_emb = self.pope(seq_len, offset = past_seq_len) if exists(self.pope) else None

        layer_hiddens = list(default(past_layers, []))
        layer_hiddens.append(tokens)

        next_layer_memories = []

        memories_iter = iter(layer_memories) if exists(layer_memories) else None

        for attn_res, attn, ff in self.layers:
            layer_memory = next(memories_iter, None) if exists(memories_iter) else None

            attn_out, next_memory = attn(
                tokens,
                pope_pos_emb = pope_pos_emb,
                memories = layer_memory,
                return_memories = True
            )

            if return_memories:
                next_layer_memories.append(next_memory)

            tokens = attn_out + tokens
            tokens = ff(tokens) + tokens

            layer_hiddens.append(tokens)

            tokens = attn_res(layer_hiddens, mask = past_layers_mask)

        ret = (tokens, layer_hiddens) if return_hiddens else tokens

        if not return_memories:
            return ret

        next_memories = (past_seq_len + seq_len, next_layer_memories)

        return ret, next_memories

class BetaDistrReadout(Module):
    def __init__(
        self,
        source_range: tuple[float, float] = (0., 1.),
        eps = 1e-5
    ):
        super().__init__()
        self.source_range = source_range
        self.eps = eps

    def forward(
        self,
        x,
        target = None,
        sample = False,
        temperature = 1.,
        return_entropy = False,
        return_raw = False,
        return_distr = False
    ):
        source_min, source_max = self.source_range

        # derive alpha and beta from the raw logits

        alpha, beta = rearrange(x, '... (alpha_beta d) -> alpha_beta ... d', alpha_beta = 2)

        alpha = F.softplus(alpha) + 1. + self.eps
        beta = F.softplus(beta) + 1. + self.eps

        # maybe apply temperature

        if temperature != 1.:
            alpha = (alpha - 1.) / temperature + 1.
            beta = (beta - 1.) / temperature + 1.

        if return_raw:
            return alpha, beta

        distr = Beta(alpha, beta)

        # maybe calculate negative log probs if target is passed in

        if exists(target):
            # scale from source range to (0, 1) bounds for beta distribution
            target = (target - source_min) / (source_max - source_min)

            target = target.clamp(self.eps, 1. - self.eps)
            return -distr.log_prob(target)

        # maybe return the distribution itself

        if return_distr:
            return distr

        # sample or get the mode

        if sample:
            out = distr.rsample()
        else:
            out = (alpha - 1.) / (alpha + beta - 2.).clamp_min(self.eps)

        # scale back to source range

        out = out * (source_max - source_min) + source_min

        # maybe return entropy for naive exploration during planning

        if not return_entropy:
            return out

        return out, distr.entropy()

# classes

class WorldModel(Module):
    def __init__(
        self,
        *,
        state_encoder: Module | ModuleList | list[Module] | tuple[Module, ...],
        action_encoder: Module,
        model: Module,
        action_decoder: Module | None = None,
        state_transition: Module | None = None,
        state_transition_for_planning: Module | None = None,
        transition_action_space: Literal['raw', 'local', 'global'] = 'raw',
        dim_action_latent = None,
        dim_state_latent = None,
        state_latent_clamp_value = 10.,
        ema_beta = 0.95,
        actor_model: Module | None = None,
        pass_world_model_hiddens_to_actor = True,
        dim_action = None,
        continuous_actions = True,
        action_eps = 1e-5,
        action_recon_loss_weight = 1.,
        next_encoded_state_pred_loss_weight = 1.,
        plan_state_pred_loss_weight = 1.,
        actor_loss_weight = 1.,
        value_loss_weight = 1.,
        pass_sensory_hiddens_to_world_model = False,
        pass_sensory_hiddens_to_actor = False,
        frac_gradients = 0.,
        reg_next_state_weight = 0.,
        reg_next_encoded_weight = 0.,
        action_latent_wasserstein_loss_weight = 0.,
        temporal_straightening_loss_weight = 0.,
        learn_goal_generator = False,
        goal_loss_weight = 1.,
        returns_norm_momentum = 0.01,
        probabilistic_state_transition = False,
        probabilistic_plan_state_transition = False,
        state_transition_eps = 1e-5,
        reg: Module | None = None,
        reg_loss_kwargs: dict | None = None,
        state_linear_rnn_depth = 1,
        action_linear_rnn_depth = 1
    ):
        super().__init__()

        # dimensions

        self.dim_action = dim_action

        dim = model.dim
        dim_state_latent = default(dim_state_latent, dim)
        dim_action_latent = default(dim_action_latent, dim)

        self.dim_state_latent = dim_state_latent
        self.dim_action_latent = dim_action_latent
        self.state_latent_clamp_value = state_latent_clamp_value

        if not isinstance(state_encoder, (list, tuple, ModuleList)):
            state_encoder = [state_encoder]

        if isinstance(state_encoder, (list, tuple)):
            state_encoder = ModuleList(state_encoder)

        self.state_encoder = state_encoder

        if pass_sensory_hiddens_to_world_model or pass_sensory_hiddens_to_actor:
            assert len(self.state_encoder) > 1, 'passing sensory layers to attention residual requires more than one modality'

        self.action_encoder = action_encoder

        self.state_linear_rnn = minGRUBlocks(dim = dim, depth = state_linear_rnn_depth)
        self.action_linear_rnn = minGRUBlocks(dim = dim, depth = action_linear_rnn_depth)

        # the learnt state transition that is popularly used for learning the so called world model, but may not be the only prediction needed

        # different types of action spaces for the transition

        if transition_action_space == 'raw': # raw actions from -1. to 1., only for continuous actions
            assert continuous_actions and exists(dim_action)

            dim_transition_action_input = dim_action
            need_learned_action_decoder = False

        elif transition_action_space == 'local':  # encoded, but no context
            dim_transition_action_input = dim_action_latent
            need_learned_action_decoder = True

        elif transition_action_space == 'global': # sees past actions
            dim_transition_action_input = dim_action_latent
            need_learned_action_decoder = True

        else:
            raise ValueError('unknown state transition action space')

        self.transition_action_space = transition_action_space
        self.is_transition_action_space_raw = transition_action_space == 'raw'
        self.has_action_latents = not self.is_transition_action_space_raw

        assert not (not self.has_action_latents and action_latent_wasserstein_loss_weight > 0.), 'uniform wasserstein loss on action latents can only be used if there are action latents'

        assert xnor(need_learned_action_decoder, exists(action_decoder)), 'you need to pass in the action_decoder'

        # probabilistic state transition related

        self.probabilistic_state_transition = probabilistic_state_transition
        self.probabilistic_plan_state_transition = probabilistic_plan_state_transition

        if probabilistic_state_transition or probabilistic_plan_state_transition:
            assert state_latent_clamp_value > 0., 'state_latent_clamp_value must be greater than 0 if either probabilistic state transition is turned on'

        if probabilistic_state_transition:
            self.state_transition_beta_distr = BetaDistrReadout(source_range = (-state_latent_clamp_value, state_latent_clamp_value), eps = state_transition_eps)

        if probabilistic_plan_state_transition:
            self.plan_state_transition_beta_distr = BetaDistrReadout(source_range = (-state_latent_clamp_value, state_latent_clamp_value), eps = state_transition_eps)

        # following Teoh, learn the residual with a 3 layer mlp, for both state transition functions, one for main forward dynamics, the other for planning
        # if probabilistic, it learns the absolute state transition parameterized by a unimodal beta distribution instead of a residual

        if not exists(state_transition):
            dim_out = dim_state_latent * 2 if probabilistic_state_transition else dim_state_latent
            state_transition = MLP(dim_state_latent + dim_transition_action_input, *((dim * 2,) * 2), dim_out)

        if not exists(state_transition_for_planning):
            dim_out = dim_state_latent * 2 if probabilistic_plan_state_transition else dim_state_latent
            state_transition_for_planning = MLP(dim_state_latent + dim_transition_action_input, *((dim * 2,) * 2), dim_out)

        self.dim_transition_action_input = dim_transition_action_input

        self.state_transition = state_transition
        self.state_transition_for_planning = state_transition_for_planning

        # maybe action decoding

        self.action_decoder = default(action_decoder, nn.Identity())
        self.need_learned_action_decoder = need_learned_action_decoder

        # main world model transformer

        self.model = model

        # the predictive head for goals

        self.to_next_encoded_state_pred = MLP(dim_state_latent + dim_transition_action_input, *((dim * 2,) * 2), dim)

        # projection to latents

        self.to_state_latent = Sequential(RMSNorm(dim), LinearNoBias(dim, dim_state_latent), SoftClamp(state_latent_clamp_value))
        self.to_action_latent = Sequential(RMSNorm(dim), LinearNoBias(dim, dim_action_latent), nn.Tanh())

        # in my experiments, EMA model still outperforms this hyped sigreg regularization.. but i may be seeing an improvement with EMA + sigreg, so lets just allow for all possibilities.

        self.ema_model = EMA(model, beta = ema_beta)

        self.ema_state_encoder = ModuleList([EMA(enc, beta = ema_beta) for enc in self.state_encoder])

        self.ema_state_transition = EMA(state_transition, beta = ema_beta)
        self.ema_state_transition_for_planning = EMA(self.state_transition_for_planning, beta = ema_beta)

        # actor / behavior clone

        self.has_actor = exists(actor_model) and exists(dim_action) and actor_loss_weight > 0.

        self.actor_state_encoder = MLP(dim_state_latent + dim, dim)
        self.actor_action_encoder = MLP(dim_transition_action_input + dim, dim)

        if not isinstance(actor_model, (list, tuple, ModuleList)) and exists(actor_model):
            actor_model = [actor_model]

        if isinstance(actor_model, (list, tuple)):
            actor_model = ModuleList(actor_model)

        self.actor_models = actor_model
        self.pass_world_model_hiddens_to_actor = pass_world_model_hiddens_to_actor
        self.pass_sensory_hiddens_to_world_model = pass_sensory_hiddens_to_world_model
        self.pass_sensory_hiddens_to_actor = pass_sensory_hiddens_to_actor

        self.to_next_action_preds = None

        if self.has_actor:
            if self.pass_sensory_hiddens_to_actor:
                self.to_actor_sensory_hiddens = ModuleList([LinearNoBias(dim, dim) for _ in self.state_encoder])

            dim_action_param = (dim_action * 2) if continuous_actions else dim_action
            self.to_next_action_preds = ModuleList([Sequential(RMSNorm(dim), LinearNoBias(dim, dim_action_param)) for _ in self.actor_models])

        self.continuous_actions = continuous_actions
        self.action_eps = action_eps

        if self.continuous_actions:
            self.action_beta_distr = BetaDistrReadout(source_range = (-1., 1.), eps = action_eps)

        # value head

        self.value_head = MLP(dim_state_latent + dim, *((dim * 2,) * 2), 1)
        self.frac_gradient = FracGradient(frac_gradients)

        # loss related

        self.next_encoded_state_pred_loss_weight = next_encoded_state_pred_loss_weight
        self.plan_state_pred_loss_weight = plan_state_pred_loss_weight

        self.action_recon_loss_weight = action_recon_loss_weight
        self.actor_loss_weight = actor_loss_weight
        self.value_loss_weight = value_loss_weight

        self.action_latent_wasserstein_loss_weight = action_latent_wasserstein_loss_weight

        # goal generator

        self.learn_goal_generator = learn_goal_generator
        self.goal_loss_weight = goal_loss_weight

        if learn_goal_generator:
            from x_jepa.goals import GoalGenerator, FlowMatching
            self.goal_generator = GoalGenerator(
                dim = dim_state_latent,
                returns_norm_momentum = returns_norm_momentum # controls how fast the agent ascends the hedonic treadmill
            )
            self.goal_flow_matching = FlowMatching(model = self.goal_generator)

        # regularizer - defaults to sigreg, but any drop-in (ex. visreg) can be passed in

        self.reg = default(reg, SigReg(**default(reg_loss_kwargs, dict())))

        self.reg_next_state_weight = reg_next_state_weight
        self.reg_next_encoded_weight = reg_next_encoded_weight

        self.action_latent_wasserstein_loss_weight = action_latent_wasserstein_loss_weight
        self.temporal_straightening_loss_weight = temporal_straightening_loss_weight

        self.register_buffer('zero', tensor(0.), persistent = False)

    @property
    def device(self):
        return self.zero.device

    def encode_states(
        self,
        encoder: ModuleList,
        states: list[Tensor] | tuple[Tensor, ...]
    ):
        assert len(encoder) == len(states), 'number of states must match number of state encoders'
        encoded = [enc(state) for enc, state in zip(encoder, states)]
        return sum(encoded), encoded

    def update(self):
        [ema.update() for ema in self.ema_state_encoder]
        self.ema_model.update()
        self.ema_state_transition.update()
        self.ema_state_transition_for_planning.update()

    @torch.no_grad()
    @temp_eval
    def plan(
        self,
        states: States,
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
        return_action_latent = False,
        search_space: Literal['raw', 'local_global'] | None = None,
        memories = None,
        return_memories = False
    ):
        if is_tensor(states):
            states = [states]

        batch, device = first(states).shape[0], self.device
        state_len, action_len = first(states).shape[1], actions.shape[1]

        # search space and some validation while it is still incomplete

        can_search_raw_actions = self.continuous_actions and self.transition_action_space != 'global' and exists(self.dim_action)
        search_space = default(search_space, 'raw' if can_search_raw_actions else 'local_global')

        if self.is_transition_action_space_raw:
            assert search_space == 'raw'
        elif self.transition_action_space == 'global':
            assert search_space == 'local_global'
        elif search_space == 'raw':
            assert self.continuous_actions and exists(self.dim_action), 'searching in raw action space requires continuous actions and `dim_action` to be set'

        is_search_space_raw_action = search_space == 'raw'
        dim_action = self.dim_transition_action_input if not is_search_space_raw_action else self.dim_action

        # validation

        assert exists(fitness_fn) or exists(goal_state), 'either fitness_fn or goal_state must be provided'
        assert action_len == (state_len - 1)
        assert generations > 0
        assert horizon >= 1

        actions = pad_right_at_dim_to(actions, state_len, dim = 1)

        # get the state and action latents

        state_tokens, sensory_layer_hiddens = self.encode_states(self.state_encoder, states)
        action_tokens = self.action_encoder(actions)

        # handle memories

        state_rnn_memories = action_rnn_memories = model_memories = None

        if exists(memories):
            state_rnn_memories, action_rnn_memories, model_memories = memories

        # linear rnns for contextualizing states and actions separately

        rnn_state_tokens, next_state_rnn_memories = self.state_linear_rnn(
            state_tokens,
            memories = state_rnn_memories,
            return_memories = True
        )

        rnn_action_tokens, next_action_rnn_memories = self.action_linear_rnn(
            action_tokens,
            memories = action_rnn_memories,
            return_memories = True
        )

        tokens = rearrange([rnn_state_tokens, rnn_action_tokens], 'sa b n d -> b (n sa) d')

        (embeds, _), next_model_memories = self.ema_model(
            tokens,
            return_hiddens = True,
            memories = model_memories,
            return_memories = True
        )

        state_embeds = embeds[:, -2] # (b d)

        state_latents = self.to_state_latent(state_embeds)

        state_latents = repeat(state_latents, 'b d -> b p d', p = pop_size)

        # start with naive cross entropy method

        # means and std

        num_elites = max(int(elite_frac * pop_size), 1)

        shape = (batch, 1, horizon, dim_action)

        means = torch.rand(shape, device = device) * 2. - 1.
        stds = torch.rand(shape, device = device)

        # precompute past rnn memories for rnn action latents, repeating to population size

        past_rnn_memories = tree_map_tensor(partial(batch_repeat, r = pop_size), next_action_rnn_memories)

        # iterate

        for generation in range(generations):
            is_last = generation == generations - 1

            # instantiate population

            # actions could be in raw, encoded, or contextualized latent space

            actions = means + stds * torch.randn((batch, pop_size, horizon, dim_action), device = device)
            actions.clamp_(-1., 1.)

            # the action condition into the step

            # the state transition for planning could take raw actions or action latents depending on the transition action space

            actions_cond = actions

            if is_search_space_raw_action and not self.is_transition_action_space_raw:
                actions_cond, unpack = pack_with_inverse(actions_cond, '* h d')

                actions_cond = self.action_encoder(actions_cond)

                # if using actions with global context of past actions, pass through linear rnn with previous memories

                if self.transition_action_space == 'global':
                    actions_cond = self.action_linear_rnn(actions_cond, memories = past_rnn_memories)

                actions_cond = self.to_action_latent(actions_cond)

                actions_cond = unpack(actions_cond)

            # accumulating predictions across horizon

            pred_state_latents = []
            pred_next_encoded_states = []
            pred_values = []
            pred_state_entropies = [] if self.probabilistic_plan_state_transition else None

            # step through the learnt world model

            step_state_latents = state_latents

            for step_action_cond in actions_cond.unbind(dim = 2):

                # encoded state for goal

                pred_next_encoded_state = self.to_next_encoded_state_pred((step_state_latents, step_action_cond))

                # state transition

                if self.probabilistic_plan_state_transition:
                    pred_next_state_logits = self.ema_state_transition_for_planning((step_state_latents, step_action_cond))
                    step_state_latents, step_entropy = self.plan_state_transition_beta_distr(
                        pred_next_state_logits,
                        sample = True,
                        return_entropy = True
                    )
                    pred_state_entropies.append(step_entropy)
                else:
                    pred_next_state_latent_residual = self.ema_state_transition_for_planning((step_state_latents, step_action_cond))
                    step_state_latents = step_state_latents + pred_next_state_latent_residual

                    if clamp_state_latent_to_range and exists(self.state_latent_clamp_value) and self.state_latent_clamp_value > 0.:
                        step_state_latents.clamp_(-self.state_latent_clamp_value, self.state_latent_clamp_value)

                pred_state_latents.append(step_state_latents)

                pred_next_encoded_states.append(pred_next_encoded_state)

                step_pred_value = self.value_head((pred_next_encoded_state, step_state_latents))
                pred_values.append(step_pred_value)

            pred_state_latents = rearrange(pred_state_latents, 'h b p d -> b p h d')
            if exists(pred_state_entropies):
                pred_state_entropies = rearrange(pred_state_entropies, 'h b p d -> b p h d')

            pred_next_encoded_states = rearrange(pred_next_encoded_states, 'h b p d -> b p h d')
            pred_values = rearrange(pred_values, 'h b p 1 -> b p h')

            # evaluate

            encoded_goal = None

            if exists(goal_state):
                if is_tensor(goal_state):
                    goal_state = [goal_state]

                encoded_goal, ema_encoded_goal = self.encode_states(self.ema_state_encoder, goal_state)
                encoded_goal = rearrange(encoded_goal, 'b d -> b 1 1 d')

            if exists(fitness_fn):
                # simple introspect and dependency inject into fitness function

                fn_params = inspect.signature(fitness_fn).parameters

                kwargs = dict(
                    pred_state_latents = pred_state_latents,
                    pred_next_encoded_states = pred_next_encoded_states,
                    pred_values = pred_values,
                    encoded_goal = encoded_goal,
                    pred_state_entropies = pred_state_entropies
                )

                allowed_params = set(kwargs.keys())
                unknown_params = set(fn_params.keys()) - allowed_params
                assert is_empty(unknown_params), f"fitness_fn accepts unknown parameters: {unknown_params}. Allowed parameters are: {', '.join(allowed_params)}"

                fitness_fn_kwargs = {k: v for k, v in kwargs.items() if k in fn_params}
                fitnesses = fitness_fn(**fitness_fn_kwargs)
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

        decoded_actions = winning_actions

        if not is_search_space_raw_action:
            decoded_actions = self.action_decoder(winning_actions)

        if not self.continuous_actions:
            decoded_actions = decoded_actions.argmax(dim = -1)

        # return

        out = decoded_actions

        if return_action_latent:
            out = (decoded_actions, winning_actions)

        if not return_memories:
            return out

        next_memories = (next_state_rnn_memories, next_action_rnn_memories, next_model_memories)
        return out, next_memories

    @torch.no_grad()
    @temp_eval
    def interact_with_environment(
        self,
        env,
        max_steps = 1000,
        fitness_fn = None,
        goal_state = None,
        return_cpu = True,
        **plan_kwargs
    ):
        env = EnvWrapper(env, return_cpu = return_cpu)

        states, actions, rewards, terminateds, truncateds, infos = [], [], [], [], [], []

        state, info = env.reset()

        batch, device = state.shape[0], state.device

        memories = None

        is_done = torch.zeros(batch, dtype = torch.bool, device = device)
        cumulative_rewards = torch.zeros(batch, device = device)
        episode_len = torch.zeros(batch, dtype = torch.long, device = device)

        for step in range(max_steps):
            is_last_step = step == (max_steps - 1)

            states.append(state)
            infos.append(info)

            empty_actions = torch.empty(batch, 0, self.dim_action, device = self.device)
            state = state.to(self.device)

            planned_actions, memories = self.plan(
                states = state.unsqueeze(1),
                actions = empty_actions,
                fitness_fn = fitness_fn,
                goal_state = goal_state,
                memories = memories,
                return_memories = True,
                **plan_kwargs
            )

            # pick the first action for now

            action = planned_actions[:, 0]
            actions.append(action.cpu() if return_cpu else action)

            next_state, reward, terminated, truncated, info = env.step(action)

            if is_done.any():
                reward = reward.masked_fill(is_done, 0.)

            cumulative_rewards.add_(reward)
            episode_len.add_((~is_done).long())

            if is_last_step:
                truncated = truncated | ~is_done

            is_done |= terminated | truncated

            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)

            state = next_state

            if is_done.all():
                break

        experience = (states, actions, rewards, terminateds, truncateds, infos, episode_len, cumulative_rewards)

        return Experience(*experience)

    def forward(
        self,
        states: States,
        actions,
        returns = None,
        behavior_clone = True,
        return_loss = True,
        memories = None,
        return_memories = False
    ):
        device = self.device

        if is_tensor(states):
            states = [states]

        (batch, state_len), action_len = first(states).shape[:2], actions.shape[1]
        assert action_len in {state_len, state_len - 1}

        orig_actions = actions

        actions = pad_right_at_dim_to(actions, state_len, dim = 1)

        # handle memories

        state_rnn_memories = action_rnn_memories = model_memories = None

        if exists(memories):
            state_rnn_memories, action_rnn_memories, model_memories = memories

        # handle last action not being given

        # encode the states and actions, todo: do mixture of transformers, a la VLAs

        state_tokens, sensory_layer_hiddens = self.encode_states(self.state_encoder, states)
        action_tokens = self.action_encoder(actions)

        # linear rnns for contextualizing states and actions separately, and for potential path integration and idm

        rnn_state_tokens, next_state_rnn_memories = self.state_linear_rnn(
            state_tokens,
            memories = state_rnn_memories,
            return_memories = True
        )

        rnn_action_tokens, next_action_rnn_memories = self.action_linear_rnn(
            action_tokens,
            memories = action_rnn_memories,
            return_memories = True
        )

        # now we interleave the states and actions

        tokens = rearrange([rnn_state_tokens, rnn_action_tokens], 'sa b n d -> b (n sa) d')

        # pass sensory hiddens as attention residual layer hiddens

        wm_past_layers = None
        wm_past_layers_mask = None

        if self.pass_sensory_hiddens_to_world_model:
            wm_past_layers = [rearrange([h, torch.zeros_like(h)], 'sa b n d -> b (n sa) d') for h in sensory_layer_hiddens]
            sensory_mask = tensor([True, False], device = device)
            sensory_mask = repeat(sensory_mask, 'sa -> b (n sa)', b = batch, n = state_len)
            wm_past_layers_mask = [sensory_mask] * len(sensory_layer_hiddens)

        # attention + eventually rnns - yes we need recurrence, i concede that.

        # we will follow Teoh et al's lead and use the post-norm space as the latent

        (embeds, world_model_hiddens), next_model_memories = self.model(
            tokens,
            past_layers = wm_past_layers,
            past_layers_mask = wm_past_layers_mask,
            return_hiddens = True,
            memories = model_memories,
            return_memories = True
        )

        target_embeds = self.ema_model(tokens)

        # split out the state and action embeds

        target_state_embeds, _ = rearrange(target_embeds, 'b (n sa) d -> sa b n d', sa = 2)

        next_target_state_embeds = target_state_embeds[:, 1:]

        next_target_state_latents = self.to_state_latent(next_target_state_embeds)

        # now we predict the next latent from (s_t, a_t) -> s_t+1

        state_embeds_full, action_embeds_full = rearrange(embeds, 'b (n sa) d -> sa b n d', sa = 2)
        state_latents_full = self.to_state_latent(state_embeds_full)

        # memories from world model

        next_memories = (next_state_rnn_memories, next_action_rnn_memories, next_model_memories)

        # early return with embed for testing

        if not return_loss:
            out = dict(embeds = embeds)
            return (out, next_memories) if return_memories else out

        # now ready to do losses, remove the last token for predictive coding

        state_embeds, state_latents, action_embeds = (t[:, :-1] for t in (state_embeds_full, state_latents_full, action_embeds_full))

        # the action conditioning for the state latents that determine its transition

        if self.transition_action_space == 'raw':
            action_cond = orig_actions

        elif self.transition_action_space == 'local':
            action_cond = self.to_action_latent(action_tokens)

        elif self.transition_action_space == 'global':
            action_cond = self.to_action_latent(rnn_action_tokens)

        action_cond = action_cond[:, :state_latents.shape[1]]

        # state transition prediction

        if self.probabilistic_state_transition:
            # probabilistic

            next_state_pred_logits = self.state_transition((state_latents, action_cond))

            neg_log_probs = self.state_transition_beta_distr(next_state_pred_logits, target = next_target_state_latents.detach())
            next_state_latent_pred_loss = neg_log_probs.sum(dim = -1).mean()

            # for next_state_pred to be accessible for reg loss if needed

            next_state_pred = self.state_transition_beta_distr(next_state_pred_logits)
        else:
            # deterministic

            pred_residual = self.state_transition((state_latents, action_cond))

            next_state_pred = state_latents + pred_residual

            next_state_latent_pred_loss = F.smooth_l1_loss(next_state_pred, next_target_state_latents.detach())

        # prediction for planning transition, where actions can be in raw, local encoded, or global encoded

        if self.probabilistic_plan_state_transition:
            # probabilistic

            next_state_pred_logits_plan = self.state_transition_for_planning((state_latents, action_cond))

            neg_log_probs = self.plan_state_transition_beta_distr(next_state_pred_logits_plan, target = next_target_state_latents.detach())
            plan_state_pred_loss = neg_log_probs.sum(dim = -1).mean()
        else:
            # deterministic

            pred_residual_for_planning = self.state_transition_for_planning((state_latents, action_cond))

            next_state_pred_for_plan = state_latents + pred_residual_for_planning

            plan_state_pred_loss = F.smooth_l1_loss(next_state_pred_for_plan, next_target_state_latents.detach())

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

        ema_encoded_state, ema_sensory_layer_hiddens = self.encode_states(self.ema_state_encoder, states)

        next_ema_encoded_state_target = ema_encoded_state[:, 1:]

        next_encoded_state_pred_loss = F.smooth_l1_loss(pred_next_encoded_state, next_ema_encoded_state_target.detach())

        # maybe behavior clone

        actor_loss = self.zero

        if self.has_actor and behavior_clone:
            actor_encoded_states = self.actor_state_encoder(cat((state_tokens[:, :-1], state_latents.detach()), dim = -1))
            actor_encoded_actions = self.actor_action_encoder(cat((action_tokens[:, :action_cond.shape[1]], action_cond.detach()), dim = -1))

            actor_tokens = rearrange([actor_encoded_states, actor_encoded_actions], 'sa b n d -> b (n sa) d')

            actor_past_layers = []
            actor_past_layers_mask = []

            if self.pass_sensory_hiddens_to_actor:
                projected_sensory = [proj(h[:, :-1].detach()) for proj, h in zip(self.to_actor_sensory_hiddens, sensory_layer_hiddens)]
                actor_sensory_hiddens = [rearrange([h, torch.zeros_like(h)], 'sa b n d -> b (n sa) d') for h in projected_sensory]
                actor_past_layers.extend(actor_sensory_hiddens)

                sensory_mask = tensor([True, False], device = device)
                sensory_mask = repeat(sensory_mask, 'sa -> b (n sa)', b = batch, n = state_len - 1)
                actor_past_layers_mask.extend([sensory_mask] * len(actor_sensory_hiddens))

            if self.pass_world_model_hiddens_to_actor:
                seq_len = actor_tokens.shape[1]
                detached_hiddens = tree_map_tensor(lambda t: t[:, :seq_len].detach(), world_model_hiddens)
                actor_past_layers.extend(detached_hiddens)

            actor_past_layers = tuple(actor_past_layers) if not is_empty(actor_past_layers) else None
            actor_past_layers_mask = tuple(actor_past_layers_mask) if not is_empty(actor_past_layers_mask) else None

            actor_losses = []

            for actor_model, to_next_action_pred in zip(self.actor_models, self.to_next_action_preds):
                actor_embed = actor_model(actor_tokens, past_layers = actor_past_layers, past_layers_mask = actor_past_layers_mask)

                actor_state_embed, _ = rearrange(actor_embed, 'b (n sa) d -> sa b n d', sa = 2)

                next_action_pred = to_next_action_pred(actor_state_embed)

                # log probs

                next_actions = orig_actions[:, :next_action_pred.shape[1]]

                if self.continuous_actions:
                    # use unimodal beta

                    neg_log_probs = self.action_beta_distr(next_action_pred, target = next_actions)
                    loss = neg_log_probs.sum(dim = -1).mean()

                else:
                    loss = F.cross_entropy(rearrange(next_action_pred, 'b n c -> b c n'), next_actions)

                actor_losses.append(loss)

            actor_loss = sum(actor_losses)

        # value head

        value_loss = self.zero

        pred_values = self.value_head((state_tokens, self.frac_gradient(state_latents_full)))
        pred_values = rearrange(pred_values, '... 1 -> ...')

        if exists(returns):
            value_loss = F.mse_loss(pred_values, returns)

        # regularizer loss

        reg_next_state_loss = self.zero
        reg_next_encoded_loss = self.zero

        if self.reg_next_state_weight > 0.:
            reg_next_state_loss = self.reg(next_state_pred)

        if self.reg_next_encoded_weight > 0.:
            reg_next_encoded_loss = self.reg(pred_next_encoded_state)

        # action latent uniform loss

        action_wasserstein_loss = self.zero

        if self.action_latent_wasserstein_loss_weight > 0.:
            action_wasserstein_loss = uniform_wasserstein_loss(action_cond)

        # temporal straightening loss

        temporal_straightening_loss_val = self.zero

        if self.temporal_straightening_loss_weight > 0.:
            temporal_straightening_loss_val = temporal_straightening_loss(state_latents_full)

        # goal generator loss

        goal_loss = self.zero

        if self.learn_goal_generator and exists(returns):
            flat_state_latents = rearrange(state_latents_full.detach(), '... d -> (...) d')

            flat_returns = rearrange(returns, '... -> (...)')

            goal_loss = self.goal_flow_matching(flat_state_latents, returns = flat_returns)

        # losses

        total_loss = (
            next_state_latent_pred_loss +
            plan_state_pred_loss * self.plan_state_pred_loss_weight +
            action_recon_loss * self.action_recon_loss_weight +
            next_encoded_state_pred_loss * self.next_encoded_state_pred_loss_weight +
            actor_loss * self.actor_loss_weight +
            value_loss * self.value_loss_weight +
            reg_next_state_loss * self.reg_next_state_weight +
            reg_next_encoded_loss * self.reg_next_encoded_weight +
            action_wasserstein_loss * self.action_latent_wasserstein_loss_weight +
            temporal_straightening_loss_val * self.temporal_straightening_loss_weight +
            goal_loss * self.goal_loss_weight
        )

        loss_breakdown = Losses(
            next_state_latent_pred_loss,
            plan_state_pred_loss,
            action_recon_loss,
            next_encoded_state_pred_loss,
            actor_loss,
            value_loss,
            reg_next_state_loss,
            reg_next_encoded_loss,
            action_wasserstein_loss,
            goal_loss,
            temporal_straightening_loss_val
        )

        out = (total_loss, loss_breakdown)
        return (out, next_memories) if return_memories else out
