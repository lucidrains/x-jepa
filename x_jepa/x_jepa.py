from __future__ import annotations
from typing import Callable, Literal

import math
import inspect
from random import random
from contextlib import nullcontext
from functools import partial, reduce as ft_reduce
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch.distributions import Beta, Categorical
from torch.optim import Adam, Optimizer
from torch.utils._pytree import tree_map, tree_flatten
from torch.func import functional_call, vmap

import einx
from torch import nn, cat, stack, tensor, is_tensor, Tensor
from torch.nn import Module, ModuleList, ModuleDict, Linear, RMSNorm, Sequential

from einops import einsum, rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from ema_pytorch import EMA
from assoc_scan import AssocScan

from x_mlps_pytorch import MLP

from torch_einops_utils import (
    masked_mean,
    pad_right_at_dim_to,
    temp_eval,
    batched_index_select,
    tree_map_tensor,
    tree_flatten_with_inverse,
    pack_with_inverse,
    lens_to_mask,
    maybe,
    safe_cat,
    tree_map_detach,
    tree_map_tensor_to_device
)

from torch_einops_utils.device import module_device

from PoPE_pytorch import PoPE, apply_pope_to_qk

from x_jepa.utils import EnvWrapper, Experience

from x_jepa.regularizers import SigReg, uniform_wasserstein_loss, temporal_straightening_loss
from x_jepa.min_gru import minGRUBlocks
from x_jepa.goals import GoalGenerator, RandomSinusoidalPosEmb
from x_jepa.flow_matching import FlowMatching

# constants

LinearNoBias = partial(Linear, bias = False)

States = Tensor | tuple[Tensor | list[Tensor], ...] | list[Tensor | list[Tensor]]

Losses = namedtuple('Losses', [
    'next_state_latent_pred',
    'plan_state_pred',
    'action_recon',
    'next_encoded_state_pred',
    'actor',
    'actor_losses',
    'value',
    'reg_next_state',
    'reg_next_encoded',
    'action_wasserstein',
    'goal',
    'temporal_straightening',
    'align_pre_state_action_repr',
    'align_pre_state_action_repr_sigreg',
    'cross_sensory_align',
    'cross_sensory_align_sigreg',
    'cross_sensory_align_breakdown',
    'intrinsics',
    'intrinsics_breakdown',
    'quasimetric_distance'
])

CrossSensoryPairLoss = namedtuple('CrossSensoryPairLoss', ['src', 'tgt', 'loss'])

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def identity(t):
    return t

def is_empty(t):
    return len(t) == 0

def bernoulli(prob):
    return random.random() < prob

def xnor(x, y):
    return x == y

def cast_tuple(t, length = 1):
    return t if isinstance(t, (tuple, list)) else ((t,) * length)

def first_tensor(t):
    tensors, _ = tree_flatten(t)
    return first(tensors)

def first(t):
    return None if is_empty(t) else t[0]

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def safe_divide(num, den, eps = 1e-5):
    if not is_tensor(den):
        return num / max(den, eps)

    return num / den.clamp(min = eps)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def l1norm(t, dim = -1):
    return F.normalize(t, p = 1, dim = dim)

def batch_repeat(t, r):
    return repeat(t, 'b ... -> (b r) ...', r = r)

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def SmallInitEmbed(shape, eps = 0.02):
    return nn.Parameter(torch.randn(shape) * eps)

def get_module_by_path(model, path):
    return ft_reduce(getattr, path.split('.'), model)

def set_module_by_path(model, path, new_mod):
    *parts, last = path.split('.')
    mod = ft_reduce(getattr, parts, model)
    setattr(mod, last, new_mod)

# film, for perception gating for starters, but eventually will be used for configurator modulation in the jepa diagram

class FiLM(Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.gamma + self.beta

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
        dim,
        use_pseudo_queries = False,
        dim_head = None
    ):
        super().__init__()
        self.use_pseudo_queries = use_pseudo_queries

        self.norm_keys = RMSNorm(dim)

        dim_head = default(dim_head, dim)
        self.scale = dim_head ** -0.5

        assert divisible_by(dim, dim_head), f'dimension {dim} must be divisible by dim_head {dim_head}'
        heads = dim // dim_head

        self.split_heads = Rearrange('l m (h d) -> h l m d', h = heads)
        self.merge_heads = Rearrange('h m d -> m (h d)')

        if use_pseudo_queries:
            self.pseudo_queries = nn.Parameter(torch.zeros(heads, dim_head))
            nn.init.normal_(self.pseudo_queries, std = 0.02)
        else:
            self.to_query = LinearNoBias(dim, dim)
            self.split_queries_heads = Rearrange('m (h d) -> h m d', h = heads)

        self.register_token = SmallInitEmbed(dim)

    @property
    def device(self):
        return module_device(self)

    def forward(
        self,
        hiddens,
        mask = None
    ):
        device = self.device

        if isinstance(hiddens, list):
            hiddens = stack(hiddens)

        hiddens, unpack = pack_with_inverse(hiddens, 'l * d')

        # queries and similarities

        if not self.use_pseudo_queries:
            *_, last_layer_output = hiddens
            queries = self.split_queries_heads(self.to_query(last_layer_output))

        # add register token

        m = hiddens.shape[1]
        reg_token = repeat(self.register_token, 'd -> 1 m d', m = m)
        hiddens = torch.cat((reg_token, hiddens), dim = 0)

        # cross attention

        values = hiddens
        keys = self.norm_keys(values)

        keys, values = self.split_heads(keys), self.split_heads(values)

        if self.use_pseudo_queries:
            sim = einsum(self.pseudo_queries, keys, 'h d, h l m d -> h m l') * self.scale
        else:
            sim = einsum(queries, keys, 'h m d, h l m d -> h m l') * self.scale

        # masking

        if exists(mask):
            mask = stack(mask) if isinstance(mask, (list, tuple)) else mask

            mask, _ = pack_with_inverse(mask, 'l *')

            reg_mask = torch.ones((1, mask.shape[1]), dtype = torch.bool, device = device)
            mask = torch.cat((reg_mask, mask), dim = 0)

            mask = rearrange(mask, 'l m -> m l')

            mask = pad_right_at_dim_to(mask, sim.shape[-1], value = True)

            sim = einx.where('m l, h m l, -> h m l', mask, sim, max_neg_value(sim))

        # attention and aggregate

        attn = sim.softmax(dim = -1)

        out = einsum(attn, values, 'h m l, h l m d -> h m d')
        out = self.merge_heads(out)

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

# episodic memories

class EpisodicMemories(Module):
    def __init__(
        self,
        depth,
        heads,
        seq_len,
        dim_head
    ):
        super().__init__()
        self.mem_kv = nn.Parameter(torch.randn(2, depth, heads, seq_len, dim_head))
        nn.init.normal_(self.mem_kv, std = 0.02)

    def forward(self, batch_size):
        mem_kv = repeat(self.mem_kv, '... -> b ...', b = batch_size)
        mem_k, mem_v = mem_kv.unbind(dim = 1)
        return [(k, v) for k, v in zip(mem_k.unbind(dim = 1), mem_v.unbind(dim = 1))]

# transformer

class AdaNormWrapper(Module):
    def __init__(self, dim, dim_cond_hidden, block):
        super().__init__()
        self.block = block
        self.norm = RMSNorm(dim)

        self.proj = nn.Linear(dim_cond_hidden, dim * 3)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens, cond = None, **kwargs):
        assert exists(cond)

        gamma, beta, alpha = self.proj(cond).chunk(3, dim = -1)

        normed_tokens = self.norm(tokens)
        normed_tokens = normed_tokens * (1 + gamma) + beta

        out = self.block(normed_tokens, **kwargs)

        (out, *rest), pack_out = tree_flatten_with_inverse(out)

        out = out * alpha

        return pack_out((out, *rest))

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
        use_pope = False,
        use_pseudo_queries = False,
        attn_res_dim_head = None,
        dim_cond = None
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.dim_cond = dim_cond

        self.pope = PoPE(dim_head, heads = heads) if use_pope else None

        has_cond = exists(dim_cond)

        self.cond_mlp = nn.Sequential(
            nn.Linear(dim_cond, dim * 2),
            nn.SiLU()
        ) if has_cond else None

        layers = ModuleList([])

        for _ in range(depth):
            prenorm = not has_cond
            attn = Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, prenorm = prenorm)
            attn_res = AttentionResidual(dim = dim, use_pseudo_queries = use_pseudo_queries, dim_head = attn_res_dim_head)
            ff = SwiGLUFeedForward(dim = dim, expand_factor = ff_expand_factor, prenorm = prenorm)

            if has_cond:
                attn = AdaNormWrapper(dim, dim * 2, attn)
                ff = AdaNormWrapper(dim, dim * 2, ff)

            layers.append(ModuleList([attn_res, attn, ff]))

        self.layers = layers

    def forward(
        self,
        tokens,
        past_layers = None,
        past_layers_mask = None,
        return_hiddens = False,
        memories = None,
        prepend_memories = None,
        return_memories = False,
        cond = None
    ):
        has_cond = exists(self.dim_cond)

        seq_len = tokens.shape[-2]

        if has_cond:
            assert exists(cond)
            cond = self.cond_mlp(cond)

            if cond.ndim == 2:
                cond = rearrange(cond, 'b d -> b 1 d')

        past_seq_len = 0
        layer_memories = None

        if exists(memories):
            past_seq_len, layer_memories = memories

        pope_pos_emb = self.pope(seq_len, offset = past_seq_len) if exists(self.pope) else None

        layer_hiddens = list(default(past_layers, []))
        layer_hiddens.append(tokens)

        next_layer_memories = []

        memories_iter = iter(layer_memories) if exists(layer_memories) else None
        prepend_memories_iter = iter(prepend_memories) if exists(prepend_memories) else None

        for attn_res, attn, ff in self.layers:
            layer_memory = next(memories_iter, None) if exists(memories_iter) else None

            if exists(prepend_memories_iter):
                prep_memory = next(prepend_memories_iter)
                layer_memory = tuple(safe_cat((p, m), dim = -2) for p, m in zip(prep_memory, layer_memory or (None, None)))

            attn_kwargs = dict(
                pope_pos_emb = pope_pos_emb,
                memories = layer_memory,
                return_memories = True
            )

            if has_cond:
                attn_kwargs.update(cond = cond)

            attn_out, next_memory = attn(tokens, **attn_kwargs)

            if return_memories:
                next_layer_memories.append(next_memory)

            tokens = attn_out + tokens

            ff_kwargs = dict(cond = cond) if has_cond else dict()

            tokens = ff(tokens, **ff_kwargs) + tokens

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

# value / cost network section

def calc_returns(
    rewards,
    is_done,
    next_values,
    discount_factor,
    mask = None,
    lens = None
):
    seq_len = rewards.shape[1]

    assert not (exists(mask) and exists(lens)), 'either mask or lens is given, but not both'

    mask = default(mask, maybe(lens_to_mask)(lens, seq_len))

    scan = AssocScan(reverse = True)

    if is_tensor(discount_factor) and discount_factor.ndim == 1:
        discount_factor = rearrange(discount_factor, 'b -> b 1')

    gates = discount_factor * (~is_done).float()

    if exists(mask):
        gates = torch.where(mask, gates, torch.ones_like(gates))
        rewards = torch.where(mask, rewards, torch.zeros_like(rewards))

    returns = scan(gates, rewards, prev = next_values)

    if exists(mask):
        returns = returns.masked_fill(~mask, 0.)

    return returns

class Value(Module):
    def __init__(
        self,
        dim,
        dim_state_latent,
        discount_factor = 0.99,
        ema_beta = 0.999,
        eps = 1e-20
    ):
        super().__init__()
        self.eps = eps

        self.discount_embedder = Sequential(
            LinearNoBias(3, dim),
            nn.SiLU(),
            LinearNoBias(dim, dim)
        )
        self.net = MLP(dim_state_latent + dim + dim, *((dim * 2,) * 2), 1)
        self.ema_net = EMA(self.net, beta = ema_beta)

        self.register_buffer('discount_factor', tensor(discount_factor), persistent = False)

    def embed_discount(self, discount):
        cond = stack([
            discount,
            1. - discount,
            -log(1. - discount, eps = self.eps)
        ], dim = -1)
        return self.discount_embedder(cond)

    def get_discount_embed(self, state_tokens, discount = None):
        discount = default(discount, self.discount_factor)
        discount = discount.broadcast_to(state_tokens.shape[:-1])
        return self.embed_discount(discount)

    def forward_ema(self, state_tokens, state_latents, discount = None):
        discount_embed = self.get_discount_embed(state_tokens, discount)
        return self.ema_net((state_tokens, state_latents, discount_embed))

    def update_ema(self):
        self.ema_net.update()

    def forward(self, state_tokens, state_latents, discount = None):
        discount_embed = self.get_discount_embed(state_tokens, discount)
        return self.net((state_tokens, state_latents, discount_embed))

# classes

class Actor(Module):
    def __init__(
        self,
        continuous_actions,
        dim_action,
        action_eps = 1e-5
    ):
        super().__init__()
        self.continuous_actions = continuous_actions
        self.dim_action = dim_action
        if continuous_actions:
            self.action_distr = BetaDistrReadout(source_range = (-1., 1.), eps = action_eps)

    def compute_loss(self, action_preds, target_actions):
        if self.continuous_actions:
            neg_log_probs = self.action_distr(action_preds, target = target_actions)
            return reduce(neg_log_probs, '... d -> ...', 'sum').mean()

        return F.cross_entropy(
            rearrange(action_preds, 'b ... c -> b c ...'),
            target_actions
        )

    def sample_actions(
        self,
        action_preds,
        temperature = 1.0,
        return_log_prob = False
    ):
        if self.continuous_actions:
            action = self.action_distr(action_preds, sample = True, temperature = temperature)

            if not return_log_prob:
                return action

            log_probs = -self.action_distr(action_preds, target = action)
            log_probs = reduce(log_probs, '... d -> ...', 'sum')
            return action, log_probs

        logits = safe_divide(action_preds, temperature)
        dist = Categorical(logits = logits)
        action = dist.sample()

        if not return_log_prob:
            return action

        return action, dist.log_prob(action)

    def get_action_preds(self, **kwargs):
        raise NotImplementedError

    def sample(
        self,
        temperature = 1.0,
        return_log_prob = False,
        **kwargs
    ):
        return_memories = kwargs.get('return_memories', False)
        preds = self.get_action_preds(**kwargs)

        next_memories = None
        if return_memories:
            preds, next_memories = preds

        out = self.sample_actions(preds, temperature, return_log_prob)

        if not return_memories:
            return out

        out = cast_tuple(out)
        return (*out, next_memories)

    def forward(self, target_actions = None, **kwargs):
        preds = self.get_action_preds(**kwargs)

        if not exists(target_actions):
            return preds

        return self.compute_loss(preds, target_actions)

class ReflexiveActor(Actor):
    def __init__(
        self,
        dim_state_latent,
        dim_action,
        continuous_actions,
        hidden_dim = 256,
        action_eps = 1e-5
    ):
        super().__init__(continuous_actions, dim_action, action_eps = action_eps)
        dim_out = dim_action * 2 if continuous_actions else dim_action
        self.net = MLP(dim_state_latent, hidden_dim, dim_out)

    def get_action_preds(self, state_latents, return_memories = False, **kwargs):
        pred = self.net(state_latents)

        if not return_memories:
            return pred

        return pred, None

class TransformerActor(Actor):
    def __init__(
        self,
        dim,
        dim_state_latent,
        dim_action,
        dim_transition_action_input,
        continuous_actions,
        model: Module,
        num_sensory_views = 0,
        pass_sensory_hiddens = False,
        pass_world_model_hiddens = True,
        dropout_all_but_state_latents = 0.,
        action_eps = 1e-5
    ):
        super().__init__(
            continuous_actions,
            dim_action,
            action_eps = action_eps
        )

        self.model = model

        self.dim = dim
        self.dim_action_input = dim_transition_action_input

        self.state_encoder = MLP(dim_state_latent + dim, dim)
        self.action_encoder = MLP(dim_transition_action_input + dim, dim)

        dim_out = dim_action * 2 if continuous_actions else dim_action
        self.to_action_pred = Sequential(RMSNorm(dim), LinearNoBias(dim, dim_out))

        self.pass_sensory_hiddens = pass_sensory_hiddens

        if pass_sensory_hiddens:
            self.to_sensory_hiddens = ModuleList([LinearNoBias(dim, dim) for _ in range(num_sensory_views)])

        self.pass_world_model_hiddens = pass_world_model_hiddens
        self.dropout_all_but_state_latents = dropout_all_but_state_latents
        self.has_dropout = dropout_all_but_state_latents > 0.

    def get_action_preds(
        self,
        state_latents,
        state_tokens = None,
        action_cond = None,
        action_tokens = None,
        sensory_layer_hiddens = None,
        world_model_hiddens = None,
        memories = None,
        return_memories = False,
        **kwargs
    ):
        batch, seq_len = state_latents.shape[:2]

        # condition dropout for unconditional seeding

        if self.has_dropout and self.training:
            if bernoulli(self.dropout_all_but_state_latents):
                state_tokens = None
                action_cond = None
                action_tokens = None
                sensory_layer_hiddens = None
                world_model_hiddens = None

        # handle absent states and actions

        def default_zeros(t, feature_dim):
            return state_latents.new_zeros(*state_latents.shape[:-1], feature_dim) if not exists(t) else t[:, :seq_len]

        state_tokens = default_zeros(state_tokens, self.dim)

        # project

        actor_encoded_states = self.state_encoder(cat((state_tokens, state_latents), dim = -1))

        if not exists(action_cond):
            actor_tokens = actor_encoded_states
        else:
            action_cond = default_zeros(action_cond, self.dim_action_input)
            action_tokens = default_zeros(action_tokens, self.dim)
            actor_encoded_actions = self.action_encoder(cat((action_tokens, action_cond), dim = -1))

            actor_tokens = rearrange([actor_encoded_states, actor_encoded_actions], 'sa b n d -> b (n sa) d')

        # inject sensory hiddens and world model hiddens as past layers

        actor_seq_len = actor_tokens.shape[1]

        past_layers = []
        past_layers_mask = []

        has_actions = exists(action_cond)

        if self.pass_sensory_hiddens and exists(sensory_layer_hiddens):
            projected_sensory = [proj(h[:, :seq_len]) for proj, h in zip(self.to_sensory_hiddens, sensory_layer_hiddens)]

            if has_actions:
                sensory_past_layers = [rearrange([h, torch.zeros_like(h)], 'sa b n d -> b (n sa) d') for h in projected_sensory]
                sensory_mask = tensor([True, False], device = state_latents.device)
                sensory_mask = repeat(sensory_mask, 'sa -> b (n sa)', b = batch, n = seq_len)
            else:
                sensory_past_layers = projected_sensory
                sensory_mask = torch.ones((batch, seq_len), dtype = torch.bool, device = state_latents.device)

            past_layers.extend(sensory_past_layers)
            past_layers_mask.extend([sensory_mask] * len(sensory_past_layers))

        if self.pass_world_model_hiddens and exists(world_model_hiddens):
            detached_hiddens = tree_map_tensor(lambda t: t[:, :actor_seq_len], world_model_hiddens)
            past_layers.extend(detached_hiddens)

            wm_masks = [torch.ones((batch, t.shape[1]), dtype = torch.bool, device = t.device) for t in detached_hiddens]
            past_layers_mask.extend(wm_masks)

        past_layers = tuple(past_layers) if not is_empty(past_layers) else None
        past_layers_mask = tuple(past_layers_mask) if not is_empty(past_layers_mask) else None

        # evaluate

        actor_embed = self.model(
            actor_tokens,
            past_layers = past_layers,
            past_layers_mask = past_layers_mask,
            memories = memories,
            return_memories = return_memories
        )

        next_memory = None
        if return_memories:
            actor_embed, next_memory = actor_embed

        if actor_embed.shape[1] > seq_len:
            actor_embed, _ = rearrange(actor_embed, 'b (n sa) d -> sa b n d', sa = 2)

        pred = self.to_action_pred(actor_embed)

        if not return_memories:
            return pred

        return pred, next_memory

# agent / world model

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
        actors: dict[str, Module] | None = None,
        add_reflexive_actor = False,
        add_transformer_actor = False,
        actor_model: Module | None = None,
        pass_world_model_hiddens_to_actor = True,
        pass_sensory_hiddens_to_actor = False,
        sensory_dropout_prob = 0.,
        use_structured_sensory_dropout = False,
        structured_sensory_dropout_keep_prob = 0.1,
        actor_dropout_all_but_state_latents = 0.,
        actor_loss_weights: float | dict[str, float] = 1.,
        dim_action = None,
        continuous_actions = True,
        action_eps = 1e-5,
        action_recon_loss_weight = 1.,
        next_encoded_state_pred_loss_weight = 1.,
        plan_state_pred_loss_weight = 1.,
        value_loss_weight = 1.,
        discount_factor = 0.99,
        pass_sensory_hiddens_to_world_model = False,
        frac_gradients = 0.,
        reg_next_state_weight = 0.,
        reg_next_encoded_weight = 0.,
        action_latent_wasserstein_loss_weight = 0.,
        temporal_straightening_loss_weight = 0.,
        align_pre_state_action_repr_loss_weight = 0.,
        align_pre_state_action_repr_sigreg_weight = 0.,
        learn_goal_generator = False,
        goal_loss_weight = 1.,
        returns_norm_momentum = 0.01,
        probabilistic_state_transition = False,
        probabilistic_plan_state_transition = False,
        state_transition_eps = 1e-5,
        reg: Module | None = None,
        reg_loss_kwargs: dict | None = None,
        state_linear_rnn_depth = 1,
        action_linear_rnn_depth = 1,
        use_perception_film = False,
        num_sensory_views: tuple[int, ...] | None = None,
        cross_sensory_align_pairs: list[tuple[int, int]] | None = None,
        cross_sensory_align_loss_weight: float = 1.,
        cross_sensory_align_sigreg_weight: float = 0.,
        intrinsics: Module | list[Module] | ModuleList | tuple[Module, ...] | None = None,
        intrinsic_loss_weight: float = 1.,
        intrinsic_frac_gradient: float | tuple[float, ...] | list[float] = 0.,
        episodic_mem_len = 0,
        quasimetric_distance_network = None,
        quasimetric_distance_loss_weight = 1.
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
        self.has_state_latent_clamp = state_latent_clamp_value > 0.

        self.has_episodic_mem = episodic_mem_len > 0
        if self.has_episodic_mem:
            self.episodic_mems = EpisodicMemories(
                depth = model.depth,
                heads = model.heads,
                seq_len = episodic_mem_len,
                dim_head = model.dim_head
            )

        # handle state encoders
        # which could have multiple states, and each could have multiple views (ie. eyes, ears)

        if not isinstance(state_encoder, (list, tuple, ModuleList)):
            state_encoder = [state_encoder]

        self.state_encoder = ModuleList(state_encoder)

        if not exists(num_sensory_views):
            num_sensory_views = (1,) * len(self.state_encoder)

        assert len(num_sensory_views) == len(self.state_encoder)
        self.num_sensory_views = num_sensory_views

        self.view_embs = nn.ParameterList([SmallInitEmbed((v, dim)) if v > 1 else None for v in self.num_sensory_views])

        total_sensory_views = sum(self.num_sensory_views)
        self.sensory_mask_tokens = nn.ParameterList([SmallInitEmbed(dim) for _ in range(total_sensory_views)])

        self.sensory_dropout_prob = sensory_dropout_prob
        self.has_sensory_dropout = sensory_dropout_prob > 0.

        self.use_structured_sensory_dropout = use_structured_sensory_dropout
        self.structured_sensory_dropout_keep_prob = structured_sensory_dropout_keep_prob

        assert not (self.has_sensory_dropout and len(self.state_encoder) == 1), 'sensory dropout is only allowed when there are multiple senses'
        assert not (self.use_structured_sensory_dropout and len(self.state_encoder) == 1), 'structured sensory dropout is only allowed when there are multiple senses'

        self.use_perception_film = use_perception_film
        if use_perception_film:
            self.perception_film = FiLM(dim)

        self.only_ttt_updateable_params = ('perception_film',) if use_perception_film else tuple()

        if pass_sensory_hiddens_to_world_model:
            assert len(self.state_encoder) > 1, 'passing sensory layers to attention residual requires more than one modality'

        self.action_encoder = action_encoder

        # initial linear rnns before attending

        self.state_linear_rnn = minGRUBlocks(dim = dim, depth = state_linear_rnn_depth)
        self.action_linear_rnn = minGRUBlocks(dim = dim, depth = action_linear_rnn_depth)

        # the learnt state transition that is popularly used for learning the so called world model, but may not be the only prediction needed

        # different types of action spaces for the transition

        if transition_action_space == 'raw': # raw actions from -1. to 1., only for continuous actions
            assert continuous_actions and exists(dim_action)

            dim_transition_action_input = dim_action
            need_learned_action_decoder = False

        elif transition_action_space in ('local', 'global'):
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
            assert self.has_state_latent_clamp, 'state_latent_clamp_value must be greater than 0 if either probabilistic state transition is turned on'

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

        # actors

        actors = dict(default(actors, dict()))

        if add_reflexive_actor:
            actors['reflexive'] = ReflexiveActor(
                dim_state_latent = dim_state_latent,
                dim_action = dim_action,
                continuous_actions = continuous_actions,
                action_eps = action_eps
            )

        if add_transformer_actor:
            assert exists(actor_model), 'actor_model must be provided if add_transformer_actor is True'

            num_views = len(self.state_encoder)

            actors['transformer'] = TransformerActor(
                dim = dim_state_latent,
                dim_state_latent = dim_state_latent,
                dim_action = dim_action,
                dim_transition_action_input = dim_action_latent if transition_action_space != 'raw' else dim_action,
                continuous_actions = continuous_actions,
                model = actor_model,
                num_sensory_views = num_views,
                pass_sensory_hiddens = pass_sensory_hiddens_to_actor,
                pass_world_model_hiddens = pass_world_model_hiddens_to_actor,
                dropout_all_but_state_latents = actor_dropout_all_but_state_latents,
                action_eps = action_eps
            )

        self.actors = ModuleDict(actors)
        self.has_actors = len(actors) > 0

        if not isinstance(actor_loss_weights, dict):
            actor_loss_weights = {name: actor_loss_weights for name in actors}

        self.actor_loss_weights = actor_loss_weights

        self.pass_sensory_hiddens_to_world_model = pass_sensory_hiddens_to_world_model

        self.continuous_actions = continuous_actions
        self.action_eps = action_eps

        # value head

        self.value_network = Value(dim, dim_state_latent, discount_factor, ema_beta)
        self.frac_gradient = FracGradient(frac_gradients)

        # loss related

        self.next_encoded_state_pred_loss_weight = next_encoded_state_pred_loss_weight
        self.plan_state_pred_loss_weight = plan_state_pred_loss_weight

        self.action_recon_loss_weight = action_recon_loss_weight

        self.value_loss_weight = value_loss_weight

        self.action_latent_wasserstein_loss_weight = action_latent_wasserstein_loss_weight

        # experimental alignment between states and actions
        # note: one's actions is a special modality, should be contrasted with other actions in multiagent setting

        self.align_pre_state_action_repr_loss_weight = align_pre_state_action_repr_loss_weight
        self.align_pre_state_action_repr_sigreg_weight = align_pre_state_action_repr_sigreg_weight

        self.has_align_pre_state_action_repr_loss = align_pre_state_action_repr_loss_weight > 0.
        self.has_align_pre_state_action_repr_sigreg = align_pre_state_action_repr_sigreg_weight > 0.

        if self.has_align_pre_state_action_repr_loss:
            self.state_to_action_pred = MLP(dim, *((dim * 2,) * 2), dim)
            self.action_to_state_pred = MLP(dim, *((dim * 2,) * 2), dim)

        # cross sensory alignment (vljepa method)

        self.cross_sensory_align_pairs = cross_sensory_align_pairs
        self.cross_sensory_align_loss_weight = cross_sensory_align_loss_weight
        self.cross_sensory_align_sigreg_weight = cross_sensory_align_sigreg_weight

        self.has_cross_sensory_align = exists(cross_sensory_align_pairs) and len(cross_sensory_align_pairs) > 0
        self.has_cross_sensory_align_sigreg = cross_sensory_align_sigreg_weight > 0.

        if self.has_cross_sensory_align:
            self.cross_sensory_preds = ModuleList([])

            for src_idx, tgt_idx in self.cross_sensory_align_pairs:
                src_dim = self.num_sensory_views[src_idx] * dim
                tgt_dim = self.num_sensory_views[tgt_idx] * dim
                self.cross_sensory_preds.append(MLP(src_dim, src_dim * 2, src_dim * 2, tgt_dim))

        # goal generator

        self.learn_goal_generator = learn_goal_generator
        self.goal_loss_weight = goal_loss_weight

        if self.learn_goal_generator:
            self.goal_generator = GoalGenerator(
                dim = dim_state_latent,
                returns_norm_momentum = returns_norm_momentum # controls how fast the agent ascends the hedonic treadmill
            )
            self.goal_flow_matching = FlowMatching(model = self.goal_generator)

        # quasimetric

        self.quasimetric_distance_network = quasimetric_distance_network
        self.quasimetric_distance_loss_weight = quasimetric_distance_loss_weight

        # intrinsics

        self.intrinsic_loss_weight = intrinsic_loss_weight

        if not exists(intrinsics):
            intrinsics = []
        elif not isinstance(intrinsics, (list, tuple, ModuleList)):
            intrinsics = [intrinsics]

        self.intrinsics = ModuleList(intrinsics)
        self.has_intrinsics = len(self.intrinsics) > 0

        intrinsic_frac_gradient = cast_tuple(intrinsic_frac_gradient, len(self.intrinsics))
        assert len(intrinsic_frac_gradient) == len(self.intrinsics)
        self.intrinsic_frac_gradients = ModuleList([FracGradient(frac) for frac in intrinsic_frac_gradient])

        # regularizer - defaults to sigreg, but any drop-in (ex. visreg) can be passed in

        self.reg = default(reg, SigReg(**default(reg_loss_kwargs, dict())))

        self.reg_next_state_weight = reg_next_state_weight
        self.reg_next_encoded_weight = reg_next_encoded_weight

        self.action_latent_wasserstein_loss_weight = action_latent_wasserstein_loss_weight
        self.temporal_straightening_loss_weight = temporal_straightening_loss_weight

        self.has_reg_next_state = reg_next_state_weight > 0.
        self.has_reg_next_encoded = reg_next_encoded_weight > 0.
        self.has_action_latent_wasserstein_loss = action_latent_wasserstein_loss_weight > 0.
        self.has_temporal_straightening_loss = temporal_straightening_loss_weight > 0.
        self.discount_factor = discount_factor

        self.register_buffer('zero', tensor(0.), persistent = False)

    @property
    def device(self):
        return self.zero.device

    def parameters(self, recurse = True):
        for name, param in self.named_parameters(recurse = recurse):
            if any(keyword in name for keyword in self.only_ttt_updateable_params):
                continue
            yield param

    def encode_states(
        self,
        encoder: ModuleList,
        states: States,
        masks = None
    ):
        if is_tensor(states):
            states = [states]

        assert len(encoder) == len(states), 'number of states must match number of state encoders'

        encoded = []
        sensory_layer_hiddens = []
        sensory_for_alignment = []

        total_view_idx = 0
        masks = default(masks, (None,) * len(encoder))

        for enc, state, views, view_emb, mask in zip(encoder, states, self.num_sensory_views, self.view_embs, masks):
            has_multi_views = views > 1

            if has_multi_views:
                assert isinstance(state, (list, tuple)) and len(state) == views
                state = rearrange(list(state), 'v b ... -> (v b) ...')

            embed = enc(state)

            if has_multi_views:
                embed = rearrange(embed, '(v b) ... -> v b ...', v = views)

                if exists(view_emb):
                    embed = einx.add('v ... d, v d -> v ... d', embed, view_emb)

                sensory_for_alignment.append(rearrange(embed, 'v b n d -> b n (v d)'))
            else:
                sensory_for_alignment.append(embed)

            embeds = list(embed) if has_multi_views else [embed]

            if exists(mask):
                mask = rearrange(mask, 'b n -> b n 1')

            for view_embed in embeds:
                sum_embed = view_embed

                if exists(mask):
                    mask_token = self.sensory_mask_tokens[total_view_idx]
                    sum_embed = torch.where(mask, view_embed, mask_token)

                encoded.append(sum_embed)
                sensory_layer_hiddens.append(view_embed)
                total_view_idx += 1

        return sum(encoded), sensory_layer_hiddens, sensory_for_alignment

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
        cem_ema_decay = 1.,
        cem_min_var = 1e-5,
        cem_temperature = 0.,
        gradient_steps = 0,
        gradient_lr = 1e-2,
        gradient_optimizer: Callable[..., Optimizer] = Adam,
        seed_with_actor: str | None = None,
        actor_temperature = 1.,
        clamp_state_latent_to_range = True,
        return_action_latent = False,
        search_space: Literal['raw', 'local_global'] | None = None,
        memories = None,
        return_memories = False
    ):
        batch, device = first_tensor(states).shape[0], self.device
        state_len, action_len = first_tensor(states).shape[1], actions.shape[1]

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

        has_cem_temperature = cem_temperature > 0.
        has_gradient_steps = gradient_steps > 0

        # validation

        assert exists(fitness_fn) or exists(goal_state), 'either fitness_fn or goal_state must be provided'
        assert action_len == (state_len - 1)
        assert generations > 0
        assert horizon >= 1

        actions = pad_right_at_dim_to(actions, state_len, dim = 1)

        # get the state and action latents

        state_tokens, sensory_layer_hiddens, _ = self.encode_states(self.state_encoder, states)
        action_tokens = self.action_encoder(actions)

        # handle memories

        state_rnn_memories = action_rnn_memories = model_memories = None

        if exists(memories):
            state_rnn_memories, action_rnn_memories, model_memories = memories

        next_model_memories = dict()

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

        prepend_memories = None
        if self.has_episodic_mem and not exists(model_memories):
            prepend_memories = self.episodic_mems(batch)

        (embeds, _), next_model_memories = self.ema_model(
            tokens,
            return_hiddens = True,
            memories = model_memories,
            prepend_memories = prepend_memories,
            return_memories = True
        )

        state_embeds = embeds[:, -2] # (b d)

        state_latents = self.to_state_latent(state_embeds)

        state_latents = repeat(state_latents, 'b d -> b p d', p = pop_size)

        # start with naive cross entropy method

        num_elites = max(int(elite_frac * pop_size), 1)

        shape = (batch, 1, horizon, dim_action)

        means = torch.rand(shape, device = device) * 2. - 1.
        stds = torch.rand(shape, device = device)
        variances = (stds ** 2).clamp_min(cem_min_var)        # precompute past rnn memories for rnn action latents, repeating to population size

        past_rnn_memories = tree_map_tensor(partial(batch_repeat, r = pop_size), next_action_rnn_memories)

        # helper for encoding a single action step

        def encode_action_step(action, memories):
            if not (is_search_space_raw_action and not self.is_transition_action_space_raw):
                return action, memories

            action, unpack = pack_with_inverse(action, '* d')
            encoded = self.action_encoder(action)

            if self.transition_action_space == 'global':
                encoded, memories = self.action_linear_rnn(
                    rearrange(encoded, '... d -> ... 1 d'),
                    memories = memories,
                    return_memories = True
                )
                encoded = rearrange(encoded, '... 1 d -> ... d')

            return unpack(self.to_action_latent(encoded)), memories

        # helper for advancing state

        def advance_state(state, action, return_entropy = False):
            logits = self.ema_state_transition_for_planning((state, action))

            if self.probabilistic_plan_state_transition:
                return self.plan_state_transition_beta_distr(logits, sample = True, return_entropy = return_entropy)

            next_state = state + logits

            if clamp_state_latent_to_range and self.has_state_latent_clamp:
                next_state.clamp_(-self.state_latent_clamp_value, self.state_latent_clamp_value)

            return (next_state, None) if return_entropy else next_state

        # actor seeding logic

        seed_actor = None
        if exists(seed_with_actor) and self.has_actors and seed_with_actor in self.actors:
            seed_actor = self.actors[seed_with_actor]

        if exists(seed_actor):
            step_state, step_rnn_memories = state_latents, past_rnn_memories
            seeded_actions = []
            seed_actor_memory = None
            seed_last_action_cond = None
            seed_last_action_tokens = None

            for _ in range(horizon):
                step_state_reshaped, unpack_actor = pack_with_inverse(step_state, '* d')
                step_state_reshaped = rearrange(step_state_reshaped, 'b d -> b 1 d')

                action_kwargs = dict(
                    state_latents = step_state_reshaped,
                    temperature = actor_temperature,
                    memories = seed_actor_memory,
                    return_memories = True
                )

                if exists(seed_last_action_cond):
                    action_kwargs.update(
                        action_cond = seed_last_action_cond,
                        action_tokens = seed_last_action_tokens
                    )

                step_action, seed_actor_memory = seed_actor.sample(**action_kwargs)

                step_action_seq = rearrange(step_action, 'b ... -> b 1 ...')
                seed_last_action_tokens = self.action_encoder(step_action_seq)
                if self.is_transition_action_space_raw:
                    seed_last_action_cond = step_action_seq
                else:
                    seed_last_action_cond = self.to_action_latent(seed_last_action_tokens)

                step_action = rearrange(step_action, 'b 1 ... -> b ...')
                step_action = unpack_actor(step_action)

                step_action_cond, step_rnn_memories = encode_action_step(step_action, step_rnn_memories)

                seeded_actions.append(step_action if is_search_space_raw_action else step_action_cond)
                step_state = advance_state(step_state, step_action_cond)

            seeded_actions = stack(seeded_actions, dim = 2)

        encoded_goal = None

        if exists(goal_state):
            if is_tensor(goal_state):
                goal_state = [goal_state]

            encoded_goal, _, _ = self.encode_states(self.ema_state_encoder, goal_state)
            encoded_goal = rearrange(encoded_goal, 'b d -> b 1 1 d')

        # introspect fitness_fn once

        fitness_fn_params = inspect.signature(fitness_fn).parameters if exists(fitness_fn) else None

        # helper for evaluating actions

        def evaluate_actions(eval_actions):
            # the action condition into the step

            # the state transition for planning could take raw actions or action latents depending on the transition action space

            actions_cond = eval_actions

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

                step_state_latents, step_entropy = advance_state(step_state_latents, step_action_cond, return_entropy = True)

                if exists(step_entropy):
                    pred_state_entropies.append(step_entropy)

                pred_state_latents.append(step_state_latents)

                pred_next_encoded_states.append(pred_next_encoded_state)

                step_pred_value = self.value_network(pred_next_encoded_state, step_state_latents)
                pred_values.append(step_pred_value)

            pred_state_latents = rearrange(pred_state_latents, 'h b p d -> b p h d')
            if exists(pred_state_entropies):
                pred_state_entropies = rearrange(pred_state_entropies, 'h b p d -> b p h d')

            pred_next_encoded_states = rearrange(pred_next_encoded_states, 'h b p d -> b p h d')
            pred_values = rearrange(pred_values, 'h b p 1 -> b p h')

            # evaluate

            if exists(fitness_fn):
                # simple introspect and dependency inject into fitness function

                kwargs = dict(
                    pred_state_latents = pred_state_latents,
                    pred_next_encoded_states = pred_next_encoded_states,
                    pred_values = pred_values,
                    encoded_goal = encoded_goal,
                    pred_state_entropies = pred_state_entropies
                )

                if 'pred_intrinsic_bonuses' in fitness_fn_params:
                    assert self.has_intrinsics, 'intrinsics must be passed into WorldModel'

                    flat_pred_state_latents, unpack = pack_with_inverse(pred_state_latents, '* d')
                    bonuses = tuple(unpack(intrinsic.compute_bonus(flat_pred_state_latents), '*') for intrinsic in self.intrinsics)

                    kwargs.update(pred_intrinsic_bonuses = bonuses)

                allowed_params = set(kwargs.keys())
                unknown_params = set(fitness_fn_params.keys()) - allowed_params
                assert is_empty(unknown_params), f"fitness_fn accepts unknown parameters: {unknown_params}. Allowed parameters are: {', '.join(allowed_params)}"

                fitness_fn_kwargs = {k: v for k, v in kwargs.items() if k in fitness_fn_params}
                fitnesses = fitness_fn(**fitness_fn_kwargs)

            elif not exists(goal_dist_fn) and exists(self.quasimetric_distance_network):
                # evaluate MRN distance on predicted states and goal
                # enforces state (left) to goal (right) encoder asymmetry

                _, p, h, _ = pred_state_latents.shape
                encoded_goal_expanded = encoded_goal.expand(-1, p, h, -1)
                distances = self.quasimetric_distance_network(pred_state_latents, encoded_goal_expanded)
                fitnesses = -reduce(distances, 'b p h -> b p', 'sum')
            else:
                # default to simple distance function if no fitness function or quasimetric distance provided

                dist_fn = default(goal_dist_fn, partial(F.smooth_l1_loss, reduction = 'none'))
                distance_to_goal = dist_fn(pred_next_encoded_states, encoded_goal.expand_as(pred_next_encoded_states))
                fitnesses = -reduce(distance_to_goal, 'b p h d -> b p', 'sum')

            return fitnesses

        # iterate

        for generation in range(generations):
            is_first = generation == 0
            is_last = generation == generations - 1

            # instantiate population

            if is_first and exists(seed_actor):
                actions = seeded_actions
            else:
                actions = means + variances.sqrt() * torch.randn((batch, pop_size, horizon, dim_action), device = device)

            actions = actions.clamp(-1., 1.)

            # gradient updates for actions

            if has_gradient_steps:
                actions = actions.detach().requires_grad_()
                optimizer = gradient_optimizer([actions], lr = gradient_lr)

                with torch.enable_grad():
                    for _ in range(gradient_steps):
                        optimizer.zero_grad()
                        loss = -evaluate_actions(actions).sum()
                        loss.backward()
                        optimizer.step()
                        actions.data.clamp_(-1., 1.)

                actions = actions.detach()

            fitnesses = evaluate_actions(actions)

            # select the fittest

            topk = num_elites if not is_last else 1

            elite_indices = fitnesses.topk(topk, largest = True, dim = -1).indices

            # select the elites

            fittest_actions = batched_index_select(actions, elite_indices, dim = 1)

            if not is_last:
                if has_cem_temperature:
                    elite_fitnesses = fitnesses.gather(-1, elite_indices)
                    weights = (elite_fitnesses * cem_temperature).softmax(dim = -1)

                    elite_means = einx.dot('b p, b p h d -> b 1 h d', weights, fittest_actions)

                    diffs = fittest_actions - elite_means
                    elite_vars = einx.dot('b p, b p h d -> b 1 h d', weights, diffs ** 2)
                else:
                    elite_means = reduce(fittest_actions, 'b p h d -> b 1 h d', 'mean')
                    elite_vars = reduce(fittest_actions, 'b p h d -> b 1 h d', partial(torch.var, unbiased = False))

                elite_vars = elite_vars.clamp_min(cem_min_var)

                # ema smoothing

                means.lerp_(elite_means, cem_ema_decay)
                variances.lerp_(elite_vars, cem_ema_decay)

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
        actor_module: str | None = None, # can be any custom key, but 'reflexive' or 'bc' are standard
        actor_temperature = 1.0,
        **plan_kwargs
    ):
        env = EnvWrapper(env, return_cpu = return_cpu)

        states, actions, actor_log_probs, rewards, terminateds, truncateds, infos = [], [], [], [], [], [], []

        state, info = env.reset()

        first_state = state[0] if isinstance(state, (list, tuple)) else state
        batch, device = first_state.shape[0], first_state.device

        memories = None

        is_done = torch.zeros(batch, dtype = torch.bool, device = device)
        cumulative_rewards = torch.zeros(batch, device = device)
        episode_len = torch.zeros(batch, dtype = torch.long, device = device)

        for step in range(max_steps):
            is_last_step = step == (max_steps - 1)

            states.append(state)
            infos.append(info)

            empty_actions = torch.empty(batch, 0, self.dim_action, device = self.device)
            state = tree_map_tensor_to_device(state, self.device)

            # single timestep

            state_seq = tree_map_tensor(lambda t: rearrange(t, 'b ... -> b 1 ...'), state)

            if exists(actor_module):
                # encode state
                ema_encoded_state, ema_encoded_sensory_states, patch_mask = self.encode_states(self.ema_state_encoder, state_seq)
                ema_state_tokens = rearrange(ema_encoded_state, 'b 1 ... -> b ...')
                ema_state_latents = self.to_state_latent(ema_state_tokens)

                actor = self.actors[actor_module]

                action, step_log_prob = actor.sample(
                    state_latents = ema_state_latents,
                    state_tokens = ema_state_tokens,
                    sensory_layer_hiddens = ema_encoded_sensory_states,
                    temperature = actor_temperature,
                    return_log_prob = True
                )

                action = rearrange(action, 'b 1 ... -> b ...')
                step_log_prob = rearrange(step_log_prob, 'b 1 -> b')

                actor_log_probs.append(step_log_prob.cpu() if return_cpu else step_log_prob)
                memories = None
            else:
                planned_actions, memories = self.plan(
                    states = state_seq,
                    actions = empty_actions,
                    fitness_fn = fitness_fn,
                    goal_state = goal_state,
                    memories = memories,
                    return_memories = True,
                    **plan_kwargs
                )
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

        # calculate returns

        states = tree_map(lambda *t: stack(t, dim = 1), *states)
        actions, rewards, terminateds, truncateds = tuple(torch.stack(t, dim = 1) for t in (actions, rewards, terminateds, truncateds))

        states, actions, rewards, terminateds, truncateds = tree_map_tensor_to_device((states, actions, rewards, terminateds, truncateds), self.device)

        with torch.no_grad():
            state = tree_map_tensor_to_device(state, self.device)

            # treat as a single timestep sequence

            state_seq = tree_map_tensor(lambda t: rearrange(t, 'b ... -> b 1 ...'), state)

            # encode state with ema model for bootstrapping value

            ema_encoded_state, _, _ = self.encode_states(self.ema_state_encoder, state_seq)

            ema_state_tokens = rearrange(ema_encoded_state, 'b 1 ... -> b ...')
            ema_state_latents = self.to_state_latent(ema_state_tokens)

            # get next values

            next_values = self.value_network.forward_ema(ema_state_tokens, ema_state_latents)
            next_values = rearrange(next_values, '... 1 -> ...')

        returns = calc_returns(rewards, terminateds, next_values, self.discount_factor)

        batch_discount = self.value_network.discount_factor.expand(batch)
        returns = (batch_discount, returns)

        actor_log_probs = actor_log_probs if not is_empty(actor_log_probs) else None
        actor_log_probs = maybe(stack)(actor_log_probs, dim = 1)

        experience = (states, actions, actor_log_probs, rewards, terminateds, truncateds, infos, episode_len, cumulative_rewards, returns)

        if return_cpu:
            experience = tree_map_tensor_to_device(experience, 'cpu')

        return Experience(*experience)

    def forward(
        self,
        states: States,
        actions,
        returns = None,
        behavior_clone: bool | str | tuple[str, ...] | list[str] = True,
        return_loss = True,
        memories = None,
        return_memories = False
    ):
        device = self.device

        (batch, state_len), action_len = first_tensor(states).shape[:2], actions.shape[1]
        assert action_len in {state_len, state_len - 1}

        orig_actions = actions

        actions = pad_right_at_dim_to(actions, state_len, dim = 1)

        # handle memories

        state_rnn_memories = action_rnn_memories = model_memories = None

        if exists(memories):
            state_rnn_memories, action_rnn_memories, model_memories = memories

        next_model_memories = dict()

        # generate structured dropout mask

        sensory_layer_masks = None

        if self.training and (self.has_sensory_dropout or self.use_structured_sensory_dropout):
            num_senses = len(self.state_encoder)
            sensory_layer_masks = torch.ones((num_senses, batch, state_len), dtype = torch.bool, device = device)

            if self.has_sensory_dropout:
                keep_sensory = torch.rand((num_senses, batch, 1), device = device) > self.sensory_dropout_prob
                sensory_layer_masks &= keep_sensory

            if self.use_structured_sensory_dropout:
                keep_structured = torch.rand((num_senses, batch, state_len), device = device) < self.structured_sensory_dropout_keep_prob
                sensory_layer_masks &= keep_structured

            sensory_layer_masks = sensory_layer_masks.unbind(dim = 0)

        # encode the states and actions, todo: do mixture of transformers, a la VLAs

        state_tokens, sensory_layer_hiddens, sensory_for_alignment = self.encode_states(self.state_encoder, states, sensory_layer_masks)

        # perceptual gating

        if self.use_perception_film:
            state_tokens = self.perception_film(state_tokens)

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

        # levljepa cross-modal prediction between states and actions

        align_pre_state_action_repr_loss = self.zero
        align_pre_state_action_repr_sigreg_loss = self.zero

        if (
            self.has_align_pre_state_action_repr_loss or
            self.has_align_pre_state_action_repr_sigreg
        ):
            align_next_state_tokens, align_curr_state_tokens, align_action_tokens = rnn_state_tokens[:, 1:], rnn_state_tokens[:, :-1], rnn_action_tokens[:, :-1]

            if self.has_align_pre_state_action_repr_loss:
                pred_action_from_state = self.state_to_action_pred(align_next_state_tokens)
                pred_state_from_action = self.action_to_state_pred(align_action_tokens)

                align_pre_state_action_repr_loss = (
                    F.mse_loss(pred_action_from_state, align_action_tokens) +
                    F.mse_loss(pred_state_from_action, align_curr_state_tokens)
                ) / 2

            if self.has_align_pre_state_action_repr_sigreg:
                align_pre_state_action_repr_sigreg_loss = (
                    self.reg(align_next_state_tokens) +
                    self.reg(align_action_tokens)
                ) / 2

        # cross sensory alignment (vljepa method)

        cross_sensory_align_loss = self.zero
        cross_sensory_align_sigreg_loss = self.zero
        cross_sensory_align_breakdown = tuple()

        if self.has_cross_sensory_align:
            if self.has_cross_sensory_align_sigreg:
                unique_indices = {idx for pair in self.cross_sensory_align_pairs for idx in pair}

                for idx in unique_indices:
                    cross_sensory_align_sigreg_loss = cross_sensory_align_sigreg_loss + self.reg(sensory_for_alignment[idx])

                cross_sensory_align_sigreg_loss = cross_sensory_align_sigreg_loss / len(unique_indices)

            pair_losses = []
            for (src_idx, tgt_idx), predictor in zip(self.cross_sensory_align_pairs, self.cross_sensory_preds):
                src_align_tokens = sensory_for_alignment[src_idx]
                tgt_align_tokens = sensory_for_alignment[tgt_idx]

                pred_tgt = predictor(src_align_tokens)
                pair_loss = F.mse_loss(pred_tgt, tgt_align_tokens)

                cross_sensory_align_loss = cross_sensory_align_loss + pair_loss
                pair_losses.append(CrossSensoryPairLoss(src_idx, tgt_idx, pair_loss))

            cross_sensory_align_loss = cross_sensory_align_loss / len(self.cross_sensory_align_pairs)
            cross_sensory_align_breakdown = tuple(pair_losses)

        # now we interleave the states and actions

        tokens = rearrange([rnn_state_tokens, rnn_action_tokens], 'sa b n d -> b (n sa) d')

        # pass sensory hiddens as attention residual layer hiddens

        wm_past_layers = None
        wm_past_layers_mask = None

        if self.pass_sensory_hiddens_to_world_model:
            wm_past_layers = [rearrange([h, torch.zeros_like(h)], 'sa b n d -> b (n sa) d') for h in sensory_layer_hiddens]

            wm_past_layers_mask = []
            masks = default(sensory_layer_masks, (None,) * len(self.num_sensory_views))

            for views, mask in zip(self.num_sensory_views, masks):
                state_mask = default(mask, torch.ones((batch, state_len), dtype = torch.bool, device = device))
                action_mask = torch.zeros_like(state_mask)

                interleaved_mask = rearrange([state_mask, action_mask], 'sa b n -> b (n sa)')

                for _ in range(views):
                    wm_past_layers_mask.append(interleaved_mask)

        # attention + eventually rnns - yes we need recurrence, i concede that.

        # we will follow Teoh et al's lead and use the post-norm space as the latent

        prepend_memories = None
        if self.has_episodic_mem and not exists(model_memories):
            prepend_memories = self.episodic_mems(batch)

        (embeds, world_model_hiddens), next_model_memories = self.model(
            tokens,
            past_layers = wm_past_layers,
            past_layers_mask = wm_past_layers_mask,
            return_hiddens = True,
            memories = model_memories,
            prepend_memories = prepend_memories,
            return_memories = True
        )

        target_embeds = self.ema_model(
            tokens,
            prepend_memories = prepend_memories
        )

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

        ema_encoded_state, ema_sensory_layer_hiddens, _ = self.encode_states(self.ema_state_encoder, states)

        next_ema_encoded_state_target = ema_encoded_state[:, 1:]

        next_encoded_state_pred_loss = F.smooth_l1_loss(pred_next_encoded_state, next_ema_encoded_state_target.detach())

        # behavior clone

        actor_bc_loss = self.zero
        actor_losses = dict()

        if self.has_actors and behavior_clone:

            if isinstance(behavior_clone, str):
                behavior_clone_actors = [behavior_clone]
            elif isinstance(behavior_clone, (tuple, list)):
                behavior_clone_actors = behavior_clone
            else:
                behavior_clone_actors = list(self.actors.keys())

            bc_seq_len = state_tokens.shape[1] - 1

            for name, actor in self.actors.items():
                if name not in behavior_clone_actors:
                    continue

                loss_weight = self.actor_loss_weights[name]

                if loss_weight == 0.:
                    continue

                actor_bc_loss_for_name = actor(
                    target_actions = orig_actions[:, :bc_seq_len],
                    state_latents = tree_map_detach(state_latents),
                    state_tokens = tree_map_detach(state_tokens),
                    action_cond = tree_map_detach(action_cond),
                    action_tokens = tree_map_detach(action_tokens),
                    sensory_layer_hiddens = tree_map_detach(sensory_layer_hiddens) if exists(sensory_layer_hiddens) else None,
                    world_model_hiddens = tree_map_detach(world_model_hiddens) if exists(world_model_hiddens) else None,
                    memories = model_memories[name] if exists(model_memories) and name in model_memories else None,
                    return_memories = return_memories
                )

                if return_memories:
                    actor_bc_loss_for_name, next_actor_memory = actor_bc_loss_for_name
                    next_model_memories[name] = next_actor_memory

                actor_losses[name] = actor_bc_loss_for_name
                actor_bc_loss = actor_bc_loss + actor_bc_loss_for_name * loss_weight

        # value head

        value_loss = self.zero

        discounts = None
        returns_tensor = None

        if exists(returns):
            if isinstance(returns, tuple):
                discounts, returns_tensor = returns
            else:
                returns_tensor = returns

        pred_values = self.value_network(state_tokens, self.frac_gradient(state_latents_full), discounts)
        pred_values = rearrange(pred_values, '... 1 -> ...')

        if exists(returns_tensor):
            assert pred_values.shape == returns_tensor.shape, f'predicted values shape {pred_values.shape} must match returns shape {returns_tensor.shape}'
            value_loss = F.mse_loss(pred_values, returns_tensor)

        # regularizer loss

        reg_next_state_loss = self.zero
        reg_next_encoded_loss = self.zero

        if self.has_reg_next_state:
            reg_next_state_loss = self.reg(next_state_pred)

        if self.has_reg_next_encoded:
            reg_next_encoded_loss = self.reg(pred_next_encoded_state)

        # action latent uniform loss

        action_latent_wasserstein_loss = self.zero

        if self.has_action_latent_wasserstein_loss:
            action_latent_wasserstein_loss = uniform_wasserstein_loss(action_cond)

        # temporal straightening loss

        temporal_straightening_loss_val = self.zero

        if self.has_temporal_straightening_loss:
            temporal_straightening_loss_val = temporal_straightening_loss(state_latents_full)

        # goal generator loss

        goal_loss = self.zero

        if self.learn_goal_generator and exists(returns_tensor):
            flat_state_latents = rearrange(state_latents_full.detach(), '... d -> (...) d')

            flat_returns = rearrange(returns_tensor, '... -> (...)')

            goal_loss = self.goal_flow_matching(flat_state_latents, returns = flat_returns)

        # intrinsics loss

        intrinsics_loss = self.zero
        intrinsics_breakdown = tuple()

        if self.has_intrinsics:
            flat_state_latents, _ = pack_with_inverse(state_latents_full, '* d')

            ind_losses = []
            for intrinsic, frac_grad_fn in zip(self.intrinsics, self.intrinsic_frac_gradients):
                intrinsic_latents = frac_grad_fn(flat_state_latents)
                ind_losses.append(intrinsic.compute_loss(intrinsic_latents))

            if not is_empty(ind_losses):
                intrinsics_loss = sum(ind_losses)
                intrinsics_breakdown = tuple(ind_losses)

        # quasimetric loss

        quasimetric_distance_loss = self.zero

        if exists(self.quasimetric_distance_network) and ema_encoded_state.shape[1] > 1:
            seq_len = ema_encoded_state.shape[1]

            t1 = torch.randint(0, seq_len, (batch,), device = device)
            t2 = torch.randint(0, seq_len, (batch,), device = device)

            # enforces state (left) to goal (right) encoder asymmetry
            state_left = batched_index_select(state_latents_full, t1, dim = 1)
            state_right = batched_index_select(ema_encoded_state, t2, dim = 1)

            quasimetric_distance_loss = self.quasimetric_distance_network.time_contrastive_loss(
                state_left,
                state_right,
                t1,
                t2
            )

        # losses

        loss_breakdown = Losses(
            next_state_latent_pred_loss,
            plan_state_pred_loss,
            action_recon_loss,
            next_encoded_state_pred_loss,
            actor_bc_loss,
            actor_losses,
            value_loss,
            reg_next_state_loss,
            reg_next_encoded_loss,
            action_latent_wasserstein_loss,
            goal_loss,
            temporal_straightening_loss_val,
            align_pre_state_action_repr_loss,
            align_pre_state_action_repr_sigreg_loss,
            cross_sensory_align_loss,
            cross_sensory_align_sigreg_loss,
            cross_sensory_align_breakdown,
            intrinsics_loss,
            intrinsics_breakdown,
            quasimetric_distance_loss
        )

        total_loss = (
            next_state_latent_pred_loss +
            plan_state_pred_loss * self.plan_state_pred_loss_weight +
            action_recon_loss * self.action_recon_loss_weight +
            next_encoded_state_pred_loss * self.next_encoded_state_pred_loss_weight +
            actor_bc_loss +
            value_loss * self.value_loss_weight +
            reg_next_state_loss * self.reg_next_state_weight +
            reg_next_encoded_loss * self.reg_next_encoded_weight +
            action_latent_wasserstein_loss * self.action_latent_wasserstein_loss_weight +
            goal_loss * self.goal_loss_weight +
            temporal_straightening_loss_val * self.temporal_straightening_loss_weight +
            align_pre_state_action_repr_loss * self.align_pre_state_action_repr_loss_weight +
            align_pre_state_action_repr_sigreg_loss * self.align_pre_state_action_repr_sigreg_weight +
            cross_sensory_align_loss * self.cross_sensory_align_loss_weight +
            cross_sensory_align_sigreg_loss * self.cross_sensory_align_sigreg_weight +
            intrinsics_loss * self.intrinsic_loss_weight +
            quasimetric_distance_loss * self.quasimetric_distance_loss_weight
        )

        out = (total_loss, loss_breakdown)
        return (out, next_memories) if return_memories else out

# test-time training

class TTTModuleWrapper(Module):
    def __init__(self, module: Module):
        super().__init__()
        self.module = module
        self.batch_params = None

    def forward(self, *args, **kwargs):
        if not exists(self.batch_params):
            return self.module(*args, **kwargs)

        def call_single(params, *args_single):
            return functional_call(self.module, params, args_single, kwargs)

        return vmap(call_single)(self.batch_params, *args)

class TTTMetaLearningLoss(Module):
    def __init__(
        self,
        *,
        dim,
        num_classes,
        depth = 1,
        heads = 4,
        dim_head = 64
    ):
        super().__init__()
        self.encoder = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            causal = False
        )

        self.to_prediction = nn.Sequential(
            RMSNorm(dim),
            LinearNoBias(dim, num_classes)
        )
        self.to_meta_prediction = nn.Sequential(
            RMSNorm(dim),
            LinearNoBias(dim, num_classes)
        )

    def forward(self, hiddens, mask = None):
        pred = self.to_prediction(hiddens)

        encoded = self.encoder(hiddens)
        meta_pred = self.to_meta_prediction(encoded)

        prob_earlier = pred.softmax(dim = -1)
        log_prob_encoded = meta_pred.log_softmax(dim = -1)

        loss = F.kl_div(log_prob_encoded, prob_earlier, reduction = 'none').sum(dim = -1)
        return masked_mean(loss, mask)

class TestTimeTrainer(Module):
    def __init__(
        self,
        model: Module,
        ttt_module_paths: tuple[str, ...],
        ttt_loss_module: Module,
        ttt_lr = 1e-3,
        ttt_wd = 0.01
    ):
        super().__init__()
        self.model = model
        self.ttt_module_paths = ttt_module_paths
        self.ttt_lr = ttt_lr
        self.ttt_wd = ttt_wd

        self.ttt_loss_module = ttt_loss_module
        self.ttt_wrappers = dict()

        for path in self.ttt_module_paths:
            if path in self.ttt_wrappers:
                continue

            mod = get_module_by_path(model, path)
            wrapper = TTTModuleWrapper(mod)
            set_module_by_path(model, path, wrapper)
            self.ttt_wrappers[path] = wrapper

    @property
    def has_ttt(self):
        return len(self.ttt_wrappers) > 0

    def init_ttt(self, batch):
        if not self.has_ttt:
            return

        for wrapper in self.ttt_wrappers.values():
            wrapper.batch_params = tree_map_tensor(
                lambda t: repeat(t, '... -> b ...', b = batch),
                dict(wrapper.module.named_parameters())
            )

    def detach_params(self):
        if not self.has_ttt:
            return

        for wrapper in self.ttt_wrappers.values():
            wrapper.batch_params = tree_map_detach(
                wrapper.batch_params,
                preserve_requires_grad = True
            )

    def update(
        self,
        hiddens,
        mask = None,
        create_graph = False
    ):
        if not self.has_ttt:
            return

        loss_t_per_batch = self.ttt_loss_module(hiddens, mask = mask)

        all_batch_params = [
            param
            for path in self.ttt_module_paths
            for param in self.ttt_wrappers[path].batch_params.values()
        ]

        grads = torch.autograd.grad(
            loss_t_per_batch.sum(),
            all_batch_params,
            create_graph = create_graph,
            retain_graph = create_graph,
            allow_unused = True
        )

        # update parameters with gradients

        grads_iter = iter(grads)

        for path in self.ttt_module_paths:
            wrapper = self.ttt_wrappers[path]

            new_batch_params = dict()
            for tgt_name, tgt_param in wrapper.batch_params.items():
                grad = next(grads_iter)
                if exists(grad):
                    grad_update = self.ttt_lr * grad
                    wd_update = self.ttt_lr * self.ttt_wd * tgt_param
                    tgt_param = tgt_param - grad_update - wd_update

                if not create_graph:
                    tgt_param = tgt_param.detach().requires_grad_()

                new_batch_params[tgt_name] = tgt_param

            wrapper.batch_params = new_batch_params

# rollout wrapper for chunking and test-time training

class WorldModelRolloutWrapper(Module):
    def __init__(
        self,
        world_model: Module,
        chunk_size = 128,
        tbptt_steps = 2,
        ttt_module_paths: tuple[str, ...] = tuple(),
        ttt_update_perception_film = False,
        ttt_lr = 1e-3,
        ttt_wd = 0.01,
        ttt_loss_module: Module | None = None
    ):
        super().__init__()
        self.world_model = world_model
        self.chunk_size = chunk_size
        self.tbptt_steps = tbptt_steps

        # append perception film to ttt paths if enabled

        if ttt_update_perception_film:
            assert world_model.use_perception_film, 'WorldModel must have use_perception_film=True to update it via TTT'
            ttt_module_paths = (*ttt_module_paths, 'perception_film')

        self.has_ttt = len(ttt_module_paths) > 0
        self.ttt_trainer = None

        if self.has_ttt:
            assert exists(ttt_loss_module), 'ttt_loss_module must be provided if ttt_module_paths is provided'
            self.ttt_trainer = TestTimeTrainer(
                model = world_model,
                ttt_module_paths = ttt_module_paths,
                ttt_loss_module = ttt_loss_module,
                ttt_lr = ttt_lr,
                ttt_wd = ttt_wd
            )
            assert tbptt_steps > 1, 'tbptt_steps must be greater than 1 if ttt is turned on'

        # hook to capture hiddens during ttt

        self._captured_hiddens = None
        if self.has_ttt:
            def hook(module, args, kwargs, output):
                output_tokens, layer_hiddens = output[0]
                *_, last_layer_hidden = layer_hiddens
                self._captured_hiddens = last_layer_hidden

            self.world_model.model.register_forward_hook(hook, with_kwargs = True)

    def forward(
        self,
        states,
        actions = None,
        returns = None,
        memories = None,
        return_memories = False,
        behavior_clone = True,
        return_loss = True
    ):
        state_tensor = first_tensor(states)
        batch, seq_len = state_tensor.shape[:2]

        # auto-initialize ttt if starting a new sequence

        if self.has_ttt and not exists(memories):
            self.ttt_trainer.init_ttt(batch)

        # iterate over chunks

        num_chunks = math.ceil(seq_len / self.chunk_size)
        total_loss_accum = 0.
        last_loss_breakdown = None
        out_embeds = []

        forward_context = torch.enable_grad if self.has_ttt else nullcontext

        with forward_context():
            for idx in range(num_chunks):
                is_last_chunk = (idx == num_chunks - 1)
                should_detach = divisible_by(idx + 1, self.tbptt_steps)

                # fetch chunk inputs

                start, end = idx * self.chunk_size, (idx + 1) * self.chunk_size

                def get_chunk(t):
                    return t[:, start:end] if is_tensor(t) and t.ndim > 1 else t

                tree_map_get_chunk = maybe(partial(tree_map, get_chunk))

                chunk_states = tree_map(get_chunk, states)
                chunk_actions = tree_map_get_chunk(actions)
                chunk_returns = tree_map_get_chunk(returns)

                chunk_len = first_tensor(chunk_states).shape[1]
                loss_weight = chunk_len / seq_len

                # forward world model

                wm_out = self.world_model(
                    states = chunk_states,
                    actions = chunk_actions,
                    returns = chunk_returns,
                    memories = memories,
                    return_memories = True,
                    behavior_clone = behavior_clone,
                    return_loss = return_loss
                )

                if return_loss:
                    chunk_out, next_memories = wm_out
                    chunk_total_loss, chunk_breakdown = chunk_out
                    total_loss_accum = total_loss_accum + chunk_total_loss * loss_weight
                    last_loss_breakdown = chunk_breakdown
                else:
                    out, next_memories = wm_out
                    out_embeds.append(out['embeds'])

                # ttt update

                skip_ttt_update = is_last_chunk and not return_memories

                if self.has_ttt and not skip_ttt_update and exists(self._captured_hiddens):
                    self.ttt_trainer.update(
                        hiddens = self._captured_hiddens,
                        create_graph = self.training
                    )
                    self._captured_hiddens = None

                memories = next_memories

                if should_detach:
                    if self.has_ttt:
                        self.ttt_trainer.detach_params()

                    memories = tree_map_detach(memories)

        out = (total_loss_accum, last_loss_breakdown) if return_loss else dict(embeds = cat(out_embeds, dim = 1))
        return (out, memories) if return_memories else out
