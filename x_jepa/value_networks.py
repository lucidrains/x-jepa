from __future__ import annotations

import math
from functools import partial

import einx
import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from ema_pytorch import EMA
from torch import Tensor, atleast_1d, cat, is_tensor, tensor
from torch.nn import Linear, Module, ModuleList, Sequential, SiLU

from x_mlps_pytorch import MLP

from x_jepa.flow_matching import FlowMatching
from x_jepa.utils import exists, default, masked_mean

# constants

LinearNoBias = partial(Linear, bias = False)

LOG_2_PI = math.log(2. * math.pi)

# transition_risk: aleatoric dynamics entropy | value_risk: epistemic value uncertainty
# all heads emit (..., num_statistics) value statistics, squeezed when squeeze_out = True

# helpers

def cast_tensor(t):
    return t if is_tensor(t) else tensor(t)

def log(x, eps = 1e-20):
    return torch.log(x.clamp(min = eps))

def l1norm(t):
    return F.normalize(t, p = 1., dim = -1)

# discount embedder

class DiscountEmbedder(Module):
    def __init__(
        self,
        dim,
        eps = 1e-20
    ):
        super().__init__()
        self.eps = eps
        self.net = Sequential(
            LinearNoBias(3, dim),
            SiLU(),
            LinearNoBias(dim, dim)
        )

        self.register_buffer('zero', tensor(0.), persistent = False)

    @property
    def device(self):
        return self.zero.device

    def forward(self, discount):
        device = self.device
        discount = cast_tensor(discount).to(device)

        one_minus_discount = 1. - discount
        negative_log_discount = -log(1. - discount, eps = self.eps)

        conditioning = torch.stack((discount, one_minus_discount, negative_log_discount), dim = -1)
        return self.net(conditioning)

# transition risk embedder (aleatoric / dynamics risk conditioning)

class TransitionRiskEmbedder(Module):
    def __init__(
        self,
        dim,
        num_bins = 16,
        min_risk = -2.0,
        max_risk = 1.0,
        sigma = 0.2
    ):
        super().__init__()
        self.sigma = sigma
        self.register_buffer('bin_centers', torch.linspace(min_risk, max_risk, num_bins), persistent = False)

        self.transition_risk_embedder = None

        self.net = Sequential(
            LinearNoBias(num_bins, dim),
            SiLU(),
            LinearNoBias(dim, dim)
        )

    def forward(self, risk):
        risk = rearrange(cast_tensor(risk), '... -> ... 1')

        bin_distance = risk - self.bin_centers
        bin_weights = l1norm((-0.5 * (bin_distance / self.sigma) ** 2).exp())

        return self.net(bin_weights)

# value risk embedder (epistemic uncertainty conditioning)

class ValueRiskEmbedder(TransitionRiskEmbedder):
    def __init__(
        self,
        dim,
        num_bins = 16,
        min_risk = 0.0,
        max_risk = 2.0,
        sigma = 0.1
    ):
        super().__init__(dim, num_bins = num_bins, min_risk = min_risk, max_risk = max_risk, sigma = sigma)

# combined risk embedder - independently embeds transition risk and value risk

class RiskEmbedder(Module):
    def __init__(
        self,
        dim,
        transition_risk_embedder: TransitionRiskEmbedder | Module | None = None,
        value_risk_embedder: ValueRiskEmbedder | Module | None = None,
        **kwargs
    ):
        super().__init__()
        self.transition_risk_embedder = default(transition_risk_embedder, TransitionRiskEmbedder(dim, **kwargs))
        self.value_risk_embedder = default(value_risk_embedder, ValueRiskEmbedder(dim, **kwargs))

    def embed_transition_risk(self, risk):
        return self.transition_risk_embedder(risk)

    def embed_value_risk(self, risk):
        return self.value_risk_embedder(risk)

    def forward(self, transition_risk, value_risk):
        transition_embed = 0. if not exists(transition_risk) else self.embed_transition_risk(transition_risk)
        value_embed = 0. if not exists(value_risk) else self.embed_value_risk(value_risk)
        return transition_embed + value_embed

# value head base - shared discount / transition-risk conditioning, loss and uncertainty
# interface, EMA bookkeeping

class ValueHead(Module):
    def __init__(
        self,
        dim,
        dim_state_latent,
        discount_factor = 0.99,
        transition_risk_factor: float | Tensor | None = None,
        value_risk_factor: float | Tensor | None = None,
        ema_beta = 0.999,
        discount_embedder: DiscountEmbedder | Module | None = None,
        transition_risk_embedder: TransitionRiskEmbedder | Module | None = None,
        value_risk_embedder: ValueRiskEmbedder | Module | None = None,
        risk_embedder: RiskEmbedder | Module | None = None, # combined embedder - both risks taken into account separately
        eps = 1e-20
    ):
        super().__init__()
        self.eps = eps
        self.ema_beta = ema_beta
        self.discount_factor = discount_factor
        self.transition_risk_factor = transition_risk_factor
        self.value_risk_factor = value_risk_factor

        self.discount_embedder = default(discount_embedder, DiscountEmbedder(dim, eps = eps))
        self.ema_discount_embedder = EMA(self.discount_embedder, beta = ema_beta)

        if exists(risk_embedder):
            if isinstance(risk_embedder, RiskEmbedder):
                transition_risk_embedder = default(transition_risk_embedder, risk_embedder.transition_risk_embedder)
                value_risk_embedder = default(value_risk_embedder, risk_embedder.value_risk_embedder)
            else:
                transition_risk_embedder = default(transition_risk_embedder, risk_embedder)

        self.transition_risk_embedder = default(transition_risk_embedder, TransitionRiskEmbedder(dim) if exists(transition_risk_factor) else None)
        self.value_risk_embedder = default(value_risk_embedder, ValueRiskEmbedder(dim) if exists(value_risk_factor) else None)

        if exists(self.transition_risk_embedder):
            self.ema_transition_risk_embedder = EMA(self.transition_risk_embedder, beta = ema_beta)
        else:
            self.ema_transition_risk_embedder = None

        if exists(self.value_risk_embedder):
            self.ema_value_risk_embedder = EMA(self.value_risk_embedder, beta = ema_beta)
        else:
            self.ema_value_risk_embedder = None

    def embed_discount(self, discount, ema = False):
        discount_embedder = self.ema_discount_embedder if ema else self.discount_embedder
        return discount_embedder(discount)

    def get_discount_embed(self, target_tensor, discount = None, ema = False):
        discount = default(discount, self.discount_factor)

        if not exists(discount):
            return 0.

        discount = atleast_1d(cast_tensor(discount))
        discount = rearrange(discount, 'batch -> batch 1')

        if target_tensor.ndim == 3:
            seq_len = target_tensor.shape[1]
            discount = repeat(discount, 'batch 1 -> batch seq_len', seq_len = seq_len)
        else:
            discount = rearrange(discount, 'batch 1 -> batch')

        return self.embed_discount(discount, ema = ema)

    def embed_transition_risk(self, transition_risk, ema = False):
        if not exists(self.transition_risk_embedder):
            return 0.

        transition_risk_embedder = self.ema_transition_risk_embedder if ema else self.transition_risk_embedder
        return transition_risk_embedder(transition_risk)

    def get_transition_risk_embed(self, target_tensor, transition_risk = None, ema = False):
        if not exists(self.transition_risk_embedder):
            return 0.

        transition_risk = default(transition_risk, self.transition_risk_factor)

        if not exists(transition_risk):
            return 0.

        transition_risk = atleast_1d(cast_tensor(transition_risk))
        transition_risk = rearrange(transition_risk, 'batch -> batch 1')

        if target_tensor.ndim == 3:
            seq_len = target_tensor.shape[1]
            transition_risk = repeat(transition_risk, 'batch 1 -> batch seq_len', seq_len = seq_len)
        else:
            transition_risk = rearrange(transition_risk, 'batch 1 -> batch')

        return self.embed_transition_risk(transition_risk, ema = ema)

    def embed_value_risk(self, value_risk, ema = False):
        if not exists(self.value_risk_embedder):
            return 0.

        value_risk_embedder = self.ema_value_risk_embedder if ema else self.value_risk_embedder
        return value_risk_embedder(value_risk)

    def get_value_risk_embed(self, target_tensor, value_risk = None, ema = False):
        if not exists(self.value_risk_embedder):
            return 0.

        value_risk = default(value_risk, self.value_risk_factor)

        if not exists(value_risk):
            return 0.

        value_risk = atleast_1d(cast_tensor(value_risk))
        value_risk = rearrange(value_risk, 'batch -> batch 1')

        if target_tensor.ndim == 3:
            seq_len = target_tensor.shape[1]
            value_risk = repeat(value_risk, 'batch 1 -> batch seq_len', seq_len = seq_len)
        else:
            value_risk = rearrange(value_risk, 'batch 1 -> batch')

        return self.embed_value_risk(value_risk, ema = ema)

    def condition_tokens(self, state_tokens, discount, transition_risk, value_risk = None, ema = False):
        discount_embed = self.get_discount_embed(state_tokens, discount, ema = ema)
        transition_risk_embed = self.get_transition_risk_embed(state_tokens, transition_risk, ema = ema)
        value_risk_embed = self.get_value_risk_embed(state_tokens, value_risk, ema = ema)
        return state_tokens + discount_embed + transition_risk_embed + value_risk_embed

    def squeeze_output(self, value_statistics, squeeze_out = True):
        if squeeze_out and value_statistics.shape[-1] == 1:
            return rearrange(value_statistics, '... 1 -> ...')

        return value_statistics

    def update(self):
        self.ema_discount_embedder.update()

        if exists(self.ema_transition_risk_embedder):
            self.ema_transition_risk_embedder.update()

        if exists(self.ema_value_risk_embedder):
            self.ema_value_risk_embedder.update()

    # interface:
    #   compute_loss(value_statistics, returns, cond, mask, average_mode) -> scalar
    #     returns broadcast to the statistic dimension when it is missing
    #   mean_and_uncertainty(value_statistics) -> (mean, std | None)

    def compute_loss(
        self,
        pred_values,
        returns,
        cond = None,
        mask = None,
        average_mode = 'timestep'
    ):
        if pred_values.ndim > returns.ndim:
            returns = repeat(returns, '... -> ... statistic', statistic = pred_values.shape[-1])

        assert pred_values.shape == returns.shape, f'predicted values shape {pred_values.shape} must match returns shape {returns.shape}'

        raw_value_loss = F.smooth_l1_loss(pred_values, returns, reduction = 'none')
        return masked_mean(raw_value_loss, mask, average_mode = average_mode)

    def mean_and_uncertainty(self, values):
        return values, None

    def uncertainty(self, values):
        return None

    def forward_ema(self, state_tokens, state_latents, discount = None, transition_risk = None, value_risk = None, squeeze_out = True):
        raise NotImplementedError

    def forward(self, state_tokens, state_latents, discount = None, transition_risk = None, value_risk = None, squeeze_out = True):
        raise NotImplementedError

# scalar value network - classic critic (risk-neutral)

class Value(ValueHead):
    def __init__(
        self,
        dim,
        dim_state_latent,
        **kwargs
    ):
        super().__init__(dim, dim_state_latent, **kwargs)

        self.net = MLP(dim_state_latent + dim, *((dim * 2,) * 2), 1)
        self.ema_net = EMA(self.net, beta = self.ema_beta)

    def forward_ema(self, state_tokens, state_latents, discount = None, transition_risk = None, value_risk = None, squeeze_out = True):
        state_tokens = self.condition_tokens(state_tokens, discount, transition_risk, value_risk, ema = True)
        value_statistics = self.ema_net((state_tokens, state_latents))
        return self.squeeze_output(value_statistics, squeeze_out)

    def update(self):
        super().update()
        self.ema_net.update()

    def forward(self, state_tokens, state_latents, discount = None, transition_risk = None, value_risk = None, squeeze_out = True):
        state_tokens = self.condition_tokens(state_tokens, discount, transition_risk, value_risk, ema = False)
        value_statistics = self.net((state_tokens, state_latents))
        return self.squeeze_output(value_statistics, squeeze_out)

# ensemble value network - bootstrapped heads (Osband et al. 2016)
# value_risk from head disagreement across bernoulli-masked subsets

class EnsembleValue(ValueHead):
    def __init__(
        self,
        dim,
        dim_state_latent,
        num_heads = 5,
        bootstrap_frac = 0.8,
        **kwargs
    ):
        super().__init__(dim, dim_state_latent, **kwargs)

        self.num_heads = num_heads
        self.bootstrap_frac = bootstrap_frac

        self.nets = ModuleList([
            MLP(dim_state_latent + dim, *((dim * 2,) * 2), 1)
            for _ in range(num_heads)
        ])

        self.ema_nets = ModuleList([
            EMA(net, beta = self.ema_beta) for net in self.nets
        ])

        self._bootstrap_masks = None

    def sample_bootstrap_masks(self, value_statistics):
        if not self.training or self.bootstrap_frac >= 1.0:
            self._bootstrap_masks = None
            return

        device = value_statistics.device
        mask_shape = (*value_statistics.shape[:-1], self.num_heads)
        self._bootstrap_masks = torch.rand(mask_shape, device = device) < self.bootstrap_frac

    def forward_ema(self, state_tokens, state_latents, discount = None, transition_risk = None, value_risk = None, squeeze_out = True):
        state_tokens = self.condition_tokens(state_tokens, discount, transition_risk, value_risk, ema = True)
        head_values = [ema_net((state_tokens, state_latents)) for ema_net in self.ema_nets]
        value_statistics = rearrange(torch.stack(head_values), 'heads ... 1 -> ... heads')
        return self.squeeze_output(value_statistics, squeeze_out)

    def update(self):
        super().update()

        for ema_net in self.ema_nets:
            ema_net.update()

    def forward(self, state_tokens, state_latents, discount = None, transition_risk = None, value_risk = None, squeeze_out = True):
        state_tokens = self.condition_tokens(state_tokens, discount, transition_risk, value_risk, ema = False)
        head_values = [head((state_tokens, state_latents)) for head in self.nets]
        value_statistics = rearrange(torch.stack(head_values), 'heads ... 1 -> ... heads')

        self.sample_bootstrap_masks(value_statistics)

        return self.squeeze_output(value_statistics, squeeze_out)

    def compute_loss(
        self,
        pred_values,
        returns,
        cond = None,
        mask = None,
        average_mode = 'timestep'
    ):
        if pred_values.ndim > returns.ndim:
            returns = repeat(returns, '... -> ... head', head = self.num_heads)

        assert pred_values.shape == returns.shape, f'predicted values shape {pred_values.shape} must match returns shape {returns.shape}'

        raw_value_loss = F.smooth_l1_loss(pred_values, returns, reduction = 'none')

        if exists(self._bootstrap_masks):
            raw_value_loss = raw_value_loss * self._bootstrap_masks

        return masked_mean(raw_value_loss, mask, average_mode = average_mode)

    def mean_and_uncertainty(self, values):
        return reduce(values, '... head -> ...', 'mean'), einx.std('... head -> ...', values)

    def uncertainty(self, values):
        return self.mean_and_uncertainty(values)[1]

# distributional value network - quantile regression critic (QR-DQN, Dabney et al. 2018);
# value_risk from quantile dispersion; value_at_risk (CVaR) available for the lower tail

class DistributionalValue(ValueHead):
    def __init__(
        self,
        dim,
        dim_state_latent,
        num_quantiles = 32,
        **kwargs
    ):
        super().__init__(dim, dim_state_latent, **kwargs)

        self.num_quantiles = num_quantiles

        self.net = MLP(dim_state_latent + dim, *((dim * 2,) * 2), num_quantiles)
        self.ema_net = EMA(self.net, beta = self.ema_beta)

        quantile_fracs = (torch.arange(num_quantiles, dtype = torch.float32) + 0.5) / num_quantiles
        self.register_buffer('quantile_fracs', quantile_fracs, persistent = False)

    def forward_ema(self, state_tokens, state_latents, discount = None, transition_risk = None, value_risk = None, squeeze_out = True):
        state_tokens = self.condition_tokens(state_tokens, discount, transition_risk, value_risk, ema = True)
        value_statistics = self.ema_net((state_tokens, state_latents))
        return self.squeeze_output(value_statistics, squeeze_out)

    def update(self):
        super().update()
        self.ema_net.update()

    def forward(self, state_tokens, state_latents, discount = None, transition_risk = None, value_risk = None, squeeze_out = True):
        state_tokens = self.condition_tokens(state_tokens, discount, transition_risk, value_risk, ema = False)
        value_statistics = self.net((state_tokens, state_latents))
        return self.squeeze_output(value_statistics, squeeze_out)

    def compute_loss(
        self,
        pred_values,
        returns,
        cond = None,
        mask = None,
        average_mode = 'timestep'
    ):
        if pred_values.ndim > returns.ndim:
            returns = repeat(returns, '... -> ... quantile', quantile = self.num_quantiles)

        assert pred_values.shape == returns.shape, f'predicted values shape {pred_values.shape} must match returns shape {returns.shape}'

        quantile_error = returns - pred_values

        huber_loss = F.smooth_l1_loss(pred_values, returns, reduction = 'none', beta = 1.0)
        quantile_weights = (self.quantile_fracs - (quantile_error < 0.).float()).abs()

        raw_value_loss = quantile_weights * huber_loss

        return masked_mean(raw_value_loss, mask, average_mode = average_mode)

    def mean_and_uncertainty(self, values):
        return reduce(values, '... quantile -> ...', 'mean'), einx.std('... quantile -> ...', values)

    def uncertainty(self, values):
        return self.mean_and_uncertainty(values)[1]

    def value_at_risk(self, values, alpha = 0.1):
        num_tail_quantiles = max(int(alpha * self.num_quantiles), 1)
        return reduce(values[..., :num_tail_quantiles], '... quantile -> ...', 'mean')

# mean-variance value network - heteroscedastic Gaussian critic (Kendall & Gal 2017);
# value_risk from the predicted variance

class MeanVarianceValue(ValueHead):
    def __init__(
        self,
        dim,
        dim_state_latent,
        min_logvar = -10.0,
        max_logvar = 10.0,
        **kwargs
    ):
        super().__init__(dim, dim_state_latent, **kwargs)

        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

        self.net = MLP(dim_state_latent + dim, *((dim * 2,) * 2), 2)
        self.ema_net = EMA(self.net, beta = self.ema_beta)

    def clamp_logvar(self, logvar):
        return logvar.clamp(self.min_logvar, self.max_logvar)

    def forward_ema(self, state_tokens, state_latents, discount = None, transition_risk = None, value_risk = None, squeeze_out = True):
        state_tokens = self.condition_tokens(state_tokens, discount, transition_risk, value_risk, ema = True)
        value_statistics = self.ema_net((state_tokens, state_latents))

        # bootstrap only the mean statistic - the variance describes the estimate, not a return

        value_statistics = value_statistics[..., 0:1]

        return self.squeeze_output(value_statistics, squeeze_out)

    def update(self):
        super().update()
        self.ema_net.update()

    def forward(self, state_tokens, state_latents, discount = None, transition_risk = None, value_risk = None, squeeze_out = True):
        state_tokens = self.condition_tokens(state_tokens, discount, transition_risk, value_risk, ema = False)
        value_statistics = self.net((state_tokens, state_latents))
        return self.squeeze_output(value_statistics, squeeze_out)

    def compute_loss(
        self,
        pred_values,
        returns,
        cond = None,
        mask = None,
        average_mode = 'timestep'
    ):
        mean, logvar = pred_values[..., 0], pred_values[..., 1]

        assert mean.shape == returns.shape, f'predicted values shape {pred_values.shape} must match returns shape {returns.shape}'

        logvar = self.clamp_logvar(logvar)
        variance = logvar.exp()

        raw_value_loss = 0.5 * ((returns - mean) ** 2 / variance + logvar + LOG_2_PI)

        return masked_mean(raw_value_loss, mask, average_mode = average_mode)

    def mean_and_uncertainty(self, values):
        mean, logvar = values[..., 0], values[..., 1]
        return mean, (self.clamp_logvar(logvar) / 2).exp()

    def uncertainty(self, values):
        return self.mean_and_uncertainty(values)[1]

# flow-matching value network - conditional flow matching generative return distribution
# value_risk from sample dispersion

class FlowNet(Module):
    def __init__(
        self,
        cond_dim,
        dim,
        dim_state_latent,
        out_dim = 1
    ):
        super().__init__()
        self.net = MLP(dim_state_latent + cond_dim + 2, *((dim * 2,) * 2), out_dim)

    def forward(self, noised_returns, time, cond):
        time = rearrange(time, 'batch -> batch 1')
        return self.net((noised_returns, time, cond))

class FlowValue(ValueHead):
    def __init__(
        self,
        dim,
        dim_state_latent,
        num_steps = 8,
        num_samples = 16,
        noise_std = 1.0,
        **kwargs
    ):
        super().__init__(dim, dim_state_latent, **kwargs)

        self.num_steps = num_steps
        self.num_samples = num_samples

        self.net = FlowNet(dim, dim, dim_state_latent)
        self.ema_net = EMA(self.net, beta = self.ema_beta)

        self.flow = FlowMatching(self.net, data_shape = (1,), noise_std = noise_std)
        self.ema_flow = FlowMatching(self.ema_net, data_shape = (1,), noise_std = noise_std)

    def condition(self, state_tokens, state_latents):
        return cat((state_tokens, state_latents), dim = -1)

    def sample_returns(
        self,
        flow_matching,
        conditioning,
        num_steps = None,
        num_samples = None
    ):
        num_steps = default(num_steps, self.num_steps)
        num_samples = default(num_samples, self.num_samples)

        flat_conditioning = rearrange(conditioning, '... feature -> (...) feature')
        batched_conditioning = repeat(flat_conditioning, 'total feature -> (total sample) feature', sample = num_samples)

        return_samples = flow_matching.sample(
            steps = num_steps,
            batch_size = batched_conditioning.shape[0],
            data_shape = (1,),
            cond = batched_conditioning
        )

        return self.unflatten_samples(return_samples, conditioning, num_samples)

    def unflatten_samples(self, return_samples, reference_tensor, num_samples):
        return_samples = rearrange(return_samples, '(batch sample) 1 -> batch sample', sample = num_samples)

        leading_shape = reference_tensor.shape[:-1]

        if len(leading_shape) == 2:
            batch, seq_len = leading_shape
            return rearrange(return_samples, '(batch seq_len) sample -> batch seq_len sample', batch = batch, seq_len = seq_len)

        batch, population, horizon = leading_shape
        return rearrange(return_samples, '(batch population horizon) sample -> batch population horizon sample', batch = batch, population = population, horizon = horizon)

    def forward_ema(self, state_tokens, state_latents, discount = None, transition_risk = None, value_risk = None, squeeze_out = True):
        state_tokens = self.condition_tokens(state_tokens, discount, transition_risk, value_risk, ema = True)
        return_samples = self.sample_returns(self.ema_flow, self.condition(state_tokens, state_latents))
        return self.squeeze_output(return_samples, squeeze_out)

    def update(self):
        super().update()
        self.ema_net.update()

    def forward(self, state_tokens, state_latents, discount = None, transition_risk = None, value_risk = None, squeeze_out = True):
        state_tokens = self.condition_tokens(state_tokens, discount, transition_risk, value_risk, ema = False)
        return_samples = self.sample_returns(self.flow, self.condition(state_tokens, state_latents))
        return self.squeeze_output(return_samples, squeeze_out)

    def compute_loss(
        self,
        pred_values,
        returns,
        cond = None,
        mask = None,
        average_mode = 'timestep'
    ):
        assert exists(cond), 'cond = (state_tokens, state_latents) is required for the flow value loss'

        state_tokens, state_latents = cond
        flat_conditioning = rearrange(self.condition(state_tokens, state_latents), '... feature -> (...) feature')

        flat_returns = rearrange(returns, '... -> (...) 1')
        flat_mask = rearrange(mask, '... -> (...)') if exists(mask) else None

        raw_flow_loss = self.flow(flat_returns, cond = flat_conditioning, loss_reduction = 'none')

        return masked_mean(raw_flow_loss, flat_mask, average_mode = average_mode)

    def mean_and_uncertainty(self, values):
        return reduce(values, '... sample -> ...', 'mean'), einx.std('... sample -> ...', values)

    def uncertainty(self, values):
        return self.mean_and_uncertainty(values)[1]

# factory

def create_value_network(
    value_network_type = 'value',
    dim = None,
    dim_state_latent = None,
    num_heads = 5,
    num_quantiles = 32,
    num_steps = 8,
    num_samples = 16,
    bootstrap_frac = 0.8,
    **kwargs
):
    assert exists(dim) and exists(dim_state_latent), 'dim and dim_state_latent must be provided to create_value_network'

    if value_network_type in ('value', 'scalar'):
        return Value(dim = dim, dim_state_latent = dim_state_latent, **kwargs)

    if value_network_type == 'ensemble':
        return EnsembleValue(dim = dim, dim_state_latent = dim_state_latent, num_heads = num_heads, bootstrap_frac = bootstrap_frac, **kwargs)

    if value_network_type == 'distributional':
        return DistributionalValue(dim = dim, dim_state_latent = dim_state_latent, num_quantiles = num_quantiles, **kwargs)

    if value_network_type == 'mean_variance':
        return MeanVarianceValue(dim = dim, dim_state_latent = dim_state_latent, **kwargs)

    if value_network_type in ('flow', 'flow_matching'):
        return FlowValue(dim = dim, dim_state_latent = dim_state_latent, num_steps = num_steps, num_samples = num_samples, **kwargs)

    raise ValueError(f'invalid value_network_type: {value_network_type}')
