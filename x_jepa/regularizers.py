import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module

from functools import wraps
from einops import rearrange, repeat, einsum

# helpers

def exists(v):
    return v is not None

def l2norm(t):
    return F.normalize(t, dim = -1)

# decorators

def cast_compute_dtype(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        x = x.to(compute_dtype)
        return fn(x, *args, **kwargs)
    return inner

# for action latents bounded between -1 and 1

@cast_compute_dtype
def uniform_wasserstein_loss(x):
    x = rearrange(x, 'b ... d -> (b ...) d')
    batch, dim, device, dtype = *x.shape, x.device, x.dtype

    x_sorted, _ = x.sort(dim = 0)
    quantiles = (torch.arange(batch, device = device, dtype = dtype) + 0.5) / batch
    target = quantiles * 2. - 1.
    target = repeat(target, 'b -> b d', d = dim)
    return F.mse_loss(x_sorted, target)

# Randall Balestriero et al.  https://arxiv.org/abs/2511.08544

@cast_compute_dtype
def sigreg_loss(
    x,
    num_slices = 1024,
    domain = (-5, 5),
    num_knots = 17
):
    dim, device, dtype = x.shape[-1], x.device, x.dtype

    rand_projs = torch.randn((num_slices, dim), device = device, dtype = dtype)
    rand_projs = l2norm(rand_projs)

    t = torch.linspace(*domain, num_knots, device = device, dtype = dtype)

    exp_f = (-0.5 * t.square()).exp()

    x_t = torch.einsum('... d, m d -> ... m', x, rand_projs)
    x_t = rearrange(x_t, '... m -> (...) m')

    x_t = rearrange(x_t, 'n m -> n m 1') * t
    ecf = (1j * x_t).exp().mean(dim = 0)

    err = ecf.sub(exp_f).abs().square().mul(exp_f)

    return torch.trapezoid(err, t, dim = -1).mean()

# sig reg module

class SigReg(Module):
    def __init__(
        self,
        *,
        num_slices = 1024,
        domain = (-5, 5),
        num_knots = 17
    ):
        super().__init__()
        self.num_slices = num_slices
        self.domain = domain
        self.num_knots = num_knots

    def forward(self, x):
        return sigreg_loss(
            x,
            num_slices = self.num_slices,
            domain = self.domain,
            num_knots = self.num_knots
        )

# Haiyu Wu et al.  https://arxiv.org/abs/2606.02572
# drop-in alternative to sigreg - matches the sorted 1d projections to the
# gaussian quantiles (sliced wasserstein-2 to a standard normal) instead of
# the empirical characteristic function, plus explicit center and scale terms

@cast_compute_dtype
def visreg_loss(
    x,
    num_slices = 1024,
    lambda_center = 1.,
    lambda_scale = 1.,
    lambda_shape = 1.,
    eps = 1e-6
):
    x = rearrange(x, '... d -> (...) d')
    batch, dim, device, dtype = *x.shape, x.device, x.dtype

    # center - penalize non-zero feature mean

    mu = x.mean(dim = 0, keepdim = True)
    center_loss = mu.square().mean()

    # scale - penalize per-feature std away from one

    x_centered = x - mu
    std = x_centered.norm(dim = 0).div(math.sqrt(batch)) + eps
    scale_loss = (std - 1.).square().mean()

    # shape - sliced wasserstein-2 to the standard normal

    x_norm = x_centered / std.detach()

    rand_projs = l2norm(torch.randn((num_slices, dim), device = device, dtype = dtype))
    projected = torch.einsum('n d, m d -> n m', x_norm, rand_projs)
    projected_sorted, _ = projected.sort(dim = 0)

    quantiles = torch.linspace(1, batch, batch, device = device, dtype = dtype) / (batch + 1)
    target = torch.erfinv(2. * quantiles - 1.).mul(math.sqrt(2.))
    target = rearrange(target, 'n -> n 1')

    shape_loss = (projected_sorted - target).square().mean()

    return (
        center_loss * lambda_center +
        scale_loss * lambda_scale +
        shape_loss * lambda_shape
    )

# visreg module

class VISReg(Module):
    def __init__(
        self,
        *,
        num_slices = 1024,
        lambda_center = 1.,
        lambda_scale = 1.,
        lambda_shape = 1.
    ):
        super().__init__()
        self.num_slices = num_slices
        self.lambda_center = lambda_center
        self.lambda_scale = lambda_scale
        self.lambda_shape = lambda_shape

    def forward(self, x):
        return visreg_loss(
            x,
            num_slices = self.num_slices,
            lambda_center = self.lambda_center,
            lambda_scale = self.lambda_scale,
            lambda_shape = self.lambda_shape
        )

# Ying Wang et al. https://arxiv.org/abs/2603.12231

@cast_compute_dtype
def temporal_straightening_loss(
    latents,
    eps = 1e-6
):
    *_, seq_len, __ = latents.shape

    if seq_len <= 2:
        return latents.new_zeros(())

    velocities = torch.diff(latents, dim = -2)

    past_vel, future_vel = velocities[..., :-1, :], velocities[..., 1:, :]

    cos_sim = F.cosine_similarity(past_vel, future_vel, dim = -1, eps = eps)
    return (1. - cos_sim).mean()

class TemporalStraightening(Module):
    def __init__(
        self,
        eps = 1e-6
    ):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        latents
    ):
        return temporal_straightening_loss(latents, eps = self.eps)

# jepa-anything https://arxiv.org/abs/2609.20800

@cast_compute_dtype
def coordinate_std_floor_loss(samples, min_std = 0.1, eps = 1e-6):
    samples = rearrange(samples, '... d -> (...) d')
    std = (samples.var(dim = 0, unbiased = False) + eps).sqrt()
    return F.relu(min_std - std).mean()

def factor_activity_loss(factors, min_std = 0.1, eps = 1e-6):
    return coordinate_std_floor_loss(rearrange(factors, '... k r -> (...) (k r)'), min_std = min_std, eps = eps)

def encoder_variance_loss(states, min_std = 0.1, eps = 1e-6):
    return coordinate_std_floor_loss(states, min_std = min_std, eps = eps)

# orthogonal subspaces module

class OrthogonalSubspaces(Module):
    """
    Orthogonal Subspaces Projection from JEPA-Anything (arXiv:2609.20800).
    Decomposes latent state of dimension d into K orthogonal subspaces of dimension r (d = K * r).
    """
    def __init__(
        self,
        dim,
        num_subspaces,
        use_pinv = True
    ):
        super().__init__()
        assert dim % num_subspaces == 0, f'dim {dim} must be divisible by num_subspaces {num_subspaces}'

        self.dim = dim
        self.num_subspaces = num_subspaces
        self.use_pinv = use_pinv

        self.basis = nn.Parameter(rearrange(torch.eye(dim), '(k r) d -> k r d', k = num_subspaces))

        self._cached_synthesis = None

    def train(self, mode = True):
        super().train(mode)
        self._cached_synthesis = None
        return self

    @property
    def synthesis_basis(self):
        # transpose synthesis is exact while the basis is orthonormal, the pseudoinverse stays exact as it drifts

        if not self.use_pinv:
            return self.basis

        if not self.training and exists(self._cached_synthesis):
            return self._cached_synthesis

        flat_basis = rearrange(self.basis, 'k r d -> (k r) d')
        synthesis = rearrange(torch.linalg.pinv(flat_basis).T, '(k r) d -> k r d', k = self.num_subspaces)

        if not self.training:
            self._cached_synthesis = synthesis

        return synthesis

    def project(self, state):
        return einsum(state, self.basis, '... d, k r d -> ... k r')

    def compose(self, factors):
        return einsum(factors, self.synthesis_basis, '... k r, k r d -> ... d')

    def reconstruct(self, state):
        return self.compose(self.project(state))

    def residual_factors(self, state, residual):
        residual = rearrange(residual, '... (k r) -> ... k r', k = self.num_subspaces)
        return self.project(state) + residual

    def add_residual(self, state, residual):
        return self.compose(self.residual_factors(state, residual))

    def orthogonal_loss(self):
        flat_basis = rearrange(self.basis, 'k r d -> (k r) d')
        identity = torch.eye(self.dim, device = self.basis.device, dtype = self.basis.dtype)
        return (flat_basis @ flat_basis.T - identity).square().sum()

    def forward(self, state):
        return self.project(state)
