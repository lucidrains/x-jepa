import math

import torch
import torch.nn.functional as F
from torch.nn import Module

from functools import wraps
from einops import rearrange, repeat

# helpers

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

    x_sorted, _ = x.sort(dim=0)
    target = torch.linspace(-1., 1., batch, device = device, dtype = dtype)
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
