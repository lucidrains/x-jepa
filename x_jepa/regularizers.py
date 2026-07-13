import torch
import torch.nn.functional as F
from torch.nn import Module

from einops import rearrange, repeat

# helpers

def l2norm(t):
    return F.normalize(t, dim = -1)

# Randall Balestriero et al.  https://arxiv.org/abs/2511.08544

def sigreg_loss(
    x,
    num_slices = 1024,
    domain = (-5, 5),
    num_knots = 17
):
    dim, device = x.shape[-1], x.device

    compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    x = x.to(compute_dtype)

    rand_projs = torch.randn((num_slices, dim), device = device, dtype = compute_dtype)
    rand_projs = l2norm(rand_projs)

    t = torch.linspace(*domain, num_knots, device = device, dtype = compute_dtype)

    exp_f = (-0.5 * t.square()).exp()

    x_t = torch.einsum('... d, m d -> ... m', x, rand_projs)
    x_t = rearrange(x_t, '... m -> (...) m')

    x_t = rearrange(x_t, 'n m -> n m 1') * t
    ecf = (1j * x_t).exp().mean(dim = 0)

    err = ecf.sub(exp_f).abs().square().mul(exp_f)

    return torch.trapezoid(err, t, dim = -1).mean()

# for action latents bounded between -1 and 1

def uniform_wasserstein_loss(x):
    x = rearrange(x, 'b ... d -> (b ...) d')
    batch, dim, device = *x.shape, x.device

    x_sorted, _ = x.sort(dim=0)
    target = torch.linspace(-1., 1., batch, device = device)
    target = repeat(target, 'b -> b d', d = dim)
    return F.mse_loss(x_sorted, target)

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
