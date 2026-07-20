import torch
from torch import nn, cat
from torch.nn import Module
import torch.nn.functional as F

from einops import repeat, rearrange
from torch_einops_utils import pack_with_inverse

from x_jepa.utils import exists, default
from x_jepa.goals import RandomSinusoidalPosEmb
from x_jepa.flow_matching import FlowMatching
from x_jepa.x_jepa import Transformer, RMSNorm

# vector field

class ActionVectorField(Module):
    def __init__(
        self,
        dim,
        dim_state_latent = None,
        depth = 2
    ):
        super().__init__()
        self.time_emb = RandomSinusoidalPosEmb(dim)
        self.dim_state_latent = default(dim_state_latent, 0)

        self.proj_in = nn.Linear(dim * 2 + self.dim_state_latent, dim)

        self.net = Transformer(
            dim = dim,
            depth = depth,
            causal = False,
            use_pope = True
        )
        self.proj_out = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim, bias = False)
        )

    def forward(
        self,
        action,              # b, d | b, n, d
        time,                # b,
        state_latent = None, # b, d | b, n, d
        **kwargs
    ):
        time = self.time_emb(time)

        action, unpack_action = pack_with_inverse([action], 'b * d')
        n = action.shape[-2]

        time = repeat(time, 'b d -> b n d', n = n)

        inputs = (action, time)

        if self.dim_state_latent > 0:
            if not exists(state_latent):
                state_latent = torch.zeros((action.shape[0], n, self.dim_state_latent), device = action.device, dtype = action.dtype)

            state_latent, _ = pack_with_inverse([state_latent], 'b * d')

            if state_latent.shape[-2] == 1 and n > 1:
                state_latent = repeat(state_latent, 'b 1 d -> b n d', n = n)

            inputs = (*inputs, state_latent)

        x = cat(inputs, dim = -1)
        x = self.proj_in(x)
        x = self.net(x)
        x = self.proj_out(x)

        x, = unpack_action(x)

        return x

# latent action model

def LatentActionModel(
    dim,
    dim_state_latent = None,
    depth = 2,
    noise_std = 1.0,
    loss_fn = F.mse_loss
):
    model = ActionVectorField(dim, dim_state_latent, depth = depth)

    return FlowMatching(
        model = model,
        data_shape = None,
        loss_fn = loss_fn,
        noise_std = noise_std
    )
