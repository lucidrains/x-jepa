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

# helpers

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    if prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

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
        self.has_state_latent = self.dim_state_latent > 0

        if self.has_state_latent:
            self.null_state_latent = nn.Parameter(torch.zeros(self.dim_state_latent))

        self.proj_in = nn.Linear(dim, dim)

        self.net = Transformer(
            dim = dim,
            depth = depth,
            causal = False,
            use_pope = True,
            dim_cond = dim + self.dim_state_latent
        )
        self.proj_out = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim, bias = False)
        )

    def forward(
        self,
        action,              # (b d) | (b n d)
        time,                # (b)
        state_latent = None, # (b d) | (b n d)
        cond_drop_prob = 0.,
        **kwargs
    ):
        has_cond_drop = cond_drop_prob > 0.

        time = self.time_emb(time)

        action, unpack_action = pack_with_inverse([action], 'b * d')
        n = action.shape[-2]

        time = repeat(time, 'b d -> b n d', n = n)

        cond_inputs = [time]

        if self.has_state_latent:
            batch = action.shape[0]

            if not exists(state_latent):
                state_latent = self.null_state_latent
                state_latent = repeat(state_latent, 'd -> b 1 d', b = batch)
            else:
                state_latent, _ = pack_with_inverse([state_latent], 'b * d')

                if has_cond_drop:
                    keep_mask = prob_mask_like((batch,), 1. - cond_drop_prob, device = action.device)
                    keep_mask = rearrange(keep_mask, 'b -> b 1 1')

                    null_state_latent = rearrange(self.null_state_latent, 'd -> 1 1 d')

                    state_latent = torch.where(
                        keep_mask,
                        state_latent,
                        null_state_latent
                    )

            state_seq_len = state_latent.shape[-2]

            if state_seq_len == 1 and n > 1:
                state_latent = repeat(state_latent, 'b 1 d -> b n d', n = n)

            cond_inputs.append(state_latent)

        cond = cat(cond_inputs, dim = -1)

        x = self.proj_in(action)
        x = self.net(x, cond = cond)
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
