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
        depth = 2,
        prepend_state_latent = False,
        causal_mask = False
    ):
        super().__init__()
        self.time_emb = RandomSinusoidalPosEmb(dim)
        self.dim_state_latent = default(dim_state_latent, 0)
        self.has_state_latent = self.dim_state_latent > 0
        self.prepend_state_latent = prepend_state_latent

        if self.has_state_latent:
            self.null_state_latent = nn.Parameter(torch.zeros(self.dim_state_latent))

            if self.prepend_state_latent:
                self.state_latent_proj = nn.Linear(self.dim_state_latent, dim)

        self.proj_in = nn.Linear(dim, dim)

        dim_cond = dim if self.prepend_state_latent else (dim + self.dim_state_latent)

        self.net = Transformer(
            dim = dim,
            depth = depth,
            causal = causal_mask,
            use_pope = True,
            dim_cond = dim_cond
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
        prepend_state_latent = self.prepend_state_latent

        time = self.time_emb(time)

        action, unpack_action = pack_with_inverse([action], 'b * d')
        batch, seq_len = action.shape[0], action.shape[-2]

        if prepend_state_latent:
            seq_len += 1

        time = repeat(time, 'b d -> b n d', n = seq_len)

        cond_inputs = [time]
        state_latent_tokens = None

        if self.has_state_latent:
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

            if prepend_state_latent:
                state_latent = state_latent[:, :1]
                state_latent_tokens = self.state_latent_proj(state_latent)
            else:
                state_seq_len = state_latent.shape[-2]

                if state_seq_len == 1 and seq_len > 1:
                    state_latent = repeat(state_latent, 'b 1 d -> b n d', n = seq_len)

                cond_inputs.append(state_latent)

        cond = cat(cond_inputs, dim = -1)

        x = self.proj_in(action)

        if prepend_state_latent and exists(state_latent_tokens):
            x, unpack_prepend = pack_with_inverse([state_latent_tokens, x], 'b * d')

        x = self.net(x, cond = cond)

        if prepend_state_latent and exists(state_latent_tokens):
            _, x = unpack_prepend(x)

        x = self.proj_out(x)

        x, = unpack_action(x)

        return x

# latent action model

def LatentActionModel(
    dim,
    dim_state_latent = None,
    depth = 2,
    noise_std = 1.0,
    loss_fn = F.mse_loss,
    prepend_state_latent = False,
    causal_mask = False
):
    model = ActionVectorField(
        dim,
        dim_state_latent,
        depth = depth,
        prepend_state_latent = prepend_state_latent,
        causal_mask = causal_mask
    )

    return FlowMatching(
        model = model,
        data_shape = None,
        loss_fn = loss_fn,
        noise_std = noise_std
    )
