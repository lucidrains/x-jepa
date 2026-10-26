"""log-depth recurrent backbone - https://arxiv.org/abs/2609.28212"""

import torch
from torch.nn import Module, ModuleList, RMSNorm

from log_depth_recurrent_modeling import ARGRCLayer

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

class LogDepthRecurrentModel(Module):
    """drop-in replacement for `x_jepa.x_jepa.Transformer`"""

    def __init__(
        self,
        dim,
        *,
        depth,
        max_seq_len = None,
        prenorm = True,
        shift_tokens = False
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.heads = 1
        self.dim_head = dim

        self.layers = ModuleList([
            ARGRCLayer(
                dim = dim,
                max_seq_len = max_seq_len,
                prenorm = prenorm,
                shift_tokens = shift_tokens
            ) for _ in range(depth)
        ])

        self.norm = RMSNorm(dim)

    def forward(
        self,
        tokens,
        past_layers = None,
        past_layers_mask = None,
        return_hiddens = False,
        memories = None,
        prepend_memories = None,
        return_memories = False,
        cond = None,
        prepend_embeds = None,
        excise_prepend_embeds = False
    ):
        assert not exists(past_layers) and not exists(past_layers_mask) and not exists(prepend_memories) and not exists(cond)

        num_prepended = 0

        if exists(prepend_embeds):
            num_prepended = prepend_embeds.shape[-2]
            tokens = torch.cat((prepend_embeds, tokens), dim = -2)

        seq_len = tokens.shape[-2]
        past_seq_len, layer_memories = default(memories, (0, None))
        layer_memories = default(layer_memories, [None] * len(self.layers))

        hiddens = [tokens]
        x = tokens
        next_memories = []

        for layer, memory in zip(self.layers, layer_memories):
            out, next_memory = layer(x, memory = memory, return_memory = True)
            next_memories.append(next_memory)
            x = x + out
            hiddens.append(x)

        if num_prepended > 0 and excise_prepend_embeds:
            x = x[:, num_prepended:]
            hiddens = [h[:, num_prepended:] for h in hiddens]

        x = self.norm(x)
        hiddens[-1] = x

        out = (x, hiddens) if return_hiddens else x

        if not return_memories:
            return out

        return out, (past_seq_len + seq_len, next_memories)
