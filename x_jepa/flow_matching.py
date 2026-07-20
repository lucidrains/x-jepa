import torch
from torch import cat
from torch.nn import Module
import torch.nn.functional as F

from torch_einops_utils import pad_right_ndim_to
from x_mlps_pytorch import MLP

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# flow matching

class FlowMatching(Module):
    def __init__(
        self,
        model: Module,
        data_shape = None,
        loss_fn = F.mse_loss,
        noise_std = 1.
    ):
        super().__init__()
        self.model = model
        self.data_shape = data_shape

        self.loss_fn = loss_fn
        self.noise_std = noise_std

    @torch.no_grad()
    def sample(
        self,
        steps = 16,
        batch_size = 1,
        data_shape = None,
        return_noise = False,
        noise = None,
        **kwargs
    ):

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'shape of the data must be passed in, or set at init or during training'
        device = next(self.model.parameters()).device

        init = default(noise, torch.randn((batch_size, *data_shape), device = device) * self.noise_std)
        state = init

        times = torch.linspace(0., 1., steps + 1, device = device)
        times = times[:-1]

        delta = 1. / steps

        for time in times:
            time = time.expand(batch_size)
            pred_flow = self.model(state, time = time, **kwargs)
            state = state + delta * pred_flow

        out = state

        if not return_noise:
            return out

        return out, init

    @torch.no_grad()
    def invert(
        self,
        data,
        steps = 16,
        **kwargs
    ):
        device = data.device
        batch_size = data.shape[0]

        state = data

        times = torch.linspace(1., 0., steps + 1, device = device)
        times = times[:-1]

        delta = -1. / steps

        for time in times:
            time = time.expand(batch_size)
            pred_flow = self.model(state, time = time, **kwargs)
            state = state + delta * pred_flow

        return state

    def forward(
        self,
        data,
        noise = None,
        times = None,
        loss_reduction = 'mean',
        **kwargs
    ):
        shape, ndim = data.shape, data.ndim
        self.data_shape = default(self.data_shape, shape[1:])
        batch, device = shape[0], data.device

        times = default(times, torch.rand(batch, device = device))

        noise = default(noise, torch.randn_like(data) * self.noise_std)
        flow = data - noise

        padded_times = pad_right_ndim_to(times, ndim)
        noised_data = noise.lerp(data, padded_times)

        pred_flow = self.model(noised_data, time = times, **kwargs)

        return self.loss_fn(flow, pred_flow, reduction = loss_reduction)
