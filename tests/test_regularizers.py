import pytest
import torch

from x_jepa.regularizers import uniform_wasserstein_loss


@pytest.mark.parametrize(
    'samples',
    (
        torch.tensor([[0.]]),
        torch.tensor([[-0.5], [0.5]])
    )
)
def test_uniform_wasserstein_uses_bin_midpoints(samples):
    loss = uniform_wasserstein_loss(samples)

    torch.testing.assert_close(loss, torch.tensor(0.))
