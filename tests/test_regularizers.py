import pytest
import torch

from x_jepa.regularizers import sigreg_loss


@pytest.mark.parametrize('dtype', (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64
))
def test_sigreg_supports_floating_point_dtypes(dtype):
    embeddings = torch.randn(2, 3, 8, dtype = dtype, requires_grad = True)

    loss = sigreg_loss(
        embeddings,
        num_slices = 4,
        num_knots = 5
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
