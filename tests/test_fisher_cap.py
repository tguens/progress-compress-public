"""AI-generated test: `empirical_fisher` stops at `max_n_samples`, also inside a batch,
and the mean divides by the number of processed samples (review finding 7a)."""

import torch
import torch.nn as nn

from utils import empirical_fisher


class _OneParamModel(nn.Module):
    """A KB with one parameter tensor and the (active, kb) output contract of `CNNModel`."""

    def __init__(self):
        super().__init__()
        self.kb_layers = nn.Linear(1, 2, bias=False)

    def forward(self, x: torch.Tensor, mode: str):
        y = self.kb_layers(x)  # [B, 2]
        return y, y


def test_fisher_stops_at_max_n_samples():
    torch.manual_seed(0)
    model = _OneParamModel()
    x, y = torch.randn(5, 1), torch.tensor([0, 1, 0, 1, 1])
    # 5 samples in batches of 2, 2, 1
    loader = [(x[i : i + 2], y[i : i + 2]) for i in range(0, 5, 2)]

    # One gradient of the KB parameter per processed sample
    n_processed = []
    model.kb_layers.weight.register_hook(lambda g: n_processed.append(1))
    fisher = empirical_fisher(model, loader, max_n_samples=3)
    assert len(n_processed) == 3

    # F = 1/N sum_n (d log p(y_n | x_n) / d theta)^2 over the first N = 3 samples
    expected = torch.zeros_like(model.kb_layers.weight)
    for i in range(3):
        loss = nn.functional.cross_entropy(
            model(x[i : i + 1], mode="compress")[1], y[i : i + 1]
        )
        (grad,) = torch.autograd.grad(loss, model.kb_layers.weight)
        expected += grad.pow(2)
    torch.testing.assert_close(fisher["weight"], expected / 3)
