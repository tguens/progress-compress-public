"""AI-generated test: a progress forward leaves the BatchNorm running statistics
of the frozen KB unchanged (review finding 7b)."""

import torch

from model import CNNModel


def _bn_buffers(column) -> dict[str, torch.Tensor]:
    """Running mean, running variance and batch count of every BatchNorm in a column."""
    return {
        k: v.clone()
        for k, v in column.state_dict().items()
        if "running" in k or "num_batches_tracked" in k
    }


def test_progress_keeps_kb_batchnorm_statistics():
    torch.manual_seed(0)
    model = CNNModel(
        output_dim=5, use_batchnorm=True, adaptor_init="uniform", multi_head=False
    )
    model.prep_for_task(0, reinit=True, copy_kb=False, previous_kb=None)
    model.train()
    kb_before, active_before = (
        _bn_buffers(model.kb_layers),
        _bn_buffers(model.active_layers),
    )
    assert kb_before, "the KB has no BatchNorm buffers"

    model(torch.randn(8, 1, 28, 28), mode="progress")

    kb_after, active_after = (
        _bn_buffers(model.kb_layers),
        _bn_buffers(model.active_layers),
    )
    for k in kb_before:
        assert torch.equal(kb_after[k], kb_before[k]), k
    assert model.kb_layers.training  # the forward restores the train mode of the KB
    # The active column trains, so its statistics move
    assert not torch.equal(
        active_after["blocks.0.1.running_mean"],
        active_before["blocks.0.1.running_mean"],
    )
