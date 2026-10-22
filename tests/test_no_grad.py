"""AI-generated test: an outer `torch.no_grad()` builds no graph in either column,
and outside it only the trained column gets gradients (review finding 7c)."""

import torch

from model import CNNModel, PCAgent


def test_cnn_model_respects_no_grad():
    torch.manual_seed(0)
    model = CNNModel(
        output_dim=5, use_batchnorm=False, adaptor_init="uniform", multi_head=False
    )
    model.prep_for_task(0, reinit=True, copy_kb=False, previous_kb=None)
    x = torch.randn(4, 1, 28, 28)

    with torch.no_grad():
        for mode in ("progress", "compress"):
            y_active, y_kb = model(x, mode=mode)
            assert not y_active.requires_grad and not y_kb.requires_grad, mode

    y_active, y_kb = model(x, mode="progress")
    assert y_active.requires_grad and not y_kb.requires_grad
    y_active, y_kb = model(x, mode="compress")
    assert y_kb.requires_grad and not y_active.requires_grad


def test_pc_agent_respects_no_grad():
    torch.manual_seed(0)
    agent = PCAgent(adaptor_init="uniform")
    agent.prep_for_task("Pong", n_actions=6)
    x = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)

    with torch.no_grad():
        for mode in ("progress", "compress"):
            for out in (*agent.columns(x, mode), *agent(x, mode)):
                assert not out.requires_grad, mode

    h_active, h_kb = agent.columns(x, mode="progress")
    y_active, y_kb = agent(x, mode="progress")
    assert h_active.requires_grad and y_active.requires_grad
    assert not h_kb.requires_grad and not y_kb.requires_grad
    h_active, h_kb = agent.columns(x, mode="compress")
    y_active, y_kb = agent(x, mode="compress")
    assert h_kb.requires_grad and y_kb.requires_grad
    assert not h_active.requires_grad and not y_active.requires_grad
