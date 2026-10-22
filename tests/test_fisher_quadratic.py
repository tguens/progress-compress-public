# AI-generated
"""AI-generated test of `experiments/fisher_quadratic.py`
on CPU, with a tiny stand-in KB and synthetic states: the JVP against finite
differences, the full quadratic against the measured KL at a small displacement, finite
diagonal and sampled quadratics, and the checkpoint inventory of the archived layout."""

import math

import torch
import torch.nn as nn

from experiments.fisher_quadratic import (
    check_jvp,
    diag_fisher,
    diag_quadratic,
    full_quadratic,
    inventory,
    load_kb_parts,
    measured_kl,
    sampled_fisher,
    transitions,
)

N_STATES, N_FEATURES, N_ACTIONS = 64, 8, 6


def _stand_in():
    """A backbone of 2 linear layers and a head of 6 actions, the interface of the KB
    (backbone features, then a fixed head), in float64. θ_next = θ_ref + a small Δθ."""
    torch.manual_seed(0)
    backbone = nn.Sequential(
        nn.Linear(N_FEATURES, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh()
    ).double()
    head = nn.Linear(16, N_ACTIONS).double().requires_grad_(False)
    theta = {n: p.detach() for n, p in backbone.named_parameters()}  # θ_ref
    delta = {n: 0.1 * torch.randn_like(p) for n, p in theta.items()}  # Δθ
    obs = torch.randn(N_STATES, N_FEATURES, dtype=torch.float64)  # synthetic states
    return backbone, head, theta, delta, obs


def test_jvp_matches_finite_differences():
    backbone, head, theta, delta, obs = _stand_in()
    rel_fd, rel_grad = check_jvp(backbone, head, theta, delta, obs[:16])
    assert rel_fd < 1e-5, rel_fd
    assert rel_grad < 1e-5, rel_grad


def test_full_quadratic_matches_kl_at_small_alpha():
    backbone, head, theta, delta, obs = _stand_in()
    alpha = 0.01
    q_full = alpha**2 * full_quadratic(backbone, head, theta, delta, obs, 16)
    kl = measured_kl(backbone, head, theta, delta, obs, alpha, 16)
    assert kl > 0
    assert abs(q_full - kl) / kl < 0.1, (q_full, kl)


def test_diagonal_and_sampled_quadratics_are_finite():
    backbone, head, theta, delta, obs = _stand_in()
    q_full = full_quadratic(backbone, head, theta, delta, obs, 16)
    q_diag = diag_quadratic(diag_fisher(backbone, head, theta, obs, 16), delta)
    q_sampled = diag_quadratic(sampled_fisher(backbone, head, obs, 16, seed=0), delta)
    assert all(math.isfinite(q) for q in (q_full, q_diag, q_sampled))

    # Δθ on one coordinate: no cross term, so the diagonal quadratic is the full one
    one = {n: torch.zeros_like(d) for n, d in delta.items()}
    one["0.weight"][3, 2] = 0.1
    q_full_one = full_quadratic(backbone, head, theta, one, obs, 16)
    q_diag_one = diag_quadratic(diag_fisher(backbone, head, theta, obs, 16), one)
    assert math.isclose(q_diag_one, q_full_one, rel_tol=1e-9)


def test_inventory_of_one_head_per_file(tmp_path):
    """The archived layout: teacher_T<k>.pt holds one KB head, `kb_layers.actor`, the
    head of the game of task k after its last compress before task k."""
    games = ["A-v5", "B-v5", "A-v5", "B-v5"]
    for k in range(len(games)):
        state = {
            "kb_layers.fc.weight": torch.full((2, 2), float(k)),
            "kb_layers.actor.weight": torch.zeros(3, 2),
            "kb_layers.actor.bias": torch.zeros(3),
            "active_layers.actor.weight": torch.zeros(3, 2),
        }
        torch.save(state, tmp_path / f"teacher_T{k}.pt")
    ckpts = [
        load_kb_parts(tmp_path / f"teacher_T{k}.pt", g) for k, g in enumerate(games)
    ]
    assert list(ckpts[2]["heads"]) == ["A-v5"]
    assert list(ckpts[2]["backbone"]) == ["fc.weight"]

    sources = inventory(ckpts, games)
    # Head of A after compress 0: in the file of its revisit, T2. After compress 2: in no file.
    assert [s["A-v5"] for s in sources] == [None, 2, 2, None]
    assert [s["B-v5"] for s in sources] == [None, None, 3, 3]
    assert transitions(sources, ckpts, "A-v5") == [(1, 2), (2, 2)]
    assert transitions(sources, ckpts, "B-v5") == [(2, 3)]
