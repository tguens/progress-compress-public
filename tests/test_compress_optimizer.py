"""AI-generated test. The Atari compress phase starts each DAgger round and the
online phase with a fresh optimizer, so no SGD momentum crosses a boundary, as in
the pre-cleanup entry point `main_atari.py`, which produced the archived runs (review of dc09637)."""

import sys
import types

import pytest
import torch
from omegaconf import OmegaConf

# envpool has no macOS wheel; the test stubs the environments and the rollouts.
sys.modules.setdefault("envpool", types.ModuleType("envpool"))
import run_atari  # noqa: E402
from model import PCAgent  # noqa: E402


class _Envs:
    def reset(self):
        return (torch.zeros(2, 4, 84, 84, dtype=torch.uint8), {})

    def close(self):
        pass


class _Task:
    game = "Pong-v5"
    n_actions = 6

    def train_envs(self, num_envs, env_threads):
        return _Envs()


def _cfg(online: bool):
    return OmegaConf.create(
        {
            "compress_optimizer": "sgd",
            "compress_lr": 0.01,
            "sgd_momentum": 0.9,
            "compress_online": online,
            "compress_n_rounds": 3,
            "compress_n_states": 8,
            "compress_batch_size": 1,
            "n_epochs_compress": 1,
            "compress_head_warmup": 0,
            "num_envs": 2,
            "env_threads": 1,
        }
    )


@pytest.mark.parametrize("online", [False, True])
def test_fresh_optimizer_per_round_and_online(monkeypatch, online):
    agent = PCAgent(adaptor_init="uniform")
    agent.prep_for_task("Pong-v5", 6)
    seen = []

    def fake_distill(agent, obs, logits, n_epochs, optimizer, *args, **kwargs):
        seen.append(("round", optimizer))
        # One real step, so the optimizer gets a momentum buffer.
        loss = sum(p.sum() for p in agent.parameters_for("compress"))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def fake_online(agent, teacher, task, cfg, device, optimizer, *args, **kwargs):
        seen.append(("online", optimizer))
        return []

    monkeypatch.setattr(run_atari, "rollout_states", lambda *a, **k: None)
    monkeypatch.setattr(run_atari, "_distill_epochs", fake_distill)
    monkeypatch.setattr(run_atari, "_compress_online", fake_online)

    run_atari.compress(agent, agent, _Task(), _cfg(online), "cpu", None, None, None, 0)

    kinds = [k for k, _ in seen]
    assert kinds == (["round", "online"] if online else ["round"] * 3)
    optimizers = [o for _, o in seen]
    assert len({id(o) for o in optimizers}) == len(optimizers)
    # The round-0 optimizer has momentum after its step; every later one starts empty.
    assert len(optimizers[0].state) > 0
    if online:
        assert len(optimizers[1].state) == 0
