# AI-generated
"""AI-generated script: Fisher quadratics and measured
policy drift. It moves the KB backbone from one teacher checkpoint to the next, with
the head of one game held fixed. It measures the KL of that game's KB policy and
compares it with three quadratic predictions. It trains nothing.

Run from the repository root. The inventory needs no GPU and no envpool:

    uv run --no-sync python -m experiments.fisher_quadratic --teachers outputs/atari/paper3/pc_s0 --inventory-only

The full analysis runs on the pod (GPU, envpool):

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync python -m experiments.fisher_quadratic \
        --teachers outputs/atari/paper3/pc_s0 --game Hero-v5 --states rollout --check \
        --out outputs/analysis/e4/pc_s0

What the checkpoints hold
-------------------------
`teacher_T<k>.pt` is the whole agent after progress phase k. Its KB backbone B_k is the
KB after compress k-1; B_0 is the initial KB. A KB head of game g in that file is the
head after the last compress of g before task k, or a fresh head on the first visit of g.
The archived entry point `main_atari.py` (the paper3 teachers) keeps one KB head per
file, `kb_layers.actor`, the head of the game of task k. `run_atari.py` keeps every head
seen so far in `kb_heads`. No file holds the backbone after the last compress, nor a
Fisher. The EWC anchor of compress k is B_k.

The KB policy of g after compress k-1 is (B_k, h_g). h_g is the head of g after its last
compress before task k. The inventory finds a file that holds h_g. With one head per file,
that file is the file of the next visit of g, not file k. `re_evaluate` measured this policy
after task k-1, so a rollout run replays that evaluation and compares the return with
`results.json`.

A transition k -> k+1 for game g keeps h_g fixed and moves the backbone only:

    θ_ref = B_k,   θ_next = B_{k+1},   Δθ = θ_next - θ_ref   (backbone parameters only)

When compress k is on g itself, the head moved too. Then (B_{k+1}, h_g) is not a policy
that existed; the report marks it "own compress".

Estimators, on the N states s of one transition, all in float64:

    KL(α)        = 1/N Σ_s Σ_a π_ref(a|s) [log π_ref(a|s) - log π_(θ_ref + αΔθ)(a|s)]
    v(s, a)      = ∇_θ log π_ref(a|s) · Δθ   (one forward-mode JVP gives every action)
    Q_full(α)    = α² / (2N) Σ_s Σ_a π_ref(a|s) v(s, a)²
    F_diag,j     = 1/N Σ_s Σ_a π_ref(a|s) [∂_j log π_ref(a|s)]²
    Q_diag(α)    = α² / 2 Σ_j F_diag,j Δθ_j²
    F_samp,j     = 1/N Σ_s [∂_j log π_ref(a_s|s)]²,  a_s ~ π_ref(·|s)
    Q_sampled(α) = α² / 2 Σ_j F_samp,j Δθ_j²

F_samp is the estimator of the training code (`fisher_batches` and `utils.empirical_fisher`),
here without the clip. Q_sampled is the mean of two independent draws of the actions. Its
sampling error is |Q_1 - Q_2| / 2, the standard error of a mean of two draws.
Q_full is the second-order term of the Taylor series of KL(α) at α = 0, so Q_full / KL
goes to 1 as α goes to 0. The report gives a ratio Q / KL only where KL > --noise-floor.

Outputs under --out: `<game>_T<k>_T<k+1>.md` per transition, `<game>_summary.csv` with one
row per transition and α, and `<game>_T<k>_states.pt` (the states and the rollout record).
"""

import argparse
import copy
import csv
import hashlib
import math
import subprocess
import time
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.distributions.categorical import Categorical
from torch.func import functional_call, grad, jvp, vmap

from src.model import _NatureCNN, load_teacher_state_dict
from src.utils import empirical_fisher
from src.viz import load_results

Params = Dict[str, torch.Tensor]  # keyed by `named_parameters()` of the KB backbone

ALPHAS = (1.0, 0.3, 0.1, 0.03, 0.01)  # the displacement scales α of θ_ref + αΔθ
NUM_ENVS = 16  # rollout environments, as `num_envs` of the paper3 runs
SEED_OFFSET = 2000  # rollout seed = 2000 + --seed: not the training (seed) or evaluation (seed + 1000) seeds


# ---------------------------------
#       Checkpoint inventory
# ---------------------------------
def last_compress(games: list[str], game: str, before: int) -> int | None:
    """The last task before task `before` that plays `game`: its compress trained the
    head of `game`. None when `game` has no compress before task `before`."""
    tasks = [j for j in range(before) if games[j] == game]
    return tasks[-1] if tasks else None


def load_kb_parts(path: Path, game: str) -> dict:
    """The KB parts of a teacher file whose task plays `game`: the backbone B_k
    {name: tensor}, the KB heads {game: {"weight", "bias"}}, the layout and a file hash."""
    state = torch.load(path, map_location="cpu")
    layout = "kb_layers.actor" if "kb_layers.actor.weight" in state else "kb_heads"
    state = load_teacher_state_dict(
        state, game
    )  # kb_layers.actor.* -> kb_heads.<game>.*
    backbone = {
        k.removeprefix("kb_layers."): v
        for k, v in state.items()
        if k.startswith("kb_layers.")
    }
    heads: dict[str, Params] = {}
    for k, v in state.items():
        if k.startswith("kb_heads."):
            head_game, name = k.removeprefix("kb_heads.").rsplit(".", 1)
            heads.setdefault(head_game, {})[name] = v
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return {"backbone": backbone, "heads": heads, "layout": layout, "sha256": sha256}


def inventory(
    ckpts: list[dict | None], games: list[str]
) -> list[dict[str, int | None]]:
    """sources[k][g]: a file that holds the head of g after its last compress before
    task k, so that (B_k, that head) is the KB policy of g after compress k - 1.
    None when g has no compress before task k, when no file holds that head, or when
    teacher_T<k>.pt is missing."""
    sources = []
    for k in range(len(games)):
        row: dict[str, int | None] = {}
        for g in dict.fromkeys(games):
            c = last_compress(games, g, k)
            files = [
                j
                for j, ck in enumerate(ckpts)
                if ck is not None
                and g in ck["heads"]
                and last_compress(games, g, j) == c
            ]
            usable = ckpts[k] is not None and c is not None and files
            row[g] = files[0] if usable else None
        sources.append(row)
    return sources


def transitions(
    sources: list[dict[str, int | None]],
    ckpts: list[dict | None],
    game: str,
) -> list[tuple[int, int]]:
    """(k, head file) of every pair B_k -> B_{k+1} where the KB policy of `game` at B_k
    can be reconstructed and B_{k+1} exists."""
    return [
        (k, sources[k][game])  # type: ignore[misc]
        for k in range(len(ckpts) - 1)
        if sources[k][game] is not None and ckpts[k + 1] is not None
    ]


def print_inventory(
    teachers: Path,
    games: list[str],
    ckpts: list[dict | None],
    sources: list[dict[str, int | None]],
) -> None:
    n_files = sum(ck is not None for ck in ckpts)
    print(f"Teachers: {teachers} ({n_files} of {len(games)} files teacher_T<k>.pt)")
    print(
        f"{'k':>3}  {'game':<18}{'layout':<17}{'backbone B_k':<19}KB heads in the file"
    )
    for k, (g, ck) in enumerate(zip(games, ckpts)):
        if ck is None:
            print(f"{k:>3}  {g:<18}missing")
            continue
        backbone = "initial KB" if k == 0 else f"after compress {k - 1}"
        heads = ", ".join(
            f"{h} ("
            + (
                "fresh"
                if last_compress(games, h, k) is None
                else f"after compress {last_compress(games, h, k)}"
            )
            + ")"
            for h in ck["heads"]
        )
        print(f"{k:>3}  {g:<18}{ck['layout']:<17}{backbone:<19}{heads}")

    distinct = list(dict.fromkeys(games))
    print(
        "\nKB policy of g after compress k-1 = (B_k, head of g after its last compress before task k)."
    )
    print(
        'Cell: the file that holds that head; "." = no compress of g before task k; '
        '"-" = no file holds the head; "x" = teacher_T<k>.pt is missing.'
    )
    print(f"{'k':>3}  " + "".join(f"{g:<18}" for g in distinct))
    for k in range(len(games)):
        cells = []
        for g in distinct:
            if ckpts[k] is None:
                cells.append("x")
            elif last_compress(games, g, k) is None:
                cells.append(".")
            elif sources[k][g] is None:
                cells.append("-")
            else:
                cells.append(f"T{sources[k][g]}")
        print(f"{k:>3}  " + "".join(f"{c:<18}" for c in cells))
    print(
        f"\nIn no file: the backbone after compress {len(games) - 1}, the per-task and the running Fisher. "
        "The EWC anchor of compress k is B_k."
    )


# ---------------------------------
#       Log-policy derivatives
# ---------------------------------
def _batches(obs: torch.Tensor, batch_size: int, theta: Params):
    """Batches of `obs` on the device and in the dtype of `theta`."""
    like = next(iter(theta.values()))
    for lo in range(0, obs.shape[0], batch_size):
        yield obs[lo : lo + batch_size].to(device=like.device, dtype=like.dtype)


def log_policy(
    backbone: nn.Module, head: nn.Module, theta: Params, x: torch.Tensor
) -> torch.Tensor:
    """log π_θ(·|s) [B, A] of the KB alone: the backbone with the parameters `theta`,
    then the fixed head of the game."""
    h = functional_call(backbone, theta, (x,))  # [B, 512]
    return torch.log_softmax(head(h), dim=1)  # [B, A]


def log_policy_jvp(
    backbone: nn.Module, head: nn.Module, theta: Params, delta: Params, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """log π_ref(·|s) [B, A] and v(s, a) = ∇_θ log π_ref(a|s) · Δθ [B, A] for every
    action, in one forward-mode pass (torch.func.jvp). No Jacobian is built."""
    return jvp(lambda th: log_policy(backbone, head, th, x), (theta,), (delta,))


def per_action_grads(
    backbone: nn.Module, head: nn.Module, theta: Params, x: torch.Tensor
) -> Params:
    """∂_θ log π(a|s) for every state s of the batch and every action a, as
    {name: [B, A, *p.shape]}: torch.func.vmap(grad) over the states and the actions."""
    eye = torch.eye(
        head.out_features, dtype=x.dtype, device=x.device
    )  # [A, A], row a = e_a

    def log_prob(th: Params, s: torch.Tensor, e_a: torch.Tensor) -> torch.Tensor:
        # log π(a|s) = e_a · log π(·|s), for one state s and one action a
        return (log_policy(backbone, head, th, s.unsqueeze(0))[0] * e_a).sum()

    over_actions = vmap(grad(log_prob), in_dims=(None, None, 0))
    return vmap(over_actions, in_dims=(None, 0, None))(theta, x, eye)


def check_jvp(
    backbone: nn.Module,
    head: nn.Module,
    theta: Params,
    delta: Params,
    x: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[float, float]:
    """The relative error of v(s, a) from the JVP against (1) the central difference
    [log π_(θ+εΔθ) - log π_(θ-εΔθ)] / (2ε) and (2) Σ_j ∂_j log π(a|s) Δθ_j from the
    per-sample gradients of `diag_fisher`. Run it in float64."""
    x = next(_batches(x, x.shape[0], theta))
    _, v = log_policy_jvp(backbone, head, theta, delta, x)  # [B, A]
    with torch.no_grad():
        plus = log_policy(
            backbone, head, {n: theta[n] + eps * delta[n] for n in theta}, x
        )
        minus = log_policy(
            backbone, head, {n: theta[n] - eps * delta[n] for n in theta}, x
        )
    v_fd = (plus - minus) / (2 * eps)  # [B, A]
    g = per_action_grads(backbone, head, theta, x)  # {name: [B, A, *p.shape]}
    v_grad = sum((g[n] * delta[n]).flatten(2).sum(2) for n in theta)  # [B, A]
    return (
        float((v - v_fd).norm() / v_fd.norm()),
        float((v - v_grad).norm() / v_grad.norm()),
    )


# ---------------------------------
#       KL and its quadratics
# ---------------------------------
@torch.no_grad()
def measured_kl(
    backbone: nn.Module,
    head: nn.Module,
    theta: Params,
    delta: Params,
    obs: torch.Tensor,
    alpha: float,
    batch_size: int,
) -> float:
    """KL(π_ref || π_(θ_ref + αΔθ)), the mean over the states."""
    moved = {n: theta[n] + alpha * delta[n] for n in theta}
    total = 0.0
    for x in _batches(obs, batch_size, theta):
        logp_ref = log_policy(backbone, head, theta, x)  # [B, A]
        logp = log_policy(backbone, head, moved, x)  # [B, A]
        # Σ_a π_ref(a|s) [log π_ref(a|s) - log π(a|s)]
        total += float((logp_ref.exp() * (logp_ref - logp)).sum())
    return total / obs.shape[0]


def full_quadratic(
    backbone: nn.Module,
    head: nn.Module,
    theta: Params,
    delta: Params,
    obs: torch.Tensor,
    batch_size: int,
) -> float:
    """Q_full(1) = 1/(2N) Σ_s Σ_a π_ref(a|s) v(s, a)². Q_full(α) = α² Q_full(1)."""
    total = 0.0
    for x in _batches(obs, batch_size, theta):
        logp, v = log_policy_jvp(backbone, head, theta, delta, x)  # [B, A], [B, A]
        total += float((logp.exp() * v.pow(2)).sum())
    return total / (2 * obs.shape[0])


def diag_fisher(
    backbone: nn.Module,
    head: nn.Module,
    theta: Params,
    obs: torch.Tensor,
    batch_size: int,
) -> Params:
    """F_diag,j = 1/N Σ_s Σ_a π_ref(a|s) [∂_j log π_ref(a|s)]², the model Fisher
    with the exact action weighting. Peak memory: batch_size x A x n_params values."""
    fisher = {n: torch.zeros_like(p) for n, p in theta.items()}
    for x in _batches(obs, batch_size, theta):
        with torch.no_grad():
            pi = log_policy(backbone, head, theta, x).exp()  # π_ref(a|s) [B, A]
        g = per_action_grads(backbone, head, theta, x)  # {name: [B, A, *p.shape]}
        for n, g_n in g.items():
            w = pi.view(*pi.shape, *([1] * (g_n.dim() - 2)))  # [B, A, 1, ...]
            # Σ_s Σ_a π_ref(a|s) [∂_j log π_ref(a|s)]², in place on the gradients
            fisher[n] += g_n.pow_(2).mul_(w).sum(dim=(0, 1))
    return {n: f / obs.shape[0] for n, f in fisher.items()}


class _KBPolicy(nn.Module):
    """The KB alone with one head, with the (active, kb) output contract of `PCAgent`,
    so that `utils.empirical_fisher` (the estimator of the training code) runs on it."""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.kb_layers = backbone
        self.kb_head = head

    def forward(self, x: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.kb_head(self.kb_layers(x))  # [B, A]
        return y, y


def sampled_fisher(
    backbone: nn.Module, head: nn.Module, obs: torch.Tensor, batch_size: int, seed: int
) -> Params:
    """F_samp,j = 1/N Σ_s [∂_j log π_ref(a_s|s)]², one action a_s ~ π_ref(·|s) per
    state, as `fisher_batches` and `utils.empirical_fisher` compute it in training
    (without the clip). The parameters of `backbone` must be θ_ref."""
    theta = dict(backbone.named_parameters())
    device = next(iter(theta.values())).device
    generator = torch.Generator(device=device).manual_seed(seed)

    def batches():
        for x in _batches(obs, batch_size, theta):
            with torch.no_grad():
                probs = torch.softmax(head(backbone(x)), dim=1)  # [B, A]
            action = torch.multinomial(probs, 1, generator=generator).squeeze(1)  # [B]
            yield x, action

    return empirical_fisher(_KBPolicy(backbone, head), batches(), device=device)


def diag_quadratic(fisher: Params, delta: Params) -> float:
    """Q(1) = 1/2 Σ_j F_j Δθ_j² for a diagonal Fisher F. Q(α) = α² Q(1)."""
    return 0.5 * sum(float((fisher[n] * delta[n].pow(2)).sum()) for n in delta)


def table_rows(
    kls: dict[float, float],
    q_full: float,
    q_diag: float,
    q_sampled: float,
    q_sampled_error: float,
    noise_floor: float,
) -> list[dict]:
    """One row per α: the measured KL, the three quadratics at α, the absolute errors,
    and the ratios Q / KL where KL > `noise_floor` (NaN below)."""
    rows = []
    for alpha, kl in kls.items():
        q = {
            "full": alpha**2 * q_full,
            "diag": alpha**2 * q_diag,
            "sampled": alpha**2 * q_sampled,
        }
        row = {
            "alpha": alpha,
            "kl": kl,
            "q_full": q["full"],
            "q_diag": q["diag"],
            "q_sampled": q["sampled"],
            "q_sampled_error": alpha**2 * q_sampled_error,
        }
        for name, value in q.items():
            row[f"abs_err_{name}"] = abs(value - kl)
        for name, value in q.items():
            row[f"ratio_{name}"] = value / kl if kl > noise_floor else math.nan
        rows.append(row)
    return rows


# ---------------------------------
#       States, verification
# ---------------------------------
def pick_action(
    logits: torch.Tensor, mode: Literal["greedy", "sample"]
) -> torch.Tensor:
    """As `run_atari.pick_action`: `sample` draws from the policy, `greedy` takes its argmax."""
    if mode == "sample":
        return Categorical(logits=logits).sample()
    elif mode == "greedy":
        return logits.argmax(dim=1)
    else:
        raise ValueError(f"Invalid policy mode: {mode}")


@torch.no_grad()
def collect_states(
    game: str,
    backbone: nn.Module,
    head: nn.Module,
    n_states: int,
    stride: int,
    mode: Literal["greedy", "sample"],
    seed: int,
    num_threads: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """`n_states` states of full games (the evaluation environments: no episodic-life
    reset, no reward clip), visited by the KB policy (backbone, head) with `mode`
    actions. Keep one state every `stride` steps in each of `NUM_ENVS` environments.
    Returns the states [N, 4, 84, 84] uint8 on the CPU and the rollout record."""
    from src.environments.atari import AtariTask  # envpool exists on the pod only

    envs = AtariTask(game, seed=seed).eval_envs(NUM_ENVS, num_threads, seed=seed)
    next_obs = envs.reset()[0]
    kept: list[torch.Tensor] = []
    ep_len = np.zeros(NUM_ENVS, dtype=np.int64)
    ep_return = np.zeros(NUM_ENVS)
    lengths: list[int] = []
    returns: list[float] = []
    step = 0
    while len(kept) * NUM_ENVS < n_states:
        obs = torch.as_tensor(next_obs, device=device)  # [E, 4, 84, 84] uint8
        if step % stride == 0:
            kept.append(obs.cpu())
        action = pick_action(head(backbone(obs)), mode)
        next_obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
        ep_len += 1
        ep_return += reward
        done = np.logical_or(terminated, truncated)  # the end of a full game
        lengths += ep_len[done].tolist()
        returns += ep_return[done].tolist()
        ep_len[done], ep_return[done] = 0, 0.0
        step += 1
    envs.close()
    record = {
        "policy_mode": mode,
        "stride": stride,
        "num_envs": NUM_ENVS,
        "env_seed": seed,
        "steps_per_env": step,
        "completed_episodes": len(lengths),
        "episode_lengths": lengths,
        "episode_returns": returns,
        "unfinished_episode_steps": ep_len.tolist(),
    }
    return torch.cat(kept)[:n_states], record


@torch.no_grad()
def verify_return(
    game: str, backbone: nn.Module, head: nn.Module, cfg, device: torch.device
) -> float:
    """The return of (B_k, h_g) in the evaluation of `re_evaluate`: `cfg.eval_mode`
    actions, `cfg.n_eval_episodes` full games, environment seed `cfg.seed` + 1000."""
    from src.environments.atari import AtariTask, evaluate_policy  # pod only

    return evaluate_policy(
        AtariTask(game, seed=cfg.seed),
        lambda obs: pick_action(head(backbone(obs)), cfg.eval_mode),
        cfg.n_eval_episodes,
        seed=cfg.seed + 1000,
        device=device,
        num_threads=cfg.env_threads,
    )["mean_return"]


def logged_return(results: dict | None, k: int, game: str) -> float | None:
    """The return of the KB policy of `game` after compress k - 1, as `results.json`
    logged it after task k - 1 (`re_eval_returns`)."""
    if results is None:
        return None
    log = results["task_logs"][k - 1]
    return log["re_eval_returns"][log["re_eval_games"].index(game)]


# ---------------------------------
#       One transition
# ---------------------------------
def kb_modules(
    backbone_state: Params, head_state: Params, device: torch.device
) -> tuple[nn.Module, nn.Module]:
    """The KB backbone and the fixed KB head of a game, in float32 as trained."""
    backbone = _NatureCNN()
    backbone.load_state_dict(backbone_state)
    head = nn.Linear(backbone.feature_dim, head_state["weight"].shape[0])
    head.load_state_dict(head_state)
    head.requires_grad_(False)
    return backbone.to(device).eval(), head.to(device).eval()


def git_commit() -> str:
    """The code commit, with "+dirty" when tracked files have uncommitted changes."""
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return head + ("+dirty" if dirty else "")


def _fmt(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.6e}"


def analyse_transition(
    args, cfg, games, ckpts, k: int, head_file: int, results, device, commit: str
) -> list[dict]:
    """States, verification, JVP check, the KL and its quadratics for B_k -> B_{k+1}.
    Writes the markdown report and returns the csv rows."""
    t0 = time.time()
    game = args.game
    own = games[k] == game
    backbone, head = kb_modules(
        ckpts[k]["backbone"], ckpts[head_file]["heads"][game], device
    )

    # States of π_ref: a new rollout, or a saved file ("{k}" in the path is the index k)
    verified = None
    if args.states == "rollout":
        obs, record = collect_states(
            game,
            backbone,
            head,
            args.n_states,
            args.stride,
            args.policy_mode,
            SEED_OFFSET + args.seed,
            cfg.env_threads,
            device,
        )
        torch.save({"obs": obs, "record": record}, args.out / f"{game}_T{k}_states.pt")
        verified = verify_return(game, backbone, head, cfg, device)
    else:
        saved = torch.load(args.states.format(k=k), map_location="cpu")
        obs, record = (
            (saved["obs"], saved.get("record", {}))
            if isinstance(saved, dict)
            else (saved, {})
        )
        record = {"source": args.states.format(k=k), **record}
    logged = logged_return(results, k, game)

    # float64 copies for the estimators; the parameters are cast exactly
    backbone64, head64 = copy.deepcopy(backbone).double(), copy.deepcopy(head).double()
    theta = {n: p.detach() for n, p in backbone64.named_parameters()}  # θ_ref
    delta = {
        n: ckpts[k + 1]["backbone"][n].to(device, torch.float64) - theta[n]
        for n in theta
    }  # Δθ = θ_next - θ_ref

    check_line = "not run (no --check)"
    if args.check:
        rel_fd, rel_grad = check_jvp(
            backbone64, head64, theta, delta, obs[: args.batch_size]
        )
        check_line = (
            f"{min(args.batch_size, obs.shape[0])} states, float64, ε = 1e-5: relative error "
            f"{rel_fd:.3e} to the central difference, {rel_grad:.3e} to the per-sample gradients"
        )
        print(f"[T{k} -> T{k + 1} check] {check_line}", flush=True)

    q_full = full_quadratic(backbone64, head64, theta, delta, obs, args.batch_size)
    q_diag = diag_quadratic(
        diag_fisher(backbone64, head64, theta, obs, args.batch_size), delta
    )
    q_draws = [
        diag_quadratic(
            sampled_fisher(
                backbone64, head64, obs, args.batch_size, seed=2 * args.seed + d
            ),
            delta,
        )
        for d in (0, 1)
    ]
    q_sampled, q_sampled_error = sum(q_draws) / 2, abs(q_draws[0] - q_draws[1]) / 2
    kls = {
        a: measured_kl(backbone64, head64, theta, delta, obs, a, args.batch_size)
        for a in ALPHAS
    }
    rows = table_rows(kls, q_full, q_diag, q_sampled, q_sampled_error, args.noise_floor)
    wall = time.time() - t0

    # Report
    lengths = record.get("episode_lengths", [])
    coverage = (
        f"{record.get('completed_episodes')} completed episodes, length mean {np.mean(lengths):.1f}, "
        f"min {min(lengths)}, max {max(lengths)}, return mean {np.mean(record['episode_returns']):.1f}; "
        f"unfinished episodes at steps {record.get('unfinished_episode_steps')}"
        if lengths
        else f"no completed episode; record {record}"
    )
    delta_norms = ", ".join(f"{n} {float(d.norm()):.4e}" for n, d in delta.items())
    meta = [
        f"teachers: {args.teachers}; sha256 T{k} {ckpts[k]['sha256']}, T{k + 1} {ckpts[k + 1]['sha256']}, "
        f"head file T{head_file} {ckpts[head_file]['sha256']}",
        f"code: {commit}; device: {device}; float64 for the estimators, float32 for the rollouts; seed {args.seed}",
        f"reference: θ_ref = B_{k} (the KB "
        + ("initial" if k == 0 else f"after compress {k - 1}")
        + f"), head of {game} after compress {last_compress(games, game, k)} (from teacher_T{head_file}.pt)",
        f"next: θ_next = B_{k + 1} (the KB after compress {k}, on {games[k]}); the head is held fixed",
        "own compress: "
        + (
            "yes. The head moved too; (B_k+1, head) is not a policy that existed"
            if own
            else "no"
        ),
        f"‖Δθ‖₂ = {math.sqrt(sum(float(d.pow(2).sum()) for d in delta.values())):.6e}; per tensor: {delta_norms}",
        f"states: N = {obs.shape[0]}, policy mode {record.get('policy_mode')}, stride {record.get('stride')}, "
        f"{record.get('num_envs')} environments, environment seed {record.get('env_seed')}, source {record.get('source', 'rollout')}",
        f"episode coverage: {coverage}",
        f"verification: {cfg.eval_mode} return of (B_{k}, head) over {cfg.n_eval_episodes} games = {verified}; "
        f"logged after task {k - 1}: {logged}"
        + (
            ""
            if verified is None or logged is None
            else f" (difference {verified - logged})"
        ),
        f"JVP check: {check_line}",
        f"Q_sampled draws at α = 1: {q_draws[0]:.6e}, {q_draws[1]:.6e} (one action per state, N states each)",
        f"noise floor: {args.noise_floor:g} nats; wall time: {wall:.1f} s",
        "scope: a local prediction test on these states. The checkpoints hold no running Fisher, "
        "so the clip, decay and state distribution of the training penalty are not tested.",
    ]
    header = [
        "α",
        "KL",
        "Q_full",
        "Q_diag",
        "Q_sampled",
        "Q_sampled error",
        "abs(Q_full - KL)",
        "abs(Q_diag - KL)",
        "abs(Q_sampled - KL)",
        "Q_full / KL",
        "Q_diag / KL",
        "Q_sampled / KL",
    ]
    keys = [k_ for k_ in rows[0] if k_ != "alpha"]
    lines = [f"# E4: {game}, T{k} -> T{k + 1}", "", *(f"- {m}" for m in meta), ""]
    lines += ["| " + " | ".join(header) + " |", "|" + "---:|" * len(header)]
    for r in rows:
        lines.append(
            f"| {r['alpha']:g} | " + " | ".join(_fmt(r[c]) for c in keys) + " |"
        )
    lines += ["", f"n/a: KL ≤ the noise floor ({args.noise_floor:g} nats)."]
    report = args.out / f"{game}_T{k}_T{k + 1}.md"
    report.write_text("\n".join(lines) + "\n")
    r1 = rows[0]
    print(
        f"[T{k} -> T{k + 1}] KL(1) = {r1['kl']:.4e}, Q_full/KL = {r1['ratio_full']:.4f}, "
        f"Q_diag/KL = {r1['ratio_diag']:.4f}, Q_sampled/KL = {r1['ratio_sampled']:.4f} ({wall:.0f} s) -> {report}",
        flush=True,
    )
    transition = {
        "game": game,
        "k_ref": k,
        "k_next": k + 1,
        "compress_game": games[k],
        "own_compress": own,
        "head_file": head_file,
        "n_states": obs.shape[0],
        "policy_mode": record.get("policy_mode"),
        "verified_return": verified,
        "logged_return": logged,
    }
    return [transition | r for r in rows]


# ---------------------------------
#           Main
# ---------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--teachers",
        type=Path,
        required=True,
        help="run directory with teacher_T<k>.pt and .hydra/config.yaml",
    )
    parser.add_argument(
        "--inventory-only", action="store_true", help="print the inventory and stop"
    )
    parser.add_argument(
        "--game", help="the game g whose head is held fixed, for example Hero-v5"
    )
    parser.add_argument(
        "--states",
        default="rollout",
        help='"rollout" (envpool), or a saved states file; "{k}" in the path is the index k',
    )
    parser.add_argument(
        "--policy-mode",
        choices=["sample", "greedy"],
        default="sample",
        help="the actions of π_ref in the rollout",
    )
    parser.add_argument("--n-states", type=int, default=2048, help="states per rollout")
    parser.add_argument(
        "--stride",
        type=int,
        default=32,
        help="keep one state every `stride` steps in each environment",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="rollout seed (environments: 2000 + seed) and sampled-Fisher draws (2 seed, 2 seed + 1)",
    )
    parser.add_argument(
        "--noise-floor",
        type=float,
        default=1e-4,
        help="report Q / KL only where KL is above this value (nats)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check the JVP against a finite difference on the first batch",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="states per batch; the per-sample gradients take batch x A x 1.7 M float64 values",
    )
    parser.add_argument("--out", type=Path, help="output directory")
    args = parser.parse_args()
    if not args.inventory_only and (args.game is None or args.out is None):
        parser.error("--game and --out are required unless --inventory-only")
    return args


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.teachers / ".hydra" / "config.yaml")  # the teacher run
    games = list(cfg.games[: cfg.max_n_tasks])
    paths = [args.teachers / f"teacher_T{k}.pt" for k in range(len(games))]
    ckpts = [load_kb_parts(p, g) if p.exists() else None for p, g in zip(paths, games)]
    sources = inventory(ckpts, games)
    print_inventory(args.teachers, games, ckpts, sources)
    if args.inventory_only:
        return
    if args.game not in games:
        raise ValueError(f"Invalid game: {args.game} is not in {games}")

    pairs = transitions(sources, ckpts, args.game)
    print(f"\nTransitions for {args.game} (head fixed, backbone B_k -> B_k+1):")
    for k, head_file in pairs:
        own = " (own compress: the head moved too)" if games[k] == args.game else ""
        print(
            f"  T{k} -> T{k + 1}: compress {k} on {games[k]}, head from teacher_T{head_file}.pt{own}"
        )
    if not pairs:
        print("  none")
        return

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    args.out.mkdir(parents=True, exist_ok=True)
    results = load_results(args.teachers)
    commit = git_commit()
    rows = []
    for k, head_file in pairs:
        rows += analyse_transition(
            args, cfg, games, ckpts, k, head_file, results, device, commit
        )
    csv_path = args.out / f"{args.game}_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(csv_path)


if __name__ == "__main__":
    main()
