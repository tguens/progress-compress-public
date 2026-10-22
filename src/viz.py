"""AI-generated visualization for P&C experiment results.

Usage:
    # Several seeds of one configuration, with std bands (the final plot)
    uv run src/viz.py --seeds outputs/experiments/final/best_s0 outputs/experiments/final/best_s1 --out outputs/plots/final --title "best"

    # Single run
    uv run src/viz.py outputs/experiments/sgd_mom0.9_lam100 --out outputs/plots/test

    # All runs in a sweep directory
    uv run src/viz.py outputs/experiments --out outputs/plots/experiments
"""

import argparse
import gzip
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def load_results(results_dir: Path) -> dict | None:
    p = results_dir / "results.json"
    if not p.exists() and (results_dir / "results.json.gz").exists():
        # An archived run keeps only the gzip file (gzipped above 5 MB)
        with gzip.open(results_dir / "results.json.gz", "rt") as f:
            return json.load(f)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# AI-generated function
def _smooth(y: np.ndarray, n: int, breaks: list[int] = ()) -> np.ndarray:
    """Centered moving average of window `n` along the last axis. NaN values are
    ignored. The window shrinks symmetrically at the edges, so the first and last
    points keep their raw value. `breaks` are indices where a new segment starts
    (a re-visit); no window crosses one. `n <= 1` returns `y` unchanged."""
    if n <= 1:
        return y
    out = np.full(y.shape, np.nan)
    bounds = [0, *breaks, y.shape[-1]]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        for i in range(lo, hi):
            h = min(n // 2, i - lo, hi - 1 - i)
            window = y[..., i - h : i + h + 1]
            if np.isfinite(window).any():
                out[..., i] = np.nanmean(window, axis=-1)
    return out


# AI-generated function
def _plot_band(
    ax,
    xs,
    mean,
    std,
    color,
    label,
    smooth=0,
    style="-",
    markersize=3,
    alpha=0.2,
    breaks=(),
):
    """Mean line with a std band. With `smooth > 1`, the line and the band are the
    moving average of window `smooth` (split at `breaks`), drawn over the raw means
    as faint markers."""
    if smooth > 1:
        ax.plot(xs, mean, "o", markersize=markersize, color=color, alpha=0.35)
        mean, std = _smooth(mean, smooth, breaks), _smooth(std, smooth, breaks)
        ax.plot(xs, mean, style, color=color, label=label)
    else:
        ax.plot(
            xs, mean, style, marker="o", markersize=markersize, color=color, label=label
        )
    ax.fill_between(xs, mean - std, mean + std, color=color, alpha=alpha)


# ---------------------------------------------------------------------------
# Per-epoch line plots (loss / accuracy)
# ---------------------------------------------------------------------------


def _plot_curves(task_logs, key, title, ylabel, xlabel, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for log in task_logs:
        tid = log["task_id"]
        values = log.get(key, [])
        if not values:
            continue
        ax.plot(values, label=f"Task {tid}", color=COLORS[tid % len(COLORS)])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_accuracy(task_logs, phase, eval_every, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for log in task_logs:
        tid = log["task_id"]
        color = COLORS[tid % len(COLORS)]
        train_accs = log.get(f"{phase}_train_accs", [])
        eval_accs = log.get(f"{phase}_eval_accs", [])
        epochs = [i * eval_every for i in range(len(eval_accs))]
        if train_accs:
            ax.plot(epochs, train_accs, color=color, alpha=0.4, linestyle="--")
        if eval_accs:
            ax.plot(epochs, eval_accs, color=color, label=f"Task {tid}")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Compress loss split: KL vs EWC
# ---------------------------------------------------------------------------


def _plot_compress_loss_split(task_logs, out_path):
    n = len(task_logs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    axes = axes[0]
    for i, log in enumerate(task_logs):
        ax = axes[i]
        kl = log.get("compress_kl_losses", [])
        ewc = log.get("compress_ewc_losses", [])
        total = log.get("compress_losses", [])
        if total:
            ax.plot(total, label="Total", color="black", linewidth=1.5)
        if kl:
            ax.plot(kl, label="KL (distillation)", color=COLORS[0])
        if ewc:
            ax.plot(ewc, label="EWC (λ·Mahal.)", color=COLORS[1])
        ax.set_title(f"Task {log['task_id']}")
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel("Loss")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Compress phase — KL vs EWC loss", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Re-eval accuracy heatmap (forgetting visualization)
# ---------------------------------------------------------------------------


def _plot_forgetting(task_logs, out_path):
    n = len(task_logs)
    mat = np.full((n, n), np.nan)
    for log in task_logs:
        k = log["task_id"]
        for j, acc in enumerate(log["re_eval_accs"]):
            mat[j, k] = acc

    fig, ax = plt.subplots(figsize=(max(4, n + 1), max(3, n)))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    for j in range(n):
        for k in range(n):
            v = mat[j, k]
            if not np.isnan(v):
                ax.text(k, j, f"{v:.2f}", ha="center", va="center", fontsize=10)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"After T{k}" for k in range(n)])
    ax.set_yticklabels([f"Task {j}" for j in range(n)])
    ax.set_xlabel("Training stage")
    ax.set_ylabel("Evaluated on")
    ax.set_title("KB accuracy on previous tasks (forgetting matrix)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fisher evolution per layer
# ---------------------------------------------------------------------------


def _plot_fisher_evolution(task_logs, out_path):
    if "fisher_per_layer" not in task_logs[0]:
        return

    layer_names = list(task_logs[0]["fisher_per_layer"].keys())
    n_tasks = len(task_logs)

    short_names = [n.replace("blocks.", "B").replace(".0.", ".") for n in layer_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: per-task Fisher (stacked bar)
    x = np.arange(n_tasks)
    bottom = np.zeros(n_tasks)
    for i, layer in enumerate(layer_names):
        vals = [log["fisher_per_layer"].get(layer, 0) for log in task_logs]
        ax1.bar(
            x,
            vals,
            bottom=bottom,
            label=short_names[i],
            color=COLORS[i % len(COLORS)],
            width=0.7,
        )
        bottom += np.array(vals)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"T{k}" for k in range(n_tasks)])
    ax1.set_title("Fisher mass per layer (per task)")
    ax1.set_ylabel("Mass")
    ax1.legend(fontsize=6, ncol=2, loc="upper left")
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: running Fisher (stacked bar)
    if "running_fisher_per_layer" in task_logs[0]:
        bottom = np.zeros(n_tasks)
        for i, layer in enumerate(layer_names):
            vals = [
                log.get("running_fisher_per_layer", {}).get(layer, 0)
                for log in task_logs
            ]
            ax2.bar(
                x,
                vals,
                bottom=bottom,
                label=short_names[i],
                color=COLORS[i % len(COLORS)],
                width=0.7,
            )
            bottom += np.array(vals)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"T{k}" for k in range(n_tasks)])
        ax2.set_title("Running Fisher mass per layer (cumulative)")
        ax2.set_ylabel("Mass")
        ax2.legend(fontsize=6, ncol=2, loc="upper left")
        ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def plot_run(results_dir: Path, output_dir: Path) -> bool:
    data = load_results(results_dir)
    if data is None:
        return False

    logs = data["task_logs"]
    if not logs or "progress_losses" not in logs[0]:
        print(f"  SKIP {results_dir.name} — no per-epoch curves (re-run to generate)")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    n_prog = len(logs[0].get("progress_losses", []))
    n_comp = len(logs[0].get("compress_losses", []))

    _plot_curves(
        logs,
        "progress_losses",
        "Progress phase — loss per epoch",
        "Loss",
        "Epoch",
        output_dir / "progress_loss.png",
    )
    _plot_curves(
        logs,
        "compress_losses",
        "Compress phase — loss per epoch",
        "Loss",
        "Epoch",
        output_dir / "compress_loss.png",
    )

    eval_every_prog = max(
        1, n_prog // max(len(logs[0].get("progress_eval_accs", [1])), 1)
    )
    eval_every_comp = max(
        1, n_comp // max(len(logs[0].get("compress_eval_accs", [1])), 1)
    )

    _plot_accuracy(
        logs,
        "progress",
        eval_every_prog,
        "Progress phase — accuracy",
        output_dir / "progress_accuracy.png",
    )
    _plot_accuracy(
        logs,
        "compress",
        eval_every_comp,
        "Compress phase — accuracy",
        output_dir / "compress_accuracy.png",
    )

    # New plots (gracefully skip if fields missing)
    if "compress_kl_losses" in logs[0]:
        _plot_compress_loss_split(logs, output_dir / "compress_loss_split.png")

    _plot_forgetting(logs, output_dir / "forgetting.png")

    if "fisher_per_layer" in logs[0]:
        _plot_fisher_evolution(logs, output_dir / "fisher_evolution.png")

    print(f"  OK {results_dir.name} -> {output_dir}")
    return True


def plot_sweep(sweep_dir: Path, output_dir: Path):
    subdirs = sorted(
        p for p in sweep_dir.iterdir() if p.is_dir() and load_results(p) is not None
    )
    if not subdirs:
        print(f"No results.json found in subdirectories of {sweep_dir}")
        return
    print(f"Found {len(subdirs)} runs in {sweep_dir}\n")
    for d in subdirs:
        plot_run(d, output_dir / d.name)


# ---------------------------------------------------------------------------
# Final plots over seeds: per-task accuracy after each later task, loss split
# ---------------------------------------------------------------------------


def _stack_re_eval(runs: list[dict], key: str = "re_eval_accs") -> np.ndarray:
    """[n_seeds, n_tasks, n_tasks] with acc[s, j, k] = accuracy of task j after task k."""
    n = len(runs[0]["task_logs"])
    mat = np.full((len(runs), n, n), np.nan)
    for s, data in enumerate(runs):
        for log in data["task_logs"]:
            k = log["task_id"]
            for j, acc in enumerate(log[key]):
                mat[s, j, k] = acc
    return mat


def plot_accuracy_over_seeds(
    runs: list[dict],
    title: str,
    out_path: Path,
    key: str = "re_eval_accs",
    labels: list[str] | None = None,
    ylabel: str = "KB test accuracy",
    baseline: list[dict] | None = None,
    baseline_title: str = "",
    smooth: int = 0,
):
    """One line per task: its KB accuracy after each later task. Band = std over seeds.
    With `baseline`, a second panel shows those runs on the same y axis."""
    panels = [(runs, title)] + ([(baseline, baseline_title)] if baseline else [])
    fig, axes = plt.subplots(
        1, len(panels), figsize=(8 * len(panels), 4.5), squeeze=False, sharey=True
    )
    for ax, (panel_runs, panel_title) in zip(axes[0], panels):
        _draw_accuracy_over_seeds(
            ax, panel_runs, panel_title, key, labels, ylabel, smooth
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _draw_accuracy_over_seeds(ax, runs, title, key, labels, ylabel, smooth=0):
    mat = _stack_re_eval(runs, key)  # [S, n, n]
    n = mat.shape[1]
    labels = labels or [f"Task {j}" for j in range(n)]
    with warnings.catch_warnings():  # the upper triangle is all-NaN by design
        warnings.simplefilter("ignore", RuntimeWarning)
        mean, std = np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)

    for j in range(n):
        stages = np.arange(j, n)
        color = COLORS[j % len(COLORS)]
        _plot_band(ax, stages, mean[j, j:], std[j, j:], color, labels[j], smooth)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"after T{k}" for k in range(n)], rotation=45, fontsize=8)
    ax.set_ylim(
        min(0, np.nanmin(mean - std) - 0.05), max(1.0, np.nanmax(mean + std) + 0.05)
    )
    ax.set_xlabel("Training stage")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{mat.shape[0]} seeds, band = std", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_loss_split_over_seeds(runs: list[dict], title: str, out_path: Path):
    """KL and EWC compress losses per epoch, one panel per task, mean over seeds with std band."""
    n = len(runs[0]["task_logs"])
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), squeeze=False, sharey=True)
    axes = axes[0]
    for k in range(n):
        ax = axes[k]
        for key, label, color in [
            ("compress_kl_losses", "KL (distillation)", COLORS[0]),
            ("compress_ewc_losses", "EWC (λ·Mahal.)", COLORS[1]),
        ]:
            arr = np.array([data["task_logs"][k][key] for data in runs])  # [S, epochs]
            mean, std = arr.mean(axis=0), arr.std(axis=0)
            epochs = np.arange(arr.shape[1])
            ax.plot(epochs, mean, color=color, label=label)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_title(f"Task {k}")
        ax.set_xlabel("Epoch")
        if k == 0:
            ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"{title}\ncompress loss split, mean over {len(runs)} seeds (band = std)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_seeds(
    run_dirs: list[Path],
    output_dir: Path,
    title: str,
    baseline_dirs: list[Path] | None = None,
    baseline_title: str = "",
    smooth: int = 0,
):
    runs = [d for d in (load_results(p) for p in run_dirs) if d is not None]
    if not runs:
        print(f"No results.json in {run_dirs}")
        return
    baseline = [
        d for d in (load_results(p) for p in baseline_dirs or []) if d is not None
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    if "re_eval_norm_returns" in runs[0]["task_logs"][0]:
        # Atari (src/run_atari.py): return per game, normalised to [random, active policy]
        labels = [
            f"{log['game']} (random {np.mean([r['task_logs'][j]['random_return'] for r in runs]):.1f}, "
            f"active {np.mean([r['task_logs'][j]['progress_return'] for r in runs]):.1f})"
            for j, log in enumerate(runs[0]["task_logs"])
        ]
        plot_accuracy_over_seeds(
            runs,
            title,
            output_dir / "return_per_game.png",
            key="re_eval_norm_returns",
            labels=labels,
            ylabel="KB return, 0 = random, 1 = active policy",
            baseline=baseline,
            baseline_title=baseline_title,
            smooth=smooth,
        )
        print(output_dir / "return_per_game.png")
    else:
        plot_accuracy_over_seeds(
            runs,
            title,
            output_dir / "accuracy_per_task.png",
            baseline=baseline,
            baseline_title=baseline_title,
            smooth=smooth,
        )
        print(output_dir / "accuracy_per_task.png")
    plot_loss_split_over_seeds(runs, title, output_dir / "compress_loss_split.png")
    print(output_dir / "compress_loss_split.png")


# ---------------------------------------------------------------------------
# Atari, the paper's figures: per-game KB return against total steps (Figure 4),
# forward transfer of the active column against the number of previous games (Table 1)
# ---------------------------------------------------------------------------


def _single_task_scores(single_dir: Path | None) -> dict[str, float]:
    """Mean greedy return of the single-task runs per game (`<game>_s<seed>/results.json`),
    the paper's normaliser. Empty when no directory is given."""
    scores: dict[str, list[float]] = {}
    for p in sorted(Path(single_dir).glob("*_s*")) if single_dir else []:
        data = load_results(p)
        if data is None:
            continue
        log = data["task_logs"][0]
        scores.setdefault(log["game"], []).append(log["progress_return"])
    return {g: float(np.mean(v)) for g, v in scores.items()}


def plot_paper_curves(
    named_runs: dict[str, list[dict]],
    single_dir: Path | None,
    title: str,
    out_path: Path,
    smooth: int = 0,
):
    """One panel per game: the KB return (0 = random, 1 = single-task score) after each
    task, against the cumulative PPO steps; one line per method, band = std over seeds;
    dashed vertical lines at the visits of the game."""
    first = next(iter(named_runs.values()))[0]["task_logs"]
    games = first[-1]["re_eval_games"]
    single = _single_task_scores(single_dir)
    fig, axes = plt.subplots(
        2,
        (len(games) + 1) // 2,
        figsize=(4.2 * ((len(games) + 1) // 2), 6.4),
        squeeze=False,
    )
    for ax, game in zip(axes.flat, games):
        for m, (name, runs) in enumerate(named_runs.items()):
            curves = []
            for data in runs:
                logs = data["task_logs"]
                random = next(t["random_return"] for t in logs if t["game"] == game)
                teacher = single.get(game) or max(
                    t["progress_return"] for t in logs if t["game"] == game
                )
                xs = [t["cumulative_steps"] for t in logs if game in t["re_eval_games"]]
                ys = [
                    (t["re_eval_returns"][t["re_eval_games"].index(game)] - random)
                    / (teacher - random)
                    for t in logs
                    if game in t["re_eval_games"]
                ]
                curves.append((xs, ys))
            n = min(len(c[1]) for c in curves)
            xs = np.array(curves[0][0][:n]) / 1e6
            arr = np.array([c[1][:n] for c in curves])
            visits = [t["cumulative_steps"] / 1e6 for t in first if t["game"] == game]
            breaks = [
                i for i, x in enumerate(xs) if i > 0 and x in visits
            ]  # no smoothing across a re-visit
            _plot_band(
                ax,
                xs,
                arr.mean(0),
                arr.std(0),
                COLORS[m % len(COLORS)],
                f"{name} ({len(runs)} seeds)",
                smooth,
                breaks=breaks,
            )
        for t in first:
            if t["game"] == game:
                ax.axvline(
                    t["cumulative_steps"] / 1e6,
                    color="gray",
                    linestyle="--",
                    linewidth=0.8,
                )
        ax.axhline(1.0, color="black", linewidth=0.6, alpha=0.5)
        ax.set_xlim(0, first[-1]["cumulative_steps"] / 1e6)
        ax.set_title(
            game + (f" (single-task {single[game]:.0f})" if game in single else ""),
            fontsize=9,
        )
        ax.set_xlabel("PPO steps (M)")
        ax.grid(True, alpha=0.3)
    fig.legend(
        *axes.flat[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=len(named_runs),
        fontsize=8,
    )
    fig.supylabel("KB return, 0 = random, 1 = single task", fontsize=10)
    fig.suptitle(
        f"{title}\ndashed = visits of the game, band = std over seeds", fontsize=10
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# AI-generated function
def plot_compress_curves(
    named_runs: dict[str, list[dict]],
    single_dir: Path | None,
    title: str,
    out_path: Path,
    n_bins: int = 20,
    smooth: int = 0,
):
    """One panel per game: the KB's return while it is being distilled (the paper's
    Figure 4), against the env steps of the compress phase, 0 = random and 1 = the
    single-task score. One line per method and visit (solid = visit 1, dashed = visit 2),
    the episodic returns binned into `n_bins` bins, band = std over seeds. Methods
    without `compress_curve` (the buffer compress) are skipped."""
    first = next(iter(named_runs.values()))[0]["task_logs"]
    games = first[-1]["re_eval_games"]
    single = _single_task_scores(single_dir)
    fig, axes = plt.subplots(
        2,
        (len(games) + 1) // 2,
        figsize=(4.2 * ((len(games) + 1) // 2), 6.4),
        squeeze=False,
    )
    for ax, game in zip(axes.flat, games):
        for m, (name, runs) in enumerate(named_runs.items()):
            for visit, style in ((0, "-"), (1, "--")):
                binned = []
                for data in runs:
                    logs = [t for t in data["task_logs"] if t["game"] == game]
                    if visit >= len(logs) or not logs[visit].get("compress_curve"):
                        continue
                    t = logs[visit]
                    random = t["random_return"]
                    teacher = single.get(game) or t["progress_return"]
                    curve = np.array(t["compress_curve"])  # [n_games, 2]: step, return
                    n_steps = curve[:, 0].max()
                    edges = np.linspace(0, n_steps, n_bins + 1)
                    idx = np.clip(
                        np.searchsorted(edges, curve[:, 0], side="right") - 1,
                        0,
                        n_bins - 1,
                    )
                    ys = np.array(
                        [
                            curve[idx == b, 1].mean() if (idx == b).any() else np.nan
                            for b in range(n_bins)
                        ]
                    )
                    binned.append((edges[1:] / 1e6, (ys - random) / (teacher - random)))
                if not binned:
                    continue
                xs = binned[0][0]
                arr = np.array([b[1] for b in binned])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean, std = np.nanmean(arr, 0), np.nanstd(arr, 0)
                _plot_band(
                    ax,
                    xs,
                    mean,
                    std,
                    COLORS[m % len(COLORS)],
                    f"{name}, visit {visit + 1} ({len(binned)} seeds)",
                    smooth,
                    style,
                    markersize=2,
                    alpha=0.15,
                )
        ax.axhline(1.0, color="black", linewidth=0.6, alpha=0.5)
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
        ax.set_title(
            game + (f" (single-task {single[game]:.0f})" if game in single else ""),
            fontsize=9,
        )
        ax.set_xlabel("compress steps (M)")
        ax.grid(True, alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=max(1, len(labels)), fontsize=8
    )
    fig.supylabel("KB return, 0 = random, 1 = single task", fontsize=10)
    fig.suptitle(
        f"{title}\nsolid = visit 1, dashed = visit 2, band = std over seeds",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def forward_transfer_table(
    named_runs: dict[str, list[dict]], single_dir: Path | None
) -> str:
    """The paper's Table 1: the active column's return on a game as % of the single-task
    score, against the number of games seen before it (first visits only)."""
    single = _single_task_scores(single_dir)
    lines = [
        "| method | " + " | ".join(f"{k} previous" for k in range(6)) + " |",
        "|---|" + "---|" * 6,
    ]
    for name, runs in named_runs.items():
        cells = []
        for k in range(6):
            vals = [
                100 * t["progress_return"] / single[t["game"]]
                for data in runs
                for t in data["task_logs"]
                if not t["revisit"]
                and t["task_id"] == k
                and t["game"] in single
                and single[t["game"]] != 0
            ]
            cells.append(f"{np.mean(vals):.0f} ± {np.std(vals):.0f}" if vals else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_paper(
    method_dirs: list[str],
    single_dir: Path | None,
    output_dir: Path,
    title: str,
    smooth: int = 0,
):
    """`method_dirs` entries are `name=dir1,dir2,...` (one run directory per seed)."""
    named_runs = {}
    for entry in method_dirs:
        name, dirs = entry.split("=", 1)
        named_runs[name] = [
            d for d in (load_results(Path(p)) for p in dirs.split(",")) if d is not None
        ]
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paper_curves(
        named_runs, single_dir, title, output_dir / "paper_curves.png", smooth=smooth
    )
    print(output_dir / "paper_curves.png")
    if any(
        t.get("compress_curve")
        for runs in named_runs.values()
        for d in runs
        for t in d["task_logs"]
    ):
        plot_compress_curves(
            named_runs,
            single_dir,
            title,
            output_dir / "paper_compress_curves.png",
            smooth=smooth,
        )
        print(output_dir / "paper_compress_curves.png")
    table = forward_transfer_table(named_runs, single_dir)
    (output_dir / "forward_transfer.md").write_text(table + "\n")
    print(output_dir / "forward_transfer.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot training curves from results.json"
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("outputs/experiments"),
        help="Run dir (with results.json) or sweep dir (with subdirs)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Output directory for plots"
    )
    parser.add_argument(
        "--seeds",
        type=Path,
        nargs="+",
        default=None,
        help="Run dirs of the same configuration, one per seed; plots with std bands",
    )
    parser.add_argument(
        "--title", default="", help="Configuration, written in the plot title"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        nargs="+",
        default=None,
        help="Run dirs of a baseline configuration, drawn in a second panel",
    )
    parser.add_argument(
        "--baseline-title", default="", help="Title of the baseline panel"
    )
    parser.add_argument(
        "--paper",
        nargs="+",
        default=None,
        help="Atari, the paper's figures: entries name=dir1,dir2,... one run dir per seed",
    )
    parser.add_argument(
        "--single",
        type=Path,
        default=None,
        help="Sweep dir of the single-task runs (<game>_s<seed>), the normaliser of --paper",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="Window of a centered moving average on each line and band; 0 = off",
    )
    args = parser.parse_args()

    if args.paper:
        plot_paper(
            args.paper,
            args.single,
            args.out or Path("outputs/plots/atari/paper"),
            args.title,
            args.smooth,
        )
    elif args.seeds:
        plot_seeds(
            args.seeds,
            args.out or Path("outputs/plots/final"),
            args.title,
            args.baseline,
            args.baseline_title,
            args.smooth,
        )
    elif load_results(args.path) is not None:
        out = args.out or args.path / "plots"
        plot_run(args.path, out)
    else:
        out = args.out or args.path.parent / "plots" / args.path.name
        plot_sweep(args.path, out)
