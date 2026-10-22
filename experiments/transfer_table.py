"""AI-generated script: the Omniglot transfer table. Per task
and per seed, it compares the active column that reads a learned KB (the normal P&C run,
`transfer_learned_s<seed>`) with the active column that reads a frozen random KB (the
control, `kb_random=true`, `transfer_random_s<seed>`). It reads the run directories of
`experiments/omniglot_transfer.sh` and trains nothing.

Run from the repository root:
    uv run --no-sync python -m experiments.transfer_table [RUNS_DIR] [--out DIR] [--plots DIR]
RUNS_DIR defaults to outputs/experiments/final.

Data: `progress_eval_accs` of each task, the test accuracy of the active column during the
progress phase of the task. Entry i is the accuracy after `progress_eval_epochs[i]`
completed epochs (1 = after the first epoch). The sweep evaluates at the loop indices
0, 5, ..., 395 (`eval_every_progress=5`, after the completed epochs 1, 6, ..., 396) and
after the completed epochs 25, 50, 100, 200 and 400 (`eval_epochs_progress`). The archived
runs have no `progress_eval_epochs`. The script does not read them and does not relabel
their positions as these epochs.

Pairing: both variants of a seed draw the new columns, the heads, the progress data order
and the augmentation of task k from the stream of (seed, k) (`src/run_omniglot.py`). At
task 0 neither KB has knowledge: both are the KB of model construction. So the two task-0
curves agree, exactly on the CPU, and up to non-deterministic kernels on the GPU.

Metrics.
- d_k(e) = acc_learned_k(e) - acc_random_k(e): the paired difference of one seed on task k
  (position k in the task order) after e completed epochs. d > 0: the learned KB helps.
- learning speed: d_k(50). Final accuracy: d_k(400). They are reported separately. There
  is no combined success test and no count-of-tasks rule.
- per-seed summary: the mean of d_k(e) over the tasks 1 to n-1 of one seed. Task 0 is the
  agreement check, not a transfer measurement. The tasks are not independent seeds: the
  spread is over the per-seed summaries.
- task 0 agreement: max over the common recorded epochs e of |acc_learned_0(e) - acc_random_0(e)|.
  On the GPU it is the noise floor of d from non-deterministic kernels.
- sign over seeds: "positive in every seed", "negative in every seed", or "mixed across
  seeds (inconclusive)", from the per-seed summaries. It describes the seeds. It is not a
  significance test.
Mean ± std is over seeds, std with ddof=0 (population). Per-seed values are in brackets,
in seed order. A pair is read only if both runs exist, have `progress_eval_epochs`, and the
learned run has finite compress losses (a divergent run is a failed run).
"""

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.viz import COLORS, load_results

SEEDS = (0, 1, 2, 3)  # seed 3 replaces the seed 0 pair, whose learned run diverged at T3
VARIANTS = ("learned", "random")  # tags transfer_<variant>_s<seed>
FIXED_EPOCHS = (25, 50, 100, 200, 400)  # completed epochs evaluated in every run (E7)
SPEED_EPOCH, FINAL_EPOCH = 50, 400


# AI-generated function
def load_runs(runs_dir: Path) -> tuple[dict, list[str]]:
    """runs[(variant, seed)]: the results dict with the run directory and the device of
    run.log. `missing` lists the runs that are not read, with the reason."""
    runs, missing = {}, []
    for seed in SEEDS:
        for variant in VARIANTS:
            run_dir = runs_dir / f"transfer_{variant}_s{seed}"
            data = load_results(run_dir)
            if data is None:
                missing.append(f"{run_dir}: no results.json")
                continue
            if "progress_eval_epochs" not in data["task_logs"][0]:
                missing.append(f"{run_dir}: no progress_eval_epochs (archived format)")
                continue
            log = run_dir / "run.log"
            found = (
                re.search(r"Using device: (\S+)", log.read_text())
                if log.exists()
                else None
            )
            runs[(variant, seed)] = {
                **data,
                "dir": run_dir,
                "device": found.group(1) if found else "unknown (no run.log)",
            }
    return runs, missing


def first_nonfinite_task(data: dict) -> int | None:
    """The first task with a non-finite compress loss, None if every loss is finite."""
    for t in data["task_logs"]:
        if any(not math.isfinite(x) for x in t["compress_losses"]):
            return t["task_id"]
    return None


def curve(data: dict, k: int) -> dict[int, float]:
    """{completed epoch: test accuracy} of the active column on task k."""
    t = data["task_logs"][k]
    return dict(zip(t["progress_eval_epochs"], t["progress_eval_accs"]))


# AI-generated function
def pair_checks(learned: dict, random: dict) -> dict:
    """The configuration keys that differ other than kb_random, and whether the two runs
    have the same alphabets in the same order."""
    cl, cr = learned["config"], random["config"]
    keys = sorted(
        k for k in set(cl) | set(cr) if k != "kb_random" and cl.get(k) != cr.get(k)
    )
    same_order = [t["alphabet"] for t in learned["task_logs"]] == [
        t["alphabet"] for t in random["task_logs"]
    ]
    return {"config_diff": keys, "same_order": same_order}


# AI-generated function
def paired_rows(runs: dict, seeds: list[int]) -> list[dict]:
    """One row per seed, task and epoch recorded in both runs of the pair."""
    rows = []
    for seed in seeds:
        learned, random = runs[("learned", seed)], runs[("random", seed)]
        n_tasks = min(len(learned["task_logs"]), len(random["task_logs"]))
        for k in range(n_tasks):
            cl, cr = curve(learned, k), curve(random, k)
            for e in sorted(set(cl) & set(cr)):
                rows.append(
                    {
                        "seed": seed,
                        "task": k,
                        "alphabet": learned["task_logs"][k]["alphabet"],
                        "epoch": e,
                        "fixed_epoch": e in FIXED_EPOCHS,
                        "learned_acc": cl[e],
                        "random_acc": cr[e],
                        "diff": cl[e] - cr[e],
                    }
                )
    return rows


def value(rows: list[dict], seed: int, k: int, e: int, key: str = "diff") -> float:
    """The value of one seed, task and epoch, NaN when the epoch is not recorded."""
    found = [
        r[key] for r in rows if r["seed"] == seed and r["task"] == k and r["epoch"] == e
    ]
    return found[0] if found else np.nan


def cell(values: list[float], signed: bool = True) -> str:
    """'mean ± std [per seed]', std with ddof=0; 'n.a.' when a value is missing."""
    a = np.asarray(values, dtype=float)
    fmt = "{:+.3f}" if signed else "{:.3f}"
    seeds = (
        "["
        + ", ".join("n.a." if not np.isfinite(x) else fmt.format(x) for x in a)
        + "]"
    )
    if a.size == 0 or not np.isfinite(a).all():
        return f"n.a. {seeds}"
    return f"{fmt.format(a.mean())} ± {a.std(ddof=0):.3f} {seeds}"


def sign_over_seeds(values: list[float]) -> str:
    a = np.asarray(values, dtype=float)
    if a.size == 0 or not np.isfinite(a).all():
        return "not available"
    if (a > 0).all():
        return f"positive in every seed ({a.size} of {a.size})"
    if (a < 0).all():
        return f"negative in every seed ({a.size} of {a.size})"
    return f"mixed across seeds (inconclusive): {(a > 0).sum()} positive, {(a < 0).sum()} negative, {(a == 0).sum()} zero"


# AI-generated function
def markdown(
    runs: dict,
    rows: list[dict],
    seeds: list[int],
    checks: dict,
    missing: list[str],
    runs_dir: Path,
) -> str:
    tasks = sorted({(r["task"], r["alphabet"]) for r in rows})
    later = [k for k, _ in tasks if k >= 1]
    lines = [
        "# Omniglot transfer: learned KB against a frozen random KB",
        "",
        f"Runs: `{runs_dir}/transfer_{{learned,random}}_s<seed>`, paired seeds {seeds}. "
        "Test accuracy of the active column after e completed epochs of the progress phase. "
        "d = learned − random, per seed and task. Mean ± std over seeds (ddof=0), per-seed values in brackets. "
        "k = the position in the task order. Definitions: the header of `experiments/transfer_table.py`.",
        "",
        "## Runs",
        "",
        "| run | seed | kb_random | device | tasks | first task with a non-finite compress loss |",
        "|---|---|---|---|---|---|",
    ]
    for (variant, seed), data in sorted(runs.items(), key=lambda x: (x[0][1], x[0][0])):
        nonfinite = first_nonfinite_task(data)
        status = (
            "no compress (control)"
            if data["config"].get("kb_random")
            else ("none" if nonfinite is None else f"T{nonfinite} (failed run)")
        )
        lines.append(
            f"| {data['dir'].name} | {seed} | {data['config'].get('kb_random', False)} | {data['device']} "
            f"| {len(data['task_logs'])} | {status} |"
        )
    lines += ["", "## Pairing checks", ""]
    for seed in seeds:
        c = checks[seed]
        lines.append(
            f"- seed {seed}: configuration differs only in kb_random: "
            f"{'yes' if not c['config_diff'] else 'no, also ' + ', '.join(c['config_diff'])}; "
            f"same alphabets in the same order: {'yes' if c['same_order'] else 'no'}"
        )
    lines += ["", "## Task 0 agreement (neither KB has knowledge at task 0)", ""]
    for seed in seeds:
        d0 = [abs(r["diff"]) for r in rows if r["seed"] == seed and r["task"] == 0]
        worst = max(d0) if d0 else np.nan
        lines.append(
            f"- seed {seed}, T0 {tasks[0][1] if tasks else ''}: max |learned − random| over {len(d0)} "
            f"common epochs = {worst:.6g}{' (identical)' if worst == 0 else ''}"
        )
    reading = []
    for e, name in ((SPEED_EPOCH, "Learning speed"), (FINAL_EPOCH, "Final accuracy")):
        summary = [
            np.mean([value(rows, s, k, e) for k in later]) if later else np.nan
            for s in seeds
        ]
        lines += [
            "",
            f"## {name}: accuracy after {e} completed epochs",
            "",
            "| k | alphabet | learned | random | d = learned − random |",
            "|---|---|---|---|---|",
        ]
        for k, alphabet in tasks:
            lines.append(
                f"| {k} | {alphabet} "
                f"| {cell([value(rows, s, k, e, 'learned_acc') for s in seeds], signed=False)} "
                f"| {cell([value(rows, s, k, e, 'random_acc') for s in seeds], signed=False)} "
                f"| {cell([value(rows, s, k, e) for s in seeds])} |"
            )
        lines += [
            f"| 1 to {later[-1] if later else '-'} | per-seed mean over the tasks | | | {cell(summary)} |",
            "",
            f"Sign over seeds of the per-seed mean d({e}), tasks 1 to {later[-1] if later else '-'}: {sign_over_seeds(summary)}.",
        ]
        reading.append(
            f"- {name}, d({e}): per-seed mean {cell(summary)}, {sign_over_seeds(summary)}."
        )
    lines += [
        "",
        "## Paired difference d at the fixed epochs",
        "",
        "| k | alphabet | " + " | ".join(f"d({e})" for e in FIXED_EPOCHS) + " |",
        "|---|---|" + "---|" * len(FIXED_EPOCHS),
    ]
    for k, alphabet in tasks:
        lines.append(
            f"| {k} | {alphabet} | "
            + " | ".join(
                cell([value(rows, s, k, e) for s in seeds]) for e in FIXED_EPOCHS
            )
            + " |"
        )
    lines += [
        "",
        "## Reading",
        "",
        *reading,
        "- These are two separate measurements. A mixed sign is an inconclusive result for that measurement.",
        "",
        "## Runs not read",
        "",
    ] + [f"- {m}" for m in missing or ["none"]]
    return "\n".join(lines) + "\n"


# AI-generated function
def plot(rows: list[dict], seeds: list[int], title: str, out_path: Path) -> None:
    """One panel per task: d against the completed epoch, the mean over seeds with a ±std
    band, and each seed's paired difference as a thin line."""
    tasks = sorted({(r["task"], r["alphabet"]) for r in rows})
    ncols = min(5, len(tasks))
    nrows = math.ceil(len(tasks) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 2.6 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, (k, alphabet) in zip(axes.flat, tasks):
        color = COLORS[k % len(COLORS)]
        epochs = sorted({r["epoch"] for r in rows if r["task"] == k})
        d = np.array(
            [[value(rows, s, k, e) for e in epochs] for s in seeds]
        )  # [S, n_epochs]
        mean, std = d.mean(axis=0), d.std(axis=0, ddof=0)
        for s, ds in zip(seeds, d):
            ax.plot(
                epochs,
                ds,
                color=color,
                alpha=0.35,
                linewidth=0.8,
                label="per seed" if s == seeds[0] else None,
            )
        ax.plot(epochs, mean, color=color, label="mean over seeds")
        ax.fill_between(
            epochs,
            mean - std,
            mean + std,
            color=color,
            alpha=0.2,
            label="± std (ddof=0)",
        )
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(f"T{k} {alphabet}", fontsize=8)
    axes.flat[0].legend(fontsize=6)
    for ax in axes.flat[len(tasks) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("completed epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("d = learned − random")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "runs_dir", nargs="?", type=Path, default=Path("outputs/experiments/final")
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--plots", type=Path, default=Path("outputs/plots/final"))
    args = parser.parse_args()

    runs, missing = load_runs(args.runs_dir)
    seeds, checks = [], {}
    for seed in SEEDS:
        if ("learned", seed) not in runs or ("random", seed) not in runs:
            continue
        nonfinite = first_nonfinite_task(runs[("learned", seed)])
        if nonfinite is not None:
            missing.append(
                f"seed {seed}: the learned run diverged at T{nonfinite}, the pair is not read"
            )
            continue
        seeds.append(seed)
        checks[seed] = pair_checks(runs[("learned", seed)], runs[("random", seed)])
    if not seeds:
        raise SystemExit(
            "No complete pair of runs in "
            + str(args.runs_dir)
            + ":\n"
            + "\n".join(missing)
        )

    rows = paired_rows(runs, seeds)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "omniglot_transfer.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = markdown(runs, rows, seeds, checks, missing, args.runs_dir)
    (args.out / "omniglot_transfer.md").write_text(table)
    cfg = runs[("learned", seeds[0])]["config"]
    title = (
        f"Active column test accuracy, learned KB − frozen random KB, seeds {seeds}\n"
        f"{cfg['max_n_tasks']} tasks, progress {cfg['progress_optimizer']} lr {cfg['progress_lr']} × {cfg['n_epochs_progress']} epochs, "
        f"compress {cfg['compress_optimizer']} lr {cfg['compress_lr']} × {cfg['n_epochs_compress']}, λ {cfg['ewc_lambda']}, clip {cfg['fisher_clip']}"
    )
    plot(rows, seeds, title, args.plots / "omniglot_transfer.png")
    print(table)
    print(args.out / "omniglot_transfer.md")
    print(args.out / "omniglot_transfer.csv")
    print(args.plots / "omniglot_transfer.png")
