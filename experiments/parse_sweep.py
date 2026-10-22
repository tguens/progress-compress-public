"""
AI-generated script to parse results of hyperparameter sweeps and print a summary table."""

import argparse
import gzip
import json
from pathlib import Path


def load_results(sweep_dir: str = "outputs/experiments") -> list[dict]:
    rows = []
    paths = list(Path(sweep_dir).glob("*/results.json"))
    # An archived run keeps only the gzip file (gzipped above 5 MB)
    paths += [
        p
        for p in Path(sweep_dir).glob("*/results.json.gz")
        if not p.with_suffix("").exists()
    ]
    for p in sorted(paths):
        with gzip.open(p, "rt") if p.suffix == ".gz" else open(p) as f:
            data = json.load(f)
        cfg = data["config"]
        logs = data["task_logs"]

        last = logs[-1]
        n_tasks = len(logs)
        # Atari runs log normalised returns instead of accuracies.
        final_accs = last.get("re_eval_accs", last.get("re_eval_norm_returns"))
        # Task index of the first NaN in the compress losses, or None.
        nan_task = next(
            (t["task_id"] for t in logs if any(x != x for x in t["compress_losses"])),
            None,
        )

        rows.append(
            {
                "tag": p.parent.name,
                "teacher": cfg.get("compress_teacher", "frozen_teacher"),
                "ewc_lambda": cfg["ewc_lambda"],
                "norm_fisher": cfg["normalize_fisher"],
                "compress_lr": cfg["compress_lr"],
                "reinit": cfg.get("reinit", True),
                # Key metrics
                "current_acc": final_accs[-1],
                "prev_avg_acc": sum(final_accs[:-1]) / max(len(final_accs) - 1, 1),
                "all_avg_acc": sum(final_accs) / len(final_accs),
                "n_tasks": n_tasks,
                "nan_task": nan_task,
                # Per-task detail
                "final_accs": final_accs,
                "kb_train_accs": [t.get("kb_train_acc") for t in logs],
                "progress_accs": [
                    t.get("progress_eval_acc", t.get("progress_return")) for t in logs
                ],
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'tag':<45} {'teacher':<22} {'λ':>6} {'nf':>4} {'clr':>6} {'ri':>4} "
        f"{'curr':>5} {'prev':>5} {'all':>5} {'nan':>4}  final_accs"
    )
    print(header)
    print("-" * len(header) + "----------")

    for r in sorted(rows, key=lambda x: -x["all_avg_acc"]):
        accs_str = " ".join(f"{a:.3f}" for a in r["final_accs"])
        print(
            f"{r['tag']:<45} {r['teacher']:<22} {r['ewc_lambda']:>6.0f} "
            f"{'Y' if r['norm_fisher'] else 'N':>4} {r['compress_lr']:>6.0e} "
            f"{'Y' if r['reinit'] else 'N':>4} "
            f"{r['current_acc']:>5.3f} {r['prev_avg_acc']:>5.3f} {r['all_avg_acc']:>5.3f} "
            f"{'T' + str(r['nan_task']) if r['nan_task'] is not None else '-':>4}  "
            f"{accs_str}"
        )


# AI-generated function
def print_seed_table(rows: list[dict]) -> None:
    """Group tags that differ only in a `_s<seed>` suffix. Mean and std over seeds."""
    import re
    from statistics import mean, pstdev

    groups: dict[str, list[dict]] = {}
    for r in rows:
        key = re.sub(r"_s\d+$", "", r["tag"])
        groups.setdefault(key, []).append(r)

    header = f"{'config':<45} {'n':>2} {'nan':>3} {'prev':>13} {'all':>13} {'curr':>13}"
    print(header)
    print("-" * len(header))

    def _ms(xs):
        return f"{mean(xs):.3f}±{pstdev(xs):.3f}"

    for key, rs in sorted(
        groups.items(), key=lambda kv: -mean(r["all_avg_acc"] for r in kv[1])
    ):
        n_nan = sum(r["nan_task"] is not None for r in rs)
        print(
            f"{key:<45} {len(rs):>2} {n_nan:>3} "
            f"{_ms([r['prev_avg_acc'] for r in rs]):>13} "
            f"{_ms([r['all_avg_acc'] for r in rs]):>13} "
            f"{_ms([r['current_acc'] for r in rs]):>13}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse sweep results")
    parser.add_argument(
        "sweep_dir",
        nargs="?",
        default="outputs/experiments",
        help="Directory containing run subdirs with results.json",
    )
    parser.add_argument(
        "--seeds",
        action="store_true",
        help="Group tags by config (strip _s<seed>) and report mean±std",
    )
    args = parser.parse_args()

    rows = load_results(args.sweep_dir)
    if not rows:
        print(f"No results found in {args.sweep_dir}/")
    else:
        print(f"\n{len(rows)} runs found\n")
        if args.seeds:
            print_seed_table(rows)
        else:
            print_table(rows)
        print(
            "\nLegend: curr = last task acc, prev = mean acc on earlier tasks, "
            "all = mean over all tasks, nan = first task with NaN compress loss"
        )
        print("        nf = normalize_fisher, ri = reinit, λ = ewc_lambda")
