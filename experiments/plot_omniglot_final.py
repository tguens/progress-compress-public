"""AI-generated script: plot the archived Omniglot comparison without smoothing.

Run from the repository root: uv run python -m experiments.plot_omniglot_final
"""

import json
from pathlib import Path

from src.viz import plot_accuracy_over_seeds


if __name__ == "__main__":
    source = Path("reference_outputs/experiments/final")
    runs = [
        json.loads((source / f"best_s{seed}" / "results.json").read_text())
        for seed in range(3)
    ]
    baseline = [
        json.loads((source / f"ft_lam0_s{seed}" / "results.json").read_text())
        for seed in range(3)
    ]
    output = Path("outputs/plots/final/omniglot_pc_vs_distillation.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_accuracy_over_seeds(
        runs,
        title="P&C, EWC λ = 500",
        out_path=output,
        baseline=baseline,
        baseline_title="Distillation without EWC, λ = 0",
        smooth=0,
    )
    print(output)
