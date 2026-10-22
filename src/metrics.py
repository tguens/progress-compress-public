"""Per-epoch statistics of one training phase, the evaluation (`evaluate`), the task entry
of results.json, and the Weights & Biases `Logger`."""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Literal

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


# ------------------------------- #
#       Monitoring
# ------------------------------- #
@dataclass
class Stats:
    """Records of one phase (progress or compress) of one task.

    - `losses[name][epoch]`: the per-batch values of a loss term, for example
      `losses["loss"][3]` or `losses["kl"][3]`.
    - `metrics[name][epoch]`: one `evaluate()` dict, for example `metrics["eval"][5]`.

    AI-generated
    """

    losses: Dict[str, Dict[int, list[float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    metrics: Dict[str, Dict[int, dict]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    def acc(
        self, terms: str | Dict[str, float], value: float | None = None, *, epoch: int
    ):
        """Accumulate per-batch loss values: `acc("loss", 0.3, epoch=2)` for one
        term, or `acc({"kl": 0.2, "ewc": 0.1}, epoch=2)` for several at once."""
        if isinstance(terms, str):
            self.losses[terms][epoch].append(value)
        else:
            for name, v in terms.items():
                self.losses[name][epoch].append(v)

    def record(self, name: str, metrics: dict, *, epoch: int):
        """Store one evaluation of the epoch: `record("eval", evaluate(...), epoch=5)`."""
        self.metrics[name][epoch] = metrics

    def epoch_values(self, name: str) -> list[float]:
        """Every recorded value of a term, in epoch order (one per PPO iteration)."""
        return [v for _, vs in sorted(self.losses[name].items()) for v in vs]

    def epoch_mean(self, name: str) -> list[float]:
        """Mean of a loss term per epoch, in epoch order. Empty if never recorded."""
        return [sum(v) / len(v) for _, v in sorted(self.losses[name].items())]

    def epoch_metric(self, name: str, key: str = "accuracy") -> list[float]:
        """One key of an evaluation per recorded epoch, in epoch order."""
        return [m[key] for _, m in sorted(self.metrics[name].items())]

    def last_metric(self, name: str, key: str = "accuracy") -> float | None:
        """The key of the latest evaluation, or None if none was recorded."""
        if not self.metrics[name]:
            return None
        return self.metrics[name][max(self.metrics[name])][key]

    def postfix(self, epoch: int) -> str:
        """One-line summary for the progress bar: the epoch's mean loss, the latest
        train and eval accuracy, and the KL and EWC terms when they exist."""
        parts = [f"loss={self.epoch_mean('loss')[-1]:.4f}"]
        for name, label in (("train", "tr_acc"), ("eval", "ev_acc")):
            acc = self.last_metric(name)
            parts.append(f"{label}={acc if acc is not None else float('nan'):.3f}")
        for name in ("kl", "ewc"):
            if epoch in self.losses[name]:
                values = self.losses[name][epoch]
                parts.append(f"{name}={sum(values) / len(values):.4f}")
        return " | ".join(parts)


def task_log(
    task_id: int,
    progress_stats: Stats,
    compress_stats: Stats,
    kb_train_metrics: dict,
    fisher_stats: dict,
    running_fisher: Dict[str, torch.Tensor],
    accs: list[float],
) -> dict:
    """One entry of `task_logs` in results.json: the per-epoch curves of both
    phases, the Fisher summary and the re-evaluation accuracies. The keys are
    the ones `experiments/parse_sweep.py` and `src/viz.py` read."""
    return {
        "task_id": task_id,
        "progress_eval_acc": progress_stats.last_metric("eval"),
        "kb_train_acc": kb_train_metrics["accuracy"],
        **fisher_stats,
        "running_fisher_total": sum(v.sum().item() for v in running_fisher.values()),
        "running_fisher_per_layer": {
            k: float(v.sum()) for k, v in running_fisher.items()
        },
        "re_eval_accs": accs,
        "progress_losses": progress_stats.epoch_mean("loss"),
        "progress_train_accs": progress_stats.epoch_metric("train"),
        "progress_eval_accs": progress_stats.epoch_metric("eval"),
        "compress_losses": compress_stats.epoch_mean("loss"),
        "compress_kl_losses": compress_stats.epoch_mean("kl"),
        "compress_ewc_losses": compress_stats.epoch_mean("ewc"),
        "compress_train_accs": compress_stats.epoch_metric("train"),
        "compress_eval_accs": compress_stats.epoch_metric("eval"),
    }


# ------------------------------- #
#       Eval
# ------------------------------- #


@torch.no_grad()
def evaluate(
    model,
    loader: Iterable,
    mode: Literal["progress", "compress"],
    device: torch.device,
) -> dict:
    """Run model.eval() over loader, then full metrics on concatenated logits."""
    was_training = model.training
    model.eval()
    try:
        all_logits, all_labels = [], []
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            active_logits, kb_logits = model(imgs, mode=mode)
            logits = active_logits if mode == "progress" else kb_logits
            all_logits.append(logits)
            all_labels.append(labels)
        return _get_progress_metrics(torch.cat(all_logits), torch.cat(all_labels))
    finally:
        if was_training:
            model.train()


# AI-generated function
@torch.no_grad()
def _get_progress_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Summary metrics for (N, C) logits vs (N,) labels.

    The class set is inferred from labels.unique(), so it works for both dense
    (0..n-1) and sparse (e.g. {0, 4, 8, 12}) label encodings. n_classes is the
    number of distinct classes actually present in this pass; per-class dicts
    are keyed by the real class IDs.
    """
    losses = F.cross_entropy(logits, labels, reduction="none")
    preds = logits.argmax(dim=1)
    correct = preds == labels

    cls = sorted(int(c) for c in labels.unique().tolist())
    num_classes = len(cls)

    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()

    out = {
        "n_classes": num_classes,
        "loss": float(losses.mean()),
        "accuracy": float(correct.float().mean()),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", labels=cls, zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", labels=cls, zero_division=0)
        ),
        "loss_per_class": {},
        "accuracy_per_class": {},
        "n_samples_per_class": {},
    }
    for c in cls:
        mask = labels == c
        n = int(mask.sum())
        out["n_samples_per_class"][c] = n
        out["loss_per_class"][c] = float(losses[mask].mean()) if n else float("nan")
        out["accuracy_per_class"][c] = (
            float(correct[mask].float().mean()) if n else float("nan")
        )
    return out


# ------------------------------- #
#       Weights & Biases
# ------------------------------- #
class Logger:
    """One wandb run per Hydra run directory, or a no-op when `enabled` is false.
    `log(data, step)` sends one row with `global_step` = the PPO steps over all
    tasks. `compress_epoch` and `online_step` are the x axes of `compress/*` and
    `compress_online/*`, counted over the whole run."""

    def __init__(
        self,
        enabled: bool,
        project: str,
        entity: str | None,
        output_dir: str,
        config: dict,
        tags: list[str],
    ):
        self.run = None
        self.compress_epoch = 0
        self.online_step = 0
        if not enabled:
            return
        import wandb

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=Path(output_dir).name,  # the tag
            group=Path(output_dir).parent.name,  # the sweep
            tags=tags,
            config=config,
            dir=output_dir,
        )
        self.run.define_metric("global_step")
        self.run.define_metric("*", step_metric="global_step")
        self.run.define_metric("compress/*", step_metric="compress/epoch")
        self.run.define_metric("compress_online/*", step_metric="compress_online/step")

    def log(self, data: dict, step: int) -> None:
        if self.run is not None:
            self.run.log({**data, "global_step": step})

    def finish(self, results_path: str | None = None) -> None:
        if self.run is None:
            return
        if results_path is not None:
            self.run.save(results_path, base_path=str(Path(results_path).parent))
        self.run.finish()


def task_log_atari(
    task_id: int,
    task,
    revisit: bool,
    cumulative_steps: int,
    random_return: float,
    progress_eval: dict,
    progress_curve: list,
    progress_stats: Stats,
    compress_stats: Stats,
    compress_curve: list,
    fisher_stats: dict,
    running_fisher: Dict[str, torch.Tensor],
    re_eval: dict,
    wall_time_s: float,
) -> dict:
    """One entry of `task_logs` in results.json for Atari. The keys are the ones
    `experiments/parse_sweep.py` and `src/viz.py --paper` read."""
    return {
        "task_id": task_id,
        "game": task.game,
        "revisit": revisit,
        "cumulative_steps": cumulative_steps,
        "n_actions": task.n_actions,
        "random_return": random_return,
        "progress_return": progress_eval["mean_return"],
        "progress_return_std": progress_eval["std_return"],
        "progress_curve": progress_curve,
        "progress_losses": {
            k: progress_stats.epoch_values(k) for k in progress_stats.losses
        },
        **fisher_stats,
        "running_fisher_per_layer": {
            k: float(v.sum()) for k, v in running_fisher.items()
        },
        **re_eval,
        "compress_losses": compress_stats.epoch_mean("loss"),
        "compress_kl_losses": compress_stats.epoch_mean("kl"),
        "compress_ewc_losses": compress_stats.epoch_mean("ewc"),
        "compress_curve": compress_curve,
        "wall_time_s": wall_time_s,
    }
