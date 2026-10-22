"""Simple Progress & Compress implementation

Not 100% faithful (see README.md), but rather a simple entrypoint
to study the wake/sleep cycle, and task adaptation (with weight regularization)

One task = one Omniglot alphabet. For each task:
1. Progress: train the active column and the adaptors on the task labels.
2. Compress: distill the active column into the knowledge base (KB), with
   the EWC penalty on the running Fisher of the KB.
3. Fisher: add the diagonal Fisher of the KB on the task to the running Fisher,
   then re-evaluate the KB on every task seen so far.

Control `kb_random=true`: the active column reads the frozen
random KB of model construction. Steps 2 and 3 are skipped. The re-evaluation still runs,
on the random KB, so results.json keeps its shape. The new columns, the heads, the progress
data order and its augmentation of task k draw from a random stream of (seed, k), so a
control run and a normal run of the same seed are paired.
"""

import copy
import json
from collections import defaultdict
from typing import Callable, Dict, Literal, Sequence

import hydra
import torch
import torch.nn as nn
import tqdm
from omegaconf import DictConfig, OmegaConf

from environments.omniglot import AlphabetTask, load_tasks, max_classes
from metrics import Stats, evaluate, task_log
from model import CNNModel
from utils import (
    accumulate_fisher,
    create_frozen_model,
    create_optimizer,
    distill_step,
    task_fisher,
)


# ---------------------------------
# Shared across progress / compress
# ---------------------------------
# Generic training loop
def _run_epochs(
    model,
    task: AlphabetTask,
    task_id: int,
    n_epochs: int,
    optimizer: torch.optim.Optimizer,
    eval_every: int,
    mode: Literal["progress", "compress"],
    step_fn: Callable[[torch.Tensor, torch.Tensor, int], dict],
    device: torch.device,
    eval_epochs: Sequence[int] = (),
) -> Stats:
    model.train()

    desc = (
        " ---> Progress " if mode == "progress" else " &&& Compress "
    ) + f" task {task_id}"
    tqdm_epochs = tqdm.trange(n_epochs, desc=desc)
    stats = Stats()

    for epoch in tqdm_epochs:
        # With shuffle=True, each new DataLoader should get a reshuffled minibatch order
        train_loader = task.train_loader(shuffle=True)

        for img_batch, label_batch in train_loader:
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            stats.acc(step_fn(img_batch, label_batch, epoch), epoch=epoch)

        # Evaluate the model every eval_every epochs (loop index 0, eval_every, ...),
        # and after the completed epochs in eval_epochs (1 = after the first epoch)
        if epoch % eval_every == 0 or epoch + 1 in eval_epochs:
            for name, loader in (
                ("train", task.train_loader(shuffle=False)),
                ("eval", task.test_loader(shuffle=False)),
            ):
                stats.record(
                    name, evaluate(model, loader, mode=mode, device=device), epoch=epoch
                )

        # Summary (eval-derived accs only change every `eval_every` epochs)
        tqdm_epochs.set_postfix_str(stats.postfix(epoch))

    return stats


# --------------------------------
#       ----> Progress
# --------------------------------
def progress(
    model, task: AlphabetTask, task_id: int, cfg: DictConfig, device: torch.device
):
    """Progress: train active column + adaptors on the task labels. KB frozen."""
    optimizer = create_optimizer(
        cfg.progress_optimizer,
        params=model.parameters_for("progress"),
        lr=cfg.progress_lr,
        momentum=cfg.sgd_momentum,  # Only used if progress_optimizer is sgd
    )

    # Loss: cross-entropy of the active column on the task labels
    ce_loss_fn = nn.CrossEntropyLoss()

    def step_fn(img_batch, label_batch, epoch):
        y_active, _ = model(img_batch, mode="progress")
        loss = ce_loss_fn(y_active, label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    stats = _run_epochs(
        model,
        task,
        task_id,
        n_epochs=cfg.n_epochs_progress,
        optimizer=optimizer,
        eval_every=cfg.eval_every_progress,
        mode="progress",
        step_fn=step_fn,
        device=device,
        eval_epochs=cfg.eval_epochs_progress,
    )
    return model, stats


# --------------------------------
#      &&& Compress
# --------------------------------
def compress(
    model,
    task: AlphabetTask,
    task_id: int,
    cfg: DictConfig,
    previous_kb_parameters: Dict[str, torch.Tensor] | None,
    running_fisher: Dict[str, torch.Tensor] | None,
    frozen_model,
    device: torch.device,
):
    """Compress: distill teacher into the KB plus EWC penalty on the running Fisher.

    NOTE: The first task has no previous parameters, so its loss is the KL alone.
    """
    optimizer = create_optimizer(
        cfg.compress_optimizer,
        params=model.parameters_for("compress"),
        lr=cfg.compress_lr,
        momentum=cfg.sgd_momentum,  # Only used if compress_optimizer is sgd
    )

    # Loss: tau^2 KL(teacher || KB) + lambda/2 (theta - theta*)^T (gamma F*) (theta - theta*),
    # one `distill_step` per batch (shared with Atari). The teacher strategy:
    #   frozen_teacher      = paper-correct: a frozen copy of the model after progress
    #   moving_teacher      = the active column reading the live (training) KB
    #   only_active_teacher = the active backbone alone, no adaptors (my first reading of the paper)
    def step_fn(img_batch, label_batch, epoch):
        if cfg.compress_teacher == "moving_teacher":
            y_teacher, y_kb = model(img_batch, mode="compress")
        elif cfg.compress_teacher == "only_active_teacher":
            with torch.no_grad():
                y_teacher = model.active_head(model.active_layers(img_batch))
            y_kb = model.kb_head(model.kb_layers(img_batch))  # student: the KB alone
        elif cfg.compress_teacher == "frozen_teacher":
            with torch.no_grad():
                y_teacher, _ = frozen_model(img_batch, mode="progress")
            y_kb = model.kb_head(model.kb_layers(img_batch))  # student: the KB alone
        else:
            raise ValueError(f"Invalid compress_teacher: {cfg.compress_teacher}")
        return distill_step(
            model,
            y_kb,
            y_teacher,
            optimizer,
            tau=cfg.tau,
            ewc_lambda=cfg.ewc_lambda,
            gamma=cfg.gamma,
            previous_kb_parameters=previous_kb_parameters,
            running_fisher=running_fisher,
            freeze_backbone=epoch < cfg.compress_head_warmup,  # head warm-up
        )

    stats = _run_epochs(
        model,
        task,
        task_id,
        n_epochs=cfg.n_epochs_compress,
        optimizer=optimizer,
        eval_every=cfg.eval_every_compress,
        mode="compress",
        step_fn=step_fn,
        device=device,
    )
    return model, stats


# -----------------------------
#           Main
# -------------------------------
@hydra.main(config_path="conf", config_name="omniglot", version_base=None)
def main(cfg: DictConfig):
    # Seeding
    torch.manual_seed(cfg.seed)

    # Device
    if cfg.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # Data
    tasks = load_tasks(data_root=cfg.data_root, data_augmentation=cfg.data_augmentation)
    n_classes = max_classes(tasks)

    # Model
    model = CNNModel(
        output_dim=n_classes,
        use_batchnorm=cfg.use_batchnorm,
        adaptor_init=cfg.adaptor_init,
        multi_head=cfg.multi_head,
    )
    model.to(device)

    # The KB of model construction, restored before each task when kb_random
    initial_kb = copy.deepcopy(model.kb_layers.state_dict())

    running_fisher = None
    previous_kb_parameters = None
    task_logs = []
    re_eval_metrics = defaultdict(list)

    # P&C task sequentially
    for task_id in range(len(tasks)):
        task = tasks[task_id]

        # Paired random stream of (seed, task_id) (E7): the new columns and heads, the
        # progress data order and its augmentation do not depend on what ran before.
        # The compress phase, the Fisher and the evaluations after progress keep the
        # main stream, so kb_random (which skips them) does not shift this stream.
        # devices=[]: the model is built, and the data shuffled and augmented, on the CPU.
        # (seed + 1): the stream of seed 0, task 0 must not be the main stream of seed 0,
        # or the active column of task 0 would start as a copy of the KB.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((cfg.seed + 1) * 1000 + task_id)

            # Control: the active column reads the frozen random KB of model construction
            if cfg.kb_random:
                model.kb_layers.load_state_dict(initial_kb)

            # New active column, adaptors and heads for the task (see `CNNModel.prep_for_task`)
            model.prep_for_task(
                task_id,
                reinit=cfg.reinit,
                copy_kb=cfg.copy_kb_parameters,
                previous_kb=previous_kb_parameters,
            )

            # -------------------------------------------
            # 1. Progress: train active layers + adaptors
            # --------------------------------------------
            model, progress_stats = progress(model, task, task_id, cfg, device)

        # -------------------------------------------------------
        # 2. Compress: distill into the KB (with EWC penalty)
        # ------------------------------------------------------
        if cfg.kb_random:
            compress_stats = Stats()  # Control: no compress, the KB stays random
        else:
            frozen_model = (
                create_frozen_model(model)
                if cfg.compress_teacher == "frozen_teacher"
                else None
            )
            model, compress_stats = compress(
                model,
                task,
                task_id,
                cfg,
                previous_kb_parameters,
                running_fisher,
                frozen_model,
                device,
            )

        # KB training accuracy after compression, before Fisher
        kb_train_metrics = evaluate(
            model,
            task.train_loader(shuffle=False),
            mode="compress",
            device=device,
        )
        print(
            f"[KB train acc before Fisher T{task_id}] {kb_train_metrics['accuracy']:.4f}"
        )

        # ----------------------------
        #  3. Fisher
        # ----------------------------
        # Fisher matrix for the current task (after compression), then the running Fisher
        if cfg.kb_random:
            # Control: no Fisher. results.json has no fisher_* keys and an empty running Fisher.
            fisher_stats, running_fisher = {}, {}
        else:
            fisher_matrix, fisher_stats = task_fisher(
                model,
                task.train_loader(shuffle=False),
                task_id,
                device,
                verbose=True,
                normalize_fisher=cfg.normalize_fisher,
                fisher_clip=cfg.fisher_clip,
                max_n_samples=cfg.max_n_samples,
            )
            running_fisher = accumulate_fisher(running_fisher, fisher_matrix, cfg.gamma)
            rf_total = sum(v.sum().item() for v in running_fisher.values())
            print(f"[Running Fisher after T{task_id}] total_mass={rf_total:.6e}")

            previous_kb_parameters = copy.deepcopy(model.kb_layers.state_dict())

        # --------------------------------
        # 4. Re-evaluate on previous tasks
        # --------------------------------
        # After training on task k, re-evaluate on previous tasks
        for previous_task_id in range(task_id + 1):
            with model.using(previous_task_id):
                re_eval_metrics[task_id].append(
                    evaluate(
                        model,
                        loader=tasks[previous_task_id].test_loader(shuffle=False),
                        mode="compress",
                        device=device,
                    ),
                )

        accs = [m["accuracy"] for m in re_eval_metrics[task_id]]
        summary_str = " | ".join(f"T{j}: {accs[j]:.3f}" for j in range(len(accs)))
        print(f"After T{task_id}: {summary_str}")

        # --------------------------------
        # 5. Compute metrics / task log(s)
        # --------------------------------
        task_logs.append(
            {
                **task_log(
                    task_id,
                    progress_stats,
                    compress_stats,
                    kb_train_metrics,
                    fisher_stats,
                    running_fisher,
                    accs,
                ),
                "alphabet": task.alphabet_name,
                # The completed epoch (1 = the first) of each progress_{train,eval}_accs entry
                "progress_eval_epochs": [
                    e + 1 for e in sorted(progress_stats.metrics["eval"])
                ],
            }
        )

        if task_id + 1 >= cfg.max_n_tasks:
            break

    # Save results to Hydra output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # type: ignore
    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "task_logs": task_logs,
    }
    results_path = f"{output_dir}/results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
