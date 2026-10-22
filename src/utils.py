"""Fisher information, the Mahalanobis penalty, the distillation step, optimizer and model helpers."""

import copy
from typing import Dict, Iterable

import torch
import torch.nn as nn

# --------------------------- #
#       Fisher utils
# --------------------------- #


def empirical_fisher(
    model,
    loader: Iterable,
    max_n_samples: None | int = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Diagonal empirical Fisher over loader.

    See README.md notes for more details over the design choice

    TODO: Use torch.func.jacrev + vmap ?
    """
    was_training = model.training
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    # Getting a [B, num_params] tensor is expensive;
    # instead we'll loop through the data points
    # Note that we still have one backward per example -> expensive
    diag_fisher = {
        n: torch.zeros_like(p)
        for n, p in model.kb_layers.named_parameters()
        if p.requires_grad
    }
    n_samples = 0

    for img, label in loader:
        img, label = img.to(device), label.to(device)
        _, y_kb = model(img, mode="compress")  # We only want the parameters of the KB
        out = loss_fn(y_kb, label)

        n_batch = out.shape[0]
        for j in range(n_batch):
            n_samples += 1
            model.zero_grad(set_to_none=True)

            # NOTE: Do not retain the graph for the  last sample
            out[j].backward(retain_graph=(j != n_batch - 1))

            for n, p in model.kb_layers.named_parameters():
                if (not p.requires_grad) or (n not in diag_fisher) or (p.grad is None):
                    continue

                # Out-of-place pow so we do not mutate .grad via detach aliases.
                diag_fisher[n].add_(p.grad.detach().pow(2))

            # Stop if we reach max number of samples, also inside a batch
            if (max_n_samples is not None) and (n_samples >= max_n_samples):
                break

        # Stop the batch loop on the same condition
        if (max_n_samples is not None) and (n_samples >= max_n_samples):
            break

    # Get the mean
    inv = 1.0 / n_samples
    for acc in diag_fisher.values():
        acc.mul_(inv)

    if was_training:
        model.train()

    return diag_fisher


def normalize_diagonal_fisher(
    diag_fisher: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """NOTE: AI-generated function

    Scale per-task Fisher so total mass = 1 (Schwarz et al., S4)."""
    total = sum(v.sum().item() for v in diag_fisher.values())
    if total <= 0:
        return diag_fisher
    scale = 1.0 / total
    return {k: v * scale for k, v in diag_fisher.items()}


def clip_diagonal_fisher(
    diag_fisher: Dict[str, torch.Tensor], max_value: float
) -> Dict[str, torch.Tensor]:
    """NOTE: AI-generated function

    Clip each Fisher entry at max_value. The empirical Fisher is heavy-tailed
    (max about 1000 times the mean), and the largest entries set the stability
    limit of SGD on the EWC quadratic: lr * lambda * F_max must stay small."""
    return {k: v.clamp(max=max_value) for k, v in diag_fisher.items()}


def mahalanobis_distance(
    current_model,
    previous_parameters: Dict[str, torch.Tensor],
    running_fisher: Dict[str, torch.Tensor],
    gamma: float,
) -> torch.Tensor:
    # This could benefit from a vectorized implementation (flatten everything)
    # I like the simplicity and readability of getting the named parameter (error prone)
    # -> follows paper Eq. (8): penalty precision is gamma * F*_{k-1}

    running_norm = 0.0
    for n, p in current_model.named_parameters():  # Not detached
        if n in running_fisher and n in previous_parameters:
            running_norm = (
                running_norm
                + (
                    gamma * running_fisher[n] * (p - previous_parameters[n]).pow(2)
                ).sum()
            )

    return running_norm  # type: ignore


def task_fisher(
    model,
    loader: Iterable,
    task_id: int,
    device: torch.device,
    verbose: bool,
    normalize_fisher,
    fisher_clip,
    max_n_samples,
):
    """Diagonal Fisher of the KB backbone on the task (after compress), normalized
    and clipped as configured. `loader` yields (input, target) batches: images
    and labels on Omniglot, states and actions sampled from the KB policy on
    Atari. The heads are not part of `model.kb_layers`.
    Returns the Fisher dict and a dict of its summary numbers."""

    if verbose:
        print("Computing Fisher matrix for task", task_id)

    fisher_matrix = empirical_fisher(
        model, loader, max_n_samples=max_n_samples, device=device
    )
    if normalize_fisher:
        fisher_matrix = normalize_diagonal_fisher(fisher_matrix)
    if fisher_clip is not None:
        fisher_matrix = clip_diagonal_fisher(fisher_matrix, fisher_clip)

    # Compute stats
    total_mass = sum(v.sum().item() for v in fisher_matrix.values())
    n_params = sum(v.numel() for v in fisher_matrix.values())
    fisher_max = max(v.max().item() for v in fisher_matrix.values())

    stats = {
        "fisher_total_mass": total_mass,
        "fisher_per_param_mean": total_mass / n_params,
        "fisher_max": fisher_max,
        "fisher_per_layer": {k: float(v.sum()) for k, v in fisher_matrix.items()},
    }

    # Printing
    if verbose:
        print(
            f"[Fisher T{task_id}] total_mass={total_mass:.6e}  "
            f"per-param mean={total_mass / n_params:.6e}  "
            f"max={fisher_max:.6e}"
        )
    return fisher_matrix, stats


def accumulate_fisher(
    running_fisher: Dict[str, torch.Tensor] | None,
    fisher_matrix: Dict[str, torch.Tensor],
    gamma: float,
) -> Dict[str, torch.Tensor]:
    """Running Fisher of online EWC (paper eq. 8): F*_k = gamma * F*_{k-1} + F_k."""
    if running_fisher is None:
        return fisher_matrix
    return {
        n: gamma * running_fisher[n] + fisher_matrix[n] for n in running_fisher.keys()
    }


# --------------------------- #
#    Optim & Model helpers
# --------------------------- #
def create_optimizer(name, params, lr, momentum):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr)
    else:
        raise ValueError(f"Invalid optimizer: {name}")


def create_frozen_model(model):
    device = next(model.parameters()).device
    frozen_model = copy.deepcopy(model).to(device)
    frozen_model.eval()
    for p in frozen_model.parameters():  # TODO: Is this really needed ?
        p.requires_grad_(False)
    return frozen_model


# --------------------------- #
#       Compress step
# --------------------------- #
def distill_step(
    model,
    y_kb: torch.Tensor,
    y_teacher: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    tau: float,
    ewc_lambda: float,
    gamma: float,
    previous_kb_parameters: Dict[str, torch.Tensor] | None,
    running_fisher: Dict[str, torch.Tensor] | None,
    max_grad_norm: float | None = None,
    freeze_backbone: bool = False,
) -> Dict[str, float]:
    """One SGD step of the compress phase, shared by Omniglot and Atari (paper eq. 7):
    loss = tau^2 KL(teacher || KB) + lambda/2 (theta - theta*)^T (gamma F*) (theta - theta*).
    `y_kb` are the KB logits with the graph, `y_teacher` the teacher logits (no
    graph). The first task has no previous parameters, so its loss is the KL alone.
    `freeze_backbone` keeps only the KB head's gradients (the head warm-up).
    Returns the values of the terms: {"loss", "kl", "ewc"}."""
    kl = (tau**2) * nn.functional.kl_div(
        torch.log_softmax(y_kb / tau, dim=1),  # student
        torch.log_softmax(y_teacher / tau, dim=1),  # teacher
        log_target=True,
        reduction="batchmean",
    )
    if previous_kb_parameters is not None and running_fisher is not None:
        ewc = (
            ewc_lambda
            * 0.5
            * mahalanobis_distance(
                model.kb_layers, previous_kb_parameters, running_fisher, gamma=gamma
            )
        )
    else:
        ewc = torch.zeros((), device=y_kb.device)
    loss = kl + ewc

    optimizer.zero_grad()
    loss.backward()
    if freeze_backbone:
        for p in model.kb_layers.parameters():
            p.grad = None
    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters_for("compress"), max_grad_norm)
    optimizer.step()
    return {"loss": loss.item(), "kl": kl.item(), "ewc": ewc.item()}
