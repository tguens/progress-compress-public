"""Progress & Compress on Atari games.

One game = one task. For each game:
1. Progress: PPO trains the active column and the adaptors; the knowledge
   base (KB) is frozen. The PPO loop follows CleanRL's `ppo_atari_envpool.py`
   (MIT, https://github.com/vwxyzjn/cleanrl), with the agent swapped for the
   two-column `PCAgent` of `src/model.py`.
2. Compress: the KB learns the frozen active policy by distillation
   (`utils.distill_step`, the same step as Omniglot) on states that the active
   policy visits, then on states that the KB policy itself visits: DAgger
   rounds on a buffer, or the online phase of the paper (`compress_online`).
   Plus the EWC penalty with the running diagonal Fisher of the KB.
3. Fisher: the diagonal Fisher of the KB policy on the game is added to the
   running Fisher, and the KB policy is evaluated on every game seen so far
   with the head of that game.

The single-network baselines of the paper (`mode=finetune`, `mode=ewc`) run
the same loop with `SingleAgent`: PPO on the whole network, no compress phase.

Run: `uv run src/run_atari.py max_n_tasks=2 total_timesteps=[50000,50000]`

NOTE: Mostly an AI-generated script with my instructions to
    1. follow `run_omniglot.py`
    2. uses CleanRL for tasks/ppo loops

It's about 900 lines and the better way to understand the algorithm(s)
    is to go over `run_omniglot.py`
"""

import copy
import json
import os
import time
from collections import deque

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.distributions.categorical import Categorical

from environments.atari import AtariTask, evaluate_policy, random_policy
from metrics import Logger, Stats, task_log_atari
from model import PCAgent, SingleAgent, load_teacher_state_dict
from utils import (
    accumulate_fisher,
    create_frozen_model,
    create_optimizer,
    distill_step,
    mahalanobis_distance,
    task_fisher,
)


# ---------------------------------
#       Policies and rollouts
# ---------------------------------
def pick_action(logits: torch.Tensor, mode: str) -> torch.Tensor:
    """`sample` draws from the policy; `greedy` takes its argmax."""
    if mode == "sample":
        return Categorical(logits=logits).sample()
    elif mode == "greedy":
        return logits.argmax(dim=1)
    else:
        raise ValueError(f"Invalid eval_mode: {mode}")


def kb_policy(agent, mode: str = "sample"):
    """The KB alone, with its current head: how it is evaluated and how it acts."""
    return lambda obs: pick_action(agent.kb_head(agent.kb_layers(obs)), mode)


def active_policy(agent, mode: str = "sample"):
    """The active column reading the KB through the adaptors."""
    return lambda obs: pick_action(agent(obs, mode="progress")[0], mode)


class EpisodeTracker:
    """Episode statistics as in CleanRL's RecordEpisodeStatistics: a game ends
    when `terminated` and `lives == 0`; `info["reward"]` is the unclipped
    reward. `curve` holds [step, return] of every finished game."""

    def __init__(self, num_envs: int):
        self.returns = np.zeros(num_envs, dtype=np.float32)
        self.curve: list[list[float]] = []
        self.recent = deque(maxlen=20)

    def update(self, step: int, info: dict, terminated: np.ndarray) -> None:
        self.returns += info["reward"]
        for idx in np.flatnonzero(np.logical_and(terminated, info["lives"] == 0)):
            self.curve.append([step, float(self.returns[idx])])
            self.recent.append(float(self.returns[idx]))
        self.returns *= 1 - info["terminated"]

    @property
    def mean_recent(self) -> float:
        return sum(self.recent) / len(self.recent) if self.recent else float("nan")


@torch.no_grad()
def rollout_states(
    envs,
    next_obs: torch.Tensor,
    actor,
    teacher,
    obs_out: torch.Tensor,
    logits_out: torch.Tensor,
    num_envs: int,
    tracker: EpisodeTracker | None = None,
    step: int = 0,
) -> tuple[torch.Tensor, int]:
    """Run `actor` in `envs` from `next_obs` for `obs_out.shape[0] // num_envs`
    steps. Write the visited states into `obs_out` [N, 4, 84, 84] and the
    teacher's logits for them into `logits_out` [N, A]. Returns the next
    observation and the step count, so a caller can continue the rollout."""
    for t in range(obs_out.shape[0] // num_envs):
        lo, hi = t * num_envs, (t + 1) * num_envs
        obs_out[lo:hi] = next_obs
        logits_out[lo:hi], _ = teacher(next_obs, mode="progress")
        next_obs_np, _, terminated, _, info = envs.step(actor(next_obs).cpu().numpy())
        next_obs = torch.as_tensor(next_obs_np, device=obs_out.device)
        step += num_envs
        if tracker is not None:
            tracker.update(step, info, terminated)
    return next_obs, step


def collect_states(
    task: AtariTask, actor, teacher, n_states: int, n_actions: int, cfg, device
):
    """`n_states` states visited by `actor` in fresh environments, with the
    teacher's logits. Returns (obs [N, 4, 84, 84] uint8, logits [N, A])."""
    obs = torch.zeros((n_states, 4, 84, 84), dtype=torch.uint8, device=device)
    logits = torch.zeros((n_states, n_actions), device=device)
    envs = task.train_envs(cfg.num_envs, cfg.env_threads)
    next_obs = torch.as_tensor(envs.reset()[0], device=device)
    rollout_states(envs, next_obs, actor, teacher, obs, logits, cfg.num_envs)
    envs.close()
    return obs, logits


def evaluate(task: AtariTask, policy, cfg, device) -> dict:
    """Mean return of `policy` over `n_eval_episodes` full games of `task`."""
    return evaluate_policy(
        task,
        policy,
        cfg.n_eval_episodes,
        seed=cfg.seed + 1000,
        device=device,
        num_threads=cfg.env_threads,
    )


# ---------------------------------
#       ----> Progress: PPO
# ---------------------------------
def _gae(rewards, values, dones, next_value, next_done, ppo_gamma, gae_lambda):
    """Generalised advantage estimation over a rollout [T, N] (CleanRL)."""
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    n_steps = rewards.shape[0]
    for t in reversed(range(n_steps)):
        nextnonterminal = 1.0 - (next_done if t == n_steps - 1 else dones[t + 1])
        nextvalues = next_value if t == n_steps - 1 else values[t + 1]
        delta = rewards[t] + ppo_gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = (
            delta + ppo_gamma * gae_lambda * nextnonterminal * lastgaelam
        )
    return advantages, advantages + values


def _ppo_update(agent, optimizer, params, batch: dict, cfg, ewc) -> dict:
    """`update_epochs` passes of the clipped PPO objective over the rollout
    (CleanRL), plus the online EWC penalty on `agent.kb_layers` when `ewc` =
    (previous parameters, running Fisher) is given (the single-network EWC
    baseline). Returns the terms of the last minibatch."""
    batch_size = batch["obs"].shape[0]
    minibatch_size = batch_size // cfg.num_minibatches
    b_inds = np.arange(batch_size)
    terms = {}
    for _ in range(cfg.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            mb = b_inds[start : start + minibatch_size]
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                batch["obs"][mb], batch["actions"][mb]
            )
            logratio = newlogprob - batch["logprobs"][mb]
            ratio = logratio.exp()

            mb_advantages = batch["advantages"][mb]
            if cfg.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if cfg.clip_vloss:
                v_loss_unclipped = (newvalue - batch["returns"][mb]) ** 2
                v_clipped = batch["values"][mb] + torch.clamp(
                    newvalue - batch["values"][mb], -cfg.clip_coef, cfg.clip_coef
                )
                v_loss_clipped = (v_clipped - batch["returns"][mb]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * ((newvalue - batch["returns"][mb]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef
            ewc_term = torch.zeros((), device=loss.device)
            if ewc is not None:
                ewc_term = (
                    cfg.ewc_lambda
                    * 0.5
                    * mahalanobis_distance(
                        agent.kb_layers, ewc[0], ewc[1], gamma=cfg.gamma
                    )
                )
                loss = loss + ewc_term

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()
            terms = {
                "policy_loss": pg_loss.item(),
                "value_loss": v_loss.item(),
                "entropy": entropy_loss.item(),
                "ewc": ewc_term.item(),
            }
    return terms


def progress(
    agent,
    task: AtariTask,
    total_timesteps: int,
    cfg,
    device,
    logger: Logger,
    step_offset: int,
    ewc=None,
    ckpt_prefix: str | None = None,
):
    """PPO on `agent.parameters_for("progress")`: the active column, its heads and
    the adaptors of a `PCAgent` (the KB is frozen in the forward), or the whole
    network of a `SingleAgent`. CleanRL's loop: rollout, GAE, clipped update.
    Returns the episodic returns seen in training as [step, return] and the
    per-iteration losses as `Stats`. `step_offset` is the PPO step count of
    the earlier tasks, the x axis of the wandb log. With `cfg.checkpoint_every`
    > 0 and a `ckpt_prefix`, the agent is saved as `<ckpt_prefix>_<step>.pt` at
    step 0, every `checkpoint_every` steps and at the end."""
    envs = task.train_envs(cfg.num_envs, cfg.env_threads)
    params = agent.parameters_for("progress")
    optimizer = torch.optim.Adam(params, lr=cfg.learning_rate, eps=1e-5)
    batch_size = cfg.num_envs * cfg.num_steps
    num_iterations = total_timesteps // batch_size

    shape = (cfg.num_steps, cfg.num_envs)
    obs = torch.zeros(shape + (4, 84, 84), dtype=torch.uint8, device=device)
    actions = torch.zeros(shape, dtype=torch.long, device=device)
    logprobs, rewards, dones, values = (
        torch.zeros(shape, device=device) for _ in range(4)
    )

    tracker = EpisodeTracker(cfg.num_envs)
    stats = Stats()
    global_step = 0
    start_time = time.time()
    next_obs = torch.as_tensor(envs.reset()[0], device=device)
    next_done = torch.zeros(cfg.num_envs, device=device)
    agent.train()
    save_ckpt = cfg.checkpoint_every > 0 and ckpt_prefix is not None
    if save_ckpt:
        torch.save(agent.state_dict(), f"{ckpt_prefix}_0.pt")
    next_ckpt = cfg.checkpoint_every

    for iteration in range(1, num_iterations + 1):
        if cfg.anneal_lr:
            optimizer.param_groups[0]["lr"] = (
                1.0 - (iteration - 1.0) / num_iterations
            ) * cfg.learning_rate

        # Rollout
        for step in range(cfg.num_steps):
            global_step += cfg.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminated, truncated, info = envs.step(
                action.cpu().numpy()
            )
            rewards[step] = torch.as_tensor(reward, device=device)
            next_obs = torch.as_tensor(next_obs_np, device=device)
            next_done = torch.as_tensor(
                np.logical_or(terminated, truncated), dtype=torch.float32, device=device
            )
            tracker.update(global_step, info, terminated)

        # Advantages, then the update
        with torch.no_grad():
            next_value = agent.get_action_and_value(next_obs)[3].reshape(1, -1)
            advantages, returns = _gae(
                rewards,
                values,
                dones,
                next_value,
                next_done,
                cfg.ppo_gamma,
                cfg.gae_lambda,
            )
        batch = {
            "obs": obs.reshape((-1, 4, 84, 84)),
            "actions": actions.reshape(-1),
            "logprobs": logprobs.reshape(-1),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
            "values": values.reshape(-1),
        }
        terms = _ppo_update(agent, optimizer, params, batch, cfg, ewc)

        # Stats and log
        stats.acc(terms, epoch=iteration)
        sps = int(global_step / (time.time() - start_time))
        logger.log(
            {
                "progress/return_last20": tracker.mean_recent,
                **{f"progress/{k}": v for k, v in terms.items()},
                "progress/sps": sps,
            },
            step_offset + global_step,
        )
        if iteration % 50 == 0 or iteration == num_iterations:
            print(
                f"[progress {task.game}] step={global_step} SPS={sps} return(last 20)={tracker.mean_recent:.1f} entropy={terms['entropy']:.3f}",
                flush=True,
            )
        if save_ckpt and (global_step >= next_ckpt or iteration == num_iterations):
            torch.save(agent.state_dict(), f"{ckpt_prefix}_{global_step}.pt")
            next_ckpt += cfg.checkpoint_every

    envs.close()
    return tracker.curve, stats


# ---------------------------------
#       &&& Compress: distillation
# ---------------------------------
def _distill_epochs(
    agent,
    obs: torch.Tensor,
    teacher_logits: torch.Tensor,
    n_epochs: int,
    optimizer,
    cfg,
    previous_kb_parameters,
    running_fisher,
    stats: Stats,
    logger: Logger,
    step: int,
    head_warmup: int = 0,
    verbose: bool = True,
) -> None:
    """`n_epochs` passes of `distill_step` over the buffer (obs, teacher logits)
    in minibatches. Each epoch is one entry of `stats` (key `logger.compress_epoch`)."""
    n = obs.shape[0]
    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=obs.device)
        epoch_terms = []
        for start in range(0, n, cfg.compress_batch_size):
            mb = perm[start : start + cfg.compress_batch_size]
            y_kb = agent.kb_head(agent.kb_layers(obs[mb]))  # student: the KB alone
            epoch_terms.append(
                distill_step(
                    agent,
                    y_kb,
                    teacher_logits[mb],
                    optimizer,
                    tau=cfg.tau,
                    ewc_lambda=cfg.ewc_lambda,
                    gamma=cfg.gamma,
                    previous_kb_parameters=previous_kb_parameters,
                    running_fisher=running_fisher,
                    max_grad_norm=cfg.compress_max_grad_norm,
                    freeze_backbone=epoch < head_warmup,
                )
            )
        for terms in epoch_terms:
            stats.acc(terms, epoch=logger.compress_epoch)
        means = {k: stats.epoch_mean(k)[-1] for k in ("loss", "kl", "ewc")}
        logger.log(
            {f"compress/{k}": v for k, v in means.items()}
            | {"compress/epoch": logger.compress_epoch},
            step,
        )
        logger.compress_epoch += 1
        if verbose:
            print(
                f"[compress] epoch {epoch} kl={means['kl']:.4f} ewc={means['ewc']:.4f}",
                flush=True,
            )


def _compress_online(
    agent,
    teacher,
    task,
    cfg,
    device,
    optimizer,
    previous_kb_parameters,
    running_fisher,
    stats,
    logger,
    round0_states,
    step,
):
    """The paper's compress phase (section 4): the KB policy acts in the game,
    the frozen teacher labels every state it visits, and the KB learns the
    labels as they come (continuous DAgger), with the EWC penalty. One rollout
    is `num_envs` x `num_steps` states, then `compress_online_epochs` passes of
    SGD over it. Returns the KB's episodic returns as [step, return], the KB
    reward during compress of the paper's Figure 4."""
    n_states = cfg.num_envs * cfg.num_steps
    obs = torch.zeros(
        (n_states, 4, 84, 84), dtype=torch.uint8, device=device
    )  # [N, 4, 84, 84]
    teacher_logits = torch.zeros((n_states, task.n_actions), device=device)  # [N, A]
    envs = task.train_envs(cfg.num_envs, cfg.env_threads)
    next_obs = torch.as_tensor(envs.reset()[0], device=device)
    tracker = EpisodeTracker(cfg.num_envs)
    online_step, next_report = 0, 100000
    online_base = logger.online_step  # env steps of the earlier online phases
    n_rounds = cfg.compress_online_steps // n_states
    for round_id in range(n_rounds):
        next_obs, online_step = rollout_states(
            envs,
            next_obs,
            kb_policy(agent),
            teacher,
            obs,
            teacher_logits,
            cfg.num_envs,
            tracker,
            online_step,
        )
        _distill_epochs(
            agent,
            obs,
            teacher_logits,
            cfg.compress_online_epochs,
            optimizer,
            cfg,
            previous_kb_parameters,
            running_fisher,
            stats,
            logger,
            step,
            verbose=False,
        )
        if online_step >= next_report or round_id == n_rounds - 1:
            next_report += 100000
            kl, ewc = stats.epoch_mean("kl")[-1], stats.epoch_mean("ewc")[-1]
            round0 = round0_kl(agent, *round0_states)
            logger.online_step = online_base + online_step
            print(
                f"[compress online {task.game}] step={online_step} kl={kl:.4f} ewc={ewc:.4f} round0_kl={round0:.4f} return(last 20)={tracker.mean_recent:.1f}",
                flush=True,
            )
            logger.log(
                {
                    "compress_online/kl": kl,
                    "compress_online/ewc": ewc,
                    "compress_online/round0_kl": round0,
                    "compress_online/return_last20": tracker.mean_recent,
                    "compress_online/step": logger.online_step,
                },
                step,
            )
    logger.online_step = online_base + online_step
    envs.close()
    return tracker.curve


def compress(
    agent,
    teacher,
    task: AtariTask,
    cfg,
    device,
    previous_kb_parameters,
    running_fisher,
    logger: Logger,
    step: int,
):
    """Distill the frozen teacher into the KB. Round 0 on `compress_n_states`
    states the teacher visits, `n_epochs_compress` epochs. Then either the
    DAgger rounds 1 to `compress_n_rounds` - 1 on the growing buffer, with the
    states the KB policy itself visits, or the online phase (`compress_online`).
    Returns the stats, the KB reward curve of the online phase, and the round 0
    states and logits (the Fisher's data and the round-0 KL diagnostic)."""

    def new_optimizer():
        # A fresh optimizer (no momentum carried over) for each DAgger round and
        # for the online phase, as in the entry point before the 2026-10-21 cleanup,
        # `main_atari.py`, which produced the archive.
        return create_optimizer(
            cfg.compress_optimizer,
            agent.parameters_for("compress"),
            lr=cfg.compress_lr,
            momentum=cfg.sgd_momentum,
        )

    stats = Stats()
    agent.train()
    n_rounds = 1 if cfg.compress_online else cfg.compress_n_rounds
    n_total = n_rounds * cfg.compress_n_states
    obs = torch.zeros((n_total, 4, 84, 84), dtype=torch.uint8, device=device)
    teacher_logits = torch.zeros((n_total, task.n_actions), device=device)
    envs = task.train_envs(cfg.num_envs, cfg.env_threads)
    for round_id in range(n_rounds):
        # Round 0: states of the teacher. Later rounds: states of the KB policy (DAgger).
        actor = active_policy(teacher) if round_id == 0 else kb_policy(agent)
        lo, hi = (
            round_id * cfg.compress_n_states,
            (round_id + 1) * cfg.compress_n_states,
        )
        next_obs = torch.as_tensor(envs.reset()[0], device=device)
        rollout_states(
            envs,
            next_obs,
            actor,
            teacher,
            obs[lo:hi],
            teacher_logits[lo:hi],
            cfg.num_envs,
        )
        _distill_epochs(
            agent,
            obs[:hi],
            teacher_logits[:hi],
            cfg.n_epochs_compress,
            new_optimizer(),
            cfg,
            previous_kb_parameters,
            running_fisher,
            stats,
            logger,
            step,
            head_warmup=cfg.compress_head_warmup if round_id == 0 else 0,
        )
    envs.close()

    curve: list[list[float]] = []
    n_round0 = cfg.compress_batch_size * 8
    if cfg.compress_online:
        curve = _compress_online(
            agent,
            teacher,
            task,
            cfg,
            device,
            new_optimizer(),
            previous_kb_parameters,
            running_fisher,
            stats,
            logger,
            (obs[:n_round0], teacher_logits[:n_round0]),
            step,
        )
    n0 = cfg.compress_n_states
    return stats, curve, obs[:n0], teacher_logits[:n0]


# ---------------------------------
#       Per-task diagnostics
# ---------------------------------
def fisher_batches(agent, obs: torch.Tensor, n_samples: int, batch_size: int):
    """(state, action) batches for `empirical_fisher`. The actions are sampled
    from the KB policy itself, so the squared gradient of the cross-entropy is
    a Monte Carlo estimate of the true Fisher of the policy."""
    for start in range(0, n_samples, batch_size):
        mb = obs[start : start + batch_size]
        with (
            torch.no_grad()
        ):  # not around the yield: `empirical_fisher` needs the graph
            action = Categorical(logits=agent.kb_head(agent.kb_layers(mb))).sample()
        yield mb, action


@torch.no_grad()
def round0_kl(agent, obs: torch.Tensor, teacher_logits: torch.Tensor) -> float:
    """KL(teacher || KB) on the round-0 states of a game, with that game's KB head
    (switch to it with `agent.using(game)`). The states are the first
    `compress_batch_size * 8` states of the round-0 buffer: the first steps after
    reset in each environment, and the compress phase also trains on them. So this
    KL mixes the fitting error with later drift, on early-game states. Older logs
    call it `held_out_kl`; results.json keeps the key `re_eval_kls`."""
    log_p = torch.log_softmax(teacher_logits, dim=1)
    log_q = torch.log_softmax(agent.kb_head(agent.kb_layers(obs)), dim=1)
    return float((log_p.exp() * (log_p - log_q)).sum(dim=1).mean())


def re_evaluate(
    agent,
    seen: list[str],
    tasks_by_game,
    round0_states,
    random_returns,
    teacher_returns,
    cfg,
    device,
) -> dict:
    """The KB policy on every game seen so far, with the head of that game: the
    greedy or sampled return over full games, normalised so that 0 is the random
    policy and 1 the latest active policy of the game, and the KL on round-0 states."""
    evals, kls = [], []
    for game in seen:
        with agent.using(game):
            evals.append(
                evaluate(
                    tasks_by_game[game], kb_policy(agent, cfg.eval_mode), cfg, device
                )
            )
            kls.append(round0_kl(agent, *round0_states[game]))
    returns = [m["mean_return"] for m in evals]
    norm_returns = [
        (
            (r - random_returns[g]) / (teacher_returns[g] - random_returns[g])
            if teacher_returns[g] != random_returns[g]
            else float("nan")
        )
        for g, r in zip(seen, returns)
    ]
    return {
        "re_eval_games": list(seen),
        "re_eval_returns": returns,
        "re_eval_norm_returns": norm_returns,
        "re_eval_return_stds": [m["std_return"] for m in evals],
        "re_eval_lengths": [m["mean_length"] for m in evals],
        "re_eval_kls": kls,
    }


def train_or_load_teacher(
    agent,
    task: AtariTask,
    task_id: int,
    cfg,
    device,
    logger,
    step_offset,
    ewc,
    output_dir,
):
    """The teacher of the compress phase: a frozen copy of the agent after its
    progress phase (PPO), or, in compress-only mode (`teachers_dir`), the
    snapshot saved by an earlier run. Returns the teacher, the training curve
    and the progress stats."""
    if cfg.teachers_dir is None:
        if cfg.checkpoint_every > 0:
            os.makedirs(f"{output_dir}/ckpt", exist_ok=True)
        curve, stats = progress(
            agent,
            task,
            int(cfg.total_timesteps[task_id]),
            cfg,
            device,
            logger,
            step_offset,
            ewc,
            ckpt_prefix=f"{output_dir}/ckpt/T{task_id}",
        )
        teacher = create_frozen_model(agent)
    else:
        curve, stats = [], Stats()
        teacher = PCAgent(adaptor_init=cfg.adaptor_init).to(device)
        teacher.prep_for_task(task.game, task.n_actions)
        state = torch.load(
            f"{cfg.teachers_dir}/teacher_T{task_id}.pt", map_location=device
        )
        missing, _ = teacher.load_state_dict(
            load_teacher_state_dict(state, task.game), strict=False
        )
        assert not missing, (
            f"teacher_T{task_id}.pt lacks {missing}"
        )  # other games' heads may be extra
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    if cfg.save_teachers and cfg.teachers_dir is None:
        torch.save(teacher.state_dict(), f"{output_dir}/teacher_T{task_id}.pt")
    return teacher, curve, stats


# ---------------------------------
#           Main
# ---------------------------------
@hydra.main(config_path="conf", config_name="atari", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True  # fixed input sizes

    tasks = [AtariTask(game, seed=cfg.seed) for game in cfg.games[: cfg.max_n_tasks]]
    if cfg.mode == "pc":
        agent = PCAgent(adaptor_init=cfg.adaptor_init).to(device)
    elif cfg.mode in ("finetune", "ewc"):
        agent = SingleAgent().to(device)
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}")

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # type: ignore
    logger = Logger(
        cfg.wandb,
        cfg.wandb_project,
        cfg.wandb_entity,
        output_dir,
        OmegaConf.to_container(cfg, resolve=True),  # type: ignore
        [cfg.mode],
    )

    running_fisher = None
    previous_kb_parameters = None
    task_logs = []
    seen: list[str] = []  # distinct games in first-visit order
    tasks_by_game: dict[
        str, AtariTask
    ] = {}  # the task of the first visit (the evaluation environments)
    round0_states: dict[
        str, tuple[torch.Tensor, torch.Tensor]
    ] = {}  # round-0 states and teacher logits per game (diagnostic, also trained on)
    random_returns: dict[str, float] = {}
    teacher_returns: dict[str, float] = {}  # latest, for the normalised return

    for task_id, task in enumerate(tasks):
        t0 = time.time()
        revisit = task.game in seen
        if not revisit:
            seen.append(task.game)
            tasks_by_game[task.game] = task
            random_returns[task.game] = evaluate(
                task, random_policy(task.n_actions), cfg, device
            )["mean_return"]
        print(
            f"[T{task_id} {task.game}] actions={task.n_actions} random return={random_returns[task.game]:.1f}"
            + (" (re-visit)" if revisit else "")
        )
        step_offset = int(sum(cfg.total_timesteps[:task_id]))
        cumulative_steps = int(sum(cfg.total_timesteps[: task_id + 1]))

        # New active column, adaptors and heads for the game; a revisit keeps its KB head.
        agent.prep_for_task(task.game, task.n_actions)

        # 1. Progress: PPO. The teacher is a frozen copy of the agent after it.
        ewc = (
            (previous_kb_parameters, running_fisher)
            if cfg.mode == "ewc" and running_fisher is not None
            else None
        )
        teacher, progress_curve, progress_stats = train_or_load_teacher(
            agent, task, task_id, cfg, device, logger, step_offset, ewc, output_dir
        )
        progress_eval = evaluate(
            task, active_policy(teacher, cfg.eval_mode), cfg, device
        )
        teacher_returns[task.game] = progress_eval["mean_return"]
        print(
            f"[T{task_id}] active policy return={progress_eval['mean_return']:.1f} ± {progress_eval['std_return']:.1f} ({time.time() - t0:.0f} s)"
        )

        # 2. Compress: distill the teacher into the KB (P&C only). The single
        # network only collects the teacher's states, for the Fisher.
        if cfg.mode == "pc":
            compress_stats, compress_curve, obs, teacher_logits = compress(
                agent,
                teacher,
                task,
                cfg,
                device,
                previous_kb_parameters,
                running_fisher,
                logger,
                cumulative_steps,
            )
        else:
            compress_stats, compress_curve = Stats(), []
            obs, teacher_logits = collect_states(
                task,
                active_policy(teacher),
                teacher,
                cfg.compress_n_states,
                task.n_actions,
                cfg,
                device,
            )
        del teacher
        n_round0 = cfg.compress_batch_size * 8
        round0_states[task.game] = (
            obs[:n_round0].clone(),
            teacher_logits[:n_round0].clone(),
        )

        # 3. Fisher of the KB policy on the states of this game, then the running Fisher
        fisher_matrix, fisher_stats = task_fisher(
            agent,
            fisher_batches(agent, obs, cfg.fisher_n_samples, 64),
            task_id,
            device,
            verbose=True,
            normalize_fisher=cfg.normalize_fisher,
            fisher_clip=cfg.fisher_clip,
            max_n_samples=cfg.fisher_n_samples,
        )
        running_fisher = accumulate_fisher(running_fisher, fisher_matrix, cfg.gamma)
        previous_kb_parameters = copy.deepcopy(agent.kb_layers.state_dict())
        del obs, teacher_logits
        torch.cuda.empty_cache()  # give the buffer back, so the next game's buffer is one block

        if cfg.checkpoint_every > 0:  # the agent after this task's compress phase
            torch.save(agent.state_dict(), f"{output_dir}/ckpt/T{task_id}_end.pt")

        # 4. Re-evaluate the KB policy on every game seen so far
        re_eval = re_evaluate(
            agent,
            seen,
            tasks_by_game,
            round0_states,
            random_returns,
            teacher_returns,
            cfg,
            device,
        )
        print(
            f"After T{task_id}: "
            + " | ".join(
                f"{g}: {r:.1f} (kl {kl:.3f})"
                for g, r, kl in zip(
                    seen, re_eval["re_eval_returns"], re_eval["re_eval_kls"]
                )
            ),
            flush=True,
        )
        norm = re_eval["re_eval_norm_returns"]
        norm_prev = norm[:-1] if not revisit else norm
        logger.log(
            {
                **{
                    f"eval/{g}/return": r
                    for g, r in zip(seen, re_eval["re_eval_returns"])
                },
                **{f"eval/{g}/norm": r for g, r in zip(seen, norm)},
                **{f"eval/{g}/kl": kl for g, kl in zip(seen, re_eval["re_eval_kls"])},
                "eval/all_avg_norm": float(np.nanmean(norm)),
                "eval/prev_avg_norm": (
                    float(np.nanmean(norm_prev)) if norm_prev else float("nan")
                ),
                "task/id": task_id,
                "task/progress_return": progress_eval["mean_return"],
                "task/fisher_total_mass": fisher_stats["fisher_total_mass"],
                "task/fisher_max": fisher_stats["fisher_max"],
                "task/wall_time_s": time.time() - t0,
            },
            cumulative_steps,
        )

        # 5. The task record
        task_logs.append(
            task_log_atari(
                task_id,
                task,
                revisit,
                cumulative_steps,
                random_returns[task.game],
                progress_eval,
                progress_curve,
                progress_stats,
                compress_stats,
                compress_curve,
                fisher_stats,
                running_fisher,
                re_eval,
                time.time() - t0,
            )
        )

    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "task_logs": task_logs,
    }
    results_path = f"{output_dir}/results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    logger.finish(results_path)


if __name__ == "__main__":
    main()
