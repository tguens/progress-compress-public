"""AI-generated code.

Atari games - with wrapper around envpool.

Settings (envpool v5)
- frame skip 4
- 4 stacked 84x84 frames, up to 30 no-ops at reset
- no sticky actions
- episodic life and reward clipping -> Evaluation environments use neither,
    so one episode is one full game and the return is the game score.

Reference: CleanRL, https://github.com/vwxyzjn/cleanrl (ppo_atari_envpool.py)
"""

from typing import Callable

import envpool  # type: ignore
import numpy as np
import torch


class AtariTask:
    """One Atari game = one task: its envpool training and evaluation environments."""

    def __init__(self, game: str, seed: int):
        self.game = game  # envpool id, for example "Pong-v5"
        self.seed = seed
        self.n_actions = int(envpool.make_spec(game).action_space.n)

    def train_envs(self, num_envs: int, num_threads: int):
        return envpool.make(
            self.game,
            env_type="gymnasium",
            num_envs=num_envs,
            num_threads=num_threads,
            seed=self.seed,
            episodic_life=True,
            reward_clip=True,
        )

    def eval_envs(self, num_envs: int, num_threads: int, seed: int):
        return envpool.make(
            self.game,
            env_type="gymnasium",
            num_envs=num_envs,
            num_threads=num_threads,
            seed=seed,
            episodic_life=False,
            reward_clip=False,
        )


@torch.no_grad()
def evaluate_policy(
    task: AtariTask,
    policy: Callable[[torch.Tensor], torch.Tensor],
    n_episodes: int,
    seed: int,
    device: torch.device,
    num_threads: int,
) -> dict:
    """Return statistics of `policy` over `n_episodes` full games.

    `policy` maps a uint8 observation batch [N, 4, 84, 84] on `device` to
    actions [N]. The function runs `n_episodes` environments in parallel and
    keeps the first episode of each one. envpool resets a finished environment
    on its own; the mask `done` stops the count for that environment.
    """
    envs = task.eval_envs(n_episodes, num_threads, seed)
    obs, _ = envs.reset()
    returns = np.zeros(n_episodes, dtype=np.float64)
    lengths = np.zeros(n_episodes, dtype=np.int64)
    done = np.zeros(n_episodes, dtype=bool)
    while not done.all():
        actions = policy(torch.as_tensor(obs, device=device)).cpu().numpy()
        obs, reward, terminated, truncated, _ = envs.step(actions)
        returns += reward * ~done
        lengths += ~done
        done |= np.logical_or(terminated, truncated)
    envs.close()
    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "returns": returns.tolist(),
        "mean_length": float(lengths.mean()),
    }


def random_policy(n_actions: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Uniform random actions. Its return is the floor of the normalised score."""
    return lambda obs: torch.randint(n_actions, (obs.shape[0],), device=obs.device)
