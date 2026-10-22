"""AI-generated script. Render gameplay clips from the checkpoints of a run made with
`checkpoint_every` > 0 (see src/conf/atari.yaml), for the gameplay animation.

For each checkpoint `ckpt/T<k>_<step>.pt` (during the progress phase of task k) and
`ckpt/T<k>_end.pt` (after task k), it plays one greedy game in the evaluation
environment (no episodic life, no reward clipping, a fixed seed per game, so every
clip of a game starts the same way) and keeps the first `--frames` frames the agent
sees (the newest of its 4 stacked 84x84 grayscale frames) as a PNG sprite sheet. The
game is played to its end, so the clip also carries the full return. The same
policy also plays `--n-eval` greedy games from other starts; their mean is the
return to plot, since one greedy game can be a lucky one.

Which policy plays:
- the learner: the policy trained in that progress phase on its game (P&C: the
  active column reading the KB; fine-tuning: the network);
- the memory, on every earlier game: P&C: the KB with that game's head, at the end
  of each task only (the KB is frozen during progress); fine-tuning: the network
  with that game's head, at every checkpoint (it is what drifts).

Writes `<out>/<label>/*.png` and `<out>/<label>/clips.json`.

Usage (on the pod, from the repository root):
    uv run experiments/render_clips.py outputs/atari/viz/pc_s0 --label pc --out outputs/viz/gameplay
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from environments.atari import AtariTask, evaluate_policy  # noqa: E402
from model import PCAgent, SingleAgent  # noqa: E402
from run_atari import active_policy, kb_policy  # noqa: E402

GAME_SEED = 4000  # evaluation seed of every clip
EVAL_SEED = 1234  # seed of the 10-game evaluation, as many starts as games
COLS = 20  # frames per row of a sprite sheet


def load_agent(cfg, state: dict, games: list[str], device) -> torch.nn.Module:
    """An agent with the heads of `games`, loaded from a checkpoint."""
    agent = PCAgent(adaptor_init=cfg.adaptor_init) if cfg.mode == "pc" else SingleAgent()
    agent.to(device)
    for game in games:
        agent.prep_for_task(game, AtariTask(game, 0).n_actions)
    agent.load_state_dict(state, strict=True)
    agent.eval()
    return agent


@torch.no_grad()
def play(task: AtariTask, policy, n_frames: int, device, max_steps: int = 27000):
    """One greedy game. Returns the first `n_frames` frames [n, 84, 84] uint8, the
    running score at each kept frame, the full return and the game length."""
    envs = task.eval_envs(1, 1, GAME_SEED)
    obs, _ = envs.reset()
    frames, scores, total, length = [], [], 0.0, 0
    for length in range(1, max_steps + 1):
        if len(frames) < n_frames:
            frames.append(obs[0, -1].copy())  # [84, 84], the newest frame
        action = policy(torch.as_tensor(obs, device=device)).cpu().numpy()
        obs, reward, terminated, truncated, _ = envs.step(action)
        total += float(reward[0])
        if len(scores) < n_frames:
            scores.append(total)
        if terminated[0] or truncated[0]:
            break
    envs.close()
    return np.stack(frames), scores, total, length


def save_sheet(frames: np.ndarray, path: Path) -> None:
    """Frames [n, 84, 84] as one grayscale PNG, COLS frames per row."""
    n = len(frames)
    rows = (n + COLS - 1) // COLS
    sheet = np.zeros((rows * 84, COLS * 84), dtype=np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, COLS)
        sheet[r * 84 : (r + 1) * 84, c * 84 : (c + 1) * 84] = f
    Image.fromarray(sheet, mode="L").save(path, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path, help="Run directory with ckpt/ and .hydra/")
    parser.add_argument("--label", required=True, help="Name of this learner in the page, for example pc or finetune")
    parser.add_argument("--out", type=Path, default=Path("outputs/viz/gameplay"))
    parser.add_argument("--frames", type=int, default=400, help="Frames kept per clip (15 agent steps per second of game)")
    parser.add_argument("--n-eval", type=int, default=10, help="Greedy games per clip for the mean return (one game alone is noisy)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.run / ".hydra" / "config.yaml")
    games = list(cfg.games[: cfg.max_n_tasks])
    budgets = [int(x) for x in cfg.total_timesteps[: cfg.max_n_tasks]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out / args.label
    out.mkdir(parents=True, exist_ok=True)

    ckpts = []
    for p in (args.run / "ckpt").glob("T*_*.pt"):
        k, step = re.match(r"T(\d+)_(\w+)\.pt", p.name).groups()
        ckpts.append((int(k), int(step) if step != "end" else budgets[int(k)], step == "end", p))
    ckpts.sort(key=lambda c: (c[0], c[1], c[2]))

    tasks = {g: AtariTask(g, cfg.seed) for g in games}
    clips = []
    for k, step, is_end, path in ckpts:
        agent = load_agent(cfg, torch.load(path, map_location=device), games[: k + 1], device)
        global_step = sum(budgets[:k]) + step
        jobs = []
        if not is_end:  # the learner on its own game
            jobs.append(("learner", games[k], "active"))
        if is_end or cfg.mode != "pc":  # the memory on every game seen so far
            jobs += [("memory", g, "kb") for g in dict.fromkeys(games[: k + 1]) if is_end or g != games[k]]
        for role, game, which in jobs:
            with agent.using(game):
                policy = active_policy(agent, "greedy") if which == "active" else kb_policy(agent, "greedy")
                frames, scores, total, length = play(tasks[game], policy, args.frames, device)
                ev = evaluate_policy(tasks[game], policy, args.n_eval, EVAL_SEED, device, 4)
            name = f"T{k}_{'end' if is_end else step}_{role}_{game.split('-')[0]}.png"
            save_sheet(frames, out / name)
            clips.append({
                "file": name, "role": role, "game": game, "task": k, "step": step,
                "end_of_task": is_end, "global_step": global_step, "n_frames": len(frames),
                "return": total, "length": length, "scores": scores,
                "eval_mean": ev["mean_return"], "eval_std": ev["std_return"], "eval_returns": ev["returns"],
            })
            print(f"{args.label} T{k} {'end' if is_end else step} {role} {game}: return {total:.0f} in {length} steps, {args.n_eval} games {ev['mean_return']:+.1f} ± {ev['std_return']:.1f}", flush=True)

    results_path = args.run / "results.json"  # absent while the run is still going
    results = json.load(open(results_path)) if results_path.exists() else {"task_logs": []}
    manifest = {
        "label": args.label, "mode": cfg.mode, "games": games, "budgets": budgets,
        "cols": COLS, "frame_size": 84, "game_seed": GAME_SEED,
        "progress_curves": [t.get("progress_curve", []) for t in results["task_logs"]],
        "clips": clips,
    }
    (out / "clips.json").write_text(json.dumps(manifest))
    print(out / "clips.json")


if __name__ == "__main__":
    main()
