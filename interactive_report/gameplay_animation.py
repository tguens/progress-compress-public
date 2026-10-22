"""AI-generated script. Gameplay animation: P&C and fine-tuning learn Pong, then Boxing.

Reads the clips that `experiments/render_clips.py` made from the checkpoints of the
two runs of `experiments/atari_viz.sh` (`gameplay/pc/`, `gameplay/finetune/`: one PNG
sprite sheet of 84x84 grayscale frames per clip, plus `clips.json`), and writes one
matplotlib animation:

- top: four screens, P&C (Pong, Boxing) and fine-tuning (Pong, Boxing), each playing
  the greedy policy saved at the current checkpoint. P&C: the active column while it
  learns a game, the knowledge base (KB) for the games it remembers. Fine-tuning:
  the one network, with the head of the game shown.
- bottom: the mean return over 10 greedy games at each checkpoint (`eval_mean`; the
  single games as faint dots), against PPO steps, with a cursor at the current
  checkpoint. One greedy game alone can be a lucky one.

Usage, from the repository root:
    uv run interactive_report/gameplay_animation.py --out interactive_report/gameplay.gif
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402
from PIL import Image  # noqa: E402

GAMES = ["Pong-v5", "Boxing-v5"]
LEARNERS = [("pc", "P&C"), ("finetune", "Fine-tuning")]
# Default color cycle: C0 = the P&C knowledge base, C1 = the P&C active column, C3 = fine-tuning
COLOR = {"kb": "C0", "active": "C1", "finetune": "C3"}


def load(data: Path, label: str) -> dict:
    """The manifest of one learner, with every clip's frames [n, 84, 84] loaded."""
    m = json.load(open(data / label / "clips.json"))
    for c in m["clips"]:
        sheet = np.asarray(Image.open(data / label / c["file"]))
        rows = [sheet[r : r + 84] for r in range(0, sheet.shape[0], 84)]
        frames = [row[:, j : j + 84] for row in rows for j in range(0, sheet.shape[1], 84)]
        c["frames"] = np.stack(frames[: c["n_frames"]])
    return m


def stops(m: dict, every: list[int]) -> list[tuple[int, int, bool]]:
    """(task, step, end_of_task) of the shown checkpoints: every `every[k]`-th
    progress checkpoint of task k, its last one, then the end of the task."""
    out = []
    for k in range(len(m["games"])):
        steps = sorted({c["step"] for c in m["clips"] if c["task"] == k and c["role"] == "learner"})
        keep = steps[:: every[k]] + ([steps[-1]] if (len(steps) - 1) % every[k] else [])
        out += [(k, s, False) for s in keep] + [(k, m["budgets"][k], True)]
    return out


def clip_for(m: dict, label: str, stop, game: str):
    """The clip a screen shows at a stop, and who plays it, or (None, None)."""
    k, step, end = stop
    games = m["games"]
    seen = games[: k + 1] if end else games[:k]

    def find(task, role, at_end, s=None):
        for c in m["clips"]:
            if c["task"] == task and c["role"] == role and c["game"] == game and c["end_of_task"] == at_end and (s is None or c["step"] == s):
                return c
        return None

    if not end and game == games[k]:
        return find(k, "learner", False, step), ("active" if label == "pc" else "finetune")
    if game in seen:
        if end:
            return find(k, "memory", True), ("kb" if label == "pc" else "finetune")
        if label == "pc":  # the KB is frozen during progress: its clip from the end of the last task
            return find(k - 1, "memory", True), "kb"
        return find(k, "memory", False, step), "finetune"
    return None, None


WHO = {"active": "Active column", "kb": "Knowledge base", "finetune": "Network"}


def main():
    parser = argparse.ArgumentParser(description="Gameplay animation of P&C and fine-tuning")
    here = Path(__file__).resolve().parent
    parser.add_argument("--data", type=Path, default=here / "gameplay")
    parser.add_argument("--out", type=Path, default=here / "gameplay.gif")
    parser.add_argument("--every", type=int, nargs="+", default=[2, 1], help="Per task: show every n-th checkpoint (they are 250 k steps apart)")
    parser.add_argument("--frames-per-stop", type=int, nargs="+", default=[12, 10], help="Per task: animation frames per checkpoint")
    parser.add_argument("--end-frames", type=int, nargs="+", default=[20, 50], help="Per task: animation frames at the end of the task")
    # Fine-tuning starts to lose Pong at 0.5 M steps of Boxing (half of its 10 games): slow down from there.
    parser.add_argument("--focus", type=int, nargs=3, default=[1, 500_000, 40], metavar=("TASK", "STEP", "FRAMES"),
                        help="From STEP of TASK on, FRAMES animation frames per checkpoint")
    parser.add_argument("--stride", type=int, default=2, help="Game frames per animation frame")
    parser.add_argument("--start", type=int, default=40, help="First game frame shown (skips the serve)")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    ms = {label: load(args.data, label) for label, _ in LEARNERS}
    budgets = ms["pc"]["budgets"]
    stop_list = stops(ms["pc"], args.every)
    # One entry per animation frame: (stop index, frame index within the stop)
    timeline = []
    for si, (k, step, end) in enumerate(stop_list):
        if end:
            n_frames = args.end_frames[k]
        elif k == args.focus[0] and step >= args.focus[1]:
            n_frames = args.focus[2]
        else:
            n_frames = args.frames_per_stop[k]
        timeline += [(si, f) for f in range(n_frames)]

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "normal"})
    fig = plt.figure(figsize=(8.0, 5.4), dpi=150)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.62], hspace=0.5, wspace=0.12, left=0.07, right=0.98, top=0.82, bottom=0.11)
    screens, labels = {}, {}
    for i, (label, name) in enumerate(LEARNERS):
        for j, game in enumerate(GAMES):
            ax = fig.add_subplot(gs[0, 2 * i + j])
            im = ax.imshow(np.zeros((84, 84)), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(game.split("-")[0], fontsize=8.5, color="0.25", pad=3)
            screens[(label, game)] = (ax, im)
            labels[(label, game)] = ax.text(0.5, -0.07, "", transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="0.2")
        # One header per learner, over its two screens
        left, right = screens[(label, GAMES[0])][0].get_position(), screens[(label, GAMES[1])][0].get_position()
        fig.text((left.x0 + right.x1) / 2, left.y1 + 0.045, name, ha="center", va="bottom", fontsize=10.5, weight="bold")

    # Bottom: the mean return over 10 greedy games at each checkpoint, per game
    charts = {}
    cursor_x = lambda stop: sum(budgets[: stop[0]]) + stop[1]  # noqa: E731

    def series(clips, color, ax, step=False, **kw):
        """Mean line over the clips' checkpoints, the single games as faint dots."""
        clips = sorted(clips, key=lambda c: c["global_step"])
        xs = [c["global_step"] for c in clips]
        for c in clips:
            ax.scatter([c["global_step"]] * len(c["eval_returns"]), c["eval_returns"], s=5, color=color, alpha=0.18, lw=0)
        plot = ax.step if step else ax.plot
        plot(xs, [c["eval_mean"] for c in clips], color=color, lw=1.3, **({"where": "post"} if step else {}), **kw)

    for j, game in enumerate(GAMES):
        ax = fig.add_subplot(gs[1, 2 * j : 2 * j + 2])
        for label, name in LEARNERS:
            m = ms[label]
            of_game = [c for c in m["clips"] if c["game"] == game]
            if label == "pc":
                series([c for c in of_game if c["role"] == "learner"], COLOR["active"], ax, marker=".", label="P&C, active column")
                series([c for c in of_game if c["end_of_task"]], COLOR["kb"], ax, step=True, marker="o", ms=4, label="P&C, knowledge base")
            else:
                series(of_game, COLOR["finetune"], ax, marker=".", label="Fine-tuning")
        ax.axvline(budgets[0], color="0.6", ls="--", lw=0.8)
        ax.set_xlim(0, sum(budgets))
        ax.set_xticks(np.arange(0, sum(budgets) + 1, 1_000_000))
        ax.set_xticklabels([f"{x // 1_000_000}M" for x in range(0, sum(budgets) + 1, 1_000_000)], fontsize=7.5)
        ax.tick_params(labelsize=7.5)
        ax.set_title(f"{game.split('-')[0]}: mean return over 10 games", fontsize=8.5, loc="left")
        ax.set_xlabel("Training steps", fontsize=7.5)
        if j == 0:
            ax.set_ylabel("Return", fontsize=7.5)
        ax.grid(alpha=0.3)
        charts[game] = ax.axvline(0, color="k", lw=0.8)
        if j == 0:
            ax.legend(fontsize=6.5, loc="lower right")
        else:
            # Leave room for the y tick labels of the right chart
            pos = ax.get_position()
            ax.set_position([pos.x0 + 0.03, pos.y0, pos.width - 0.03, pos.height])
    fig.suptitle("P&C vs full fine tuning on Boxing + Pong", fontsize=12, weight="bold", y=0.975)
    title = fig.text(0.5, 0.905, "", ha="center", fontsize=9.5, color="0.3")
    caption = fig.text(0.5, 0.015, "", ha="center", fontsize=8, color="0.25")

    def stop_text(stop):
        k, step, end = stop
        game = ms["pc"]["games"][k].split("-")[0]
        if end:
            return f"{game} complete: P&C compresses it into the knowledge base"
        return f"Training on {game}  ·  step {step / 1e6:.2f}M of {budgets[k] / 1e6:.0f}M"

    def caption_text(stop):
        parts = []
        for label, name in LEARNERS:
            bits = []
            for game in GAMES:
                c, who = clip_for(ms[label], label, stop, game)
                if c is not None:
                    bits.append(f"{game.split('-')[0]} {c['eval_mean']:+.0f}".replace("-", "\u2212"))
            parts.append(f"{name}: " + ", ".join(bits))
        return "Mean return over 10 games    " + "    ·    ".join(parts)

    def update(t):
        s, f = timeline[t]
        stop = stop_list[s]
        title.set_text(stop_text(stop))
        caption.set_text(caption_text(stop))
        for label, _ in LEARNERS:
            for game in GAMES:
                ax, im = screens[(label, game)]
                c, who = clip_for(ms[label], label, stop, game)
                if c is None:
                    im.set_data(np.full((84, 84), 235, dtype=np.uint8))
                    labels[(label, game)].set_text("Not trained yet")
                    color = "0.8"
                else:
                    i = min(args.start + f * args.stride, len(c["frames"]) - 1)
                    im.set_data(c["frames"][i])
                    frozen = " (frozen)" if who == "kb" and not stop[2] else ""
                    labels[(label, game)].set_text(f"{WHO[who]}{frozen}  ·  score {c['scores'][i]:+.0f}".replace("-", "\u2212"))
                    color = COLOR[who]
                for sp in ax.spines.values():
                    sp.set_color(color)
                    sp.set_linewidth(2)
        for line in charts.values():
            line.set_xdata([cursor_x(stop)] * 2)
        return []

    anim = FuncAnimation(fig, update, frames=len(timeline), interval=1000 / args.fps, blit=False)
    if args.out.suffix == ".gif" and FFMpegWriter.isAvailable():
        # ffmpeg builds one small palette from the lossless frames; PillowWriter
        # gives a file about 10 times larger.
        palette = "split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle"
        writer = FFMpegWriter(fps=args.fps, codec="gif", extra_args=["-filter_complex", palette])
    elif args.out.suffix == ".gif":
        writer = PillowWriter(fps=args.fps)
    else:
        writer = FFMpegWriter(fps=args.fps, bitrate=1800)
    anim.save(args.out, writer=writer)
    print(args.out)


if __name__ == "__main__":
    main()
