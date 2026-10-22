# Gameplay animation: P&C and fine-tuning learn Pong, then Boxing, with the agent saved
# every 250 k PPO steps (`checkpoint_every`) and after each task. The demo's settings
# (experiments/atari_teachers.sh and atari_best.sh): Pong 3 M PPO steps, Boxing 2 M,
# buffer compress with 4 DAgger rounds of 16384 states, EWC lambda 1e4, Fisher clip
# 1e-3, greedy evaluation. Fine-tuning is one network trained by PPO on both games in
# turn (mode=finetune). Then experiments/render_clips.py turns the checkpoints into clips.
# About 1 h, 2 runs in parallel.
cd "$(dirname "$0")/.."
MAIN="src/run_atari.py"
SWEEP_DIR="outputs/atari/viz"
COMMON="games=[Pong-v5,Boxing-v5] max_n_tasks=2 total_timesteps=[3000000,2000000] compress_n_rounds=4 compress_n_states=16384 ewc_lambda=10000 fisher_clip=1e-3 compress_max_grad_norm=0.5 eval_mode=greedy checkpoint_every=250000 seed=0 wandb=true"
source experiments/parallel.sh

run "pc_s0" mode=pc
run "finetune_s0" mode=finetune

JOBS=2 launch
