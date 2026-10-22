# Task 3, step 1: the teachers. The same run as atari_paper2.sh (the paper's 6 games, 2 visits
# of 10 M PPO steps per game, 3 seeds, teachers saved) under a new tag, because the paper2
# teacher snapshots were lost with the pod disk. Logged to Weights & Biases (group paper3).
# About 23 h per run, 3 in parallel. Pull the teacher_T*.pt files with rsync as each run ends.
cd "$(dirname "$0")/.."
MAIN="src/run_atari.py"
SWEEP_DIR="outputs/atari/paper3"
G="SpaceInvaders-v5,Krull-v5,BeamRider-v5,Hero-v5,StarGunner-v5,MsPacman-v5"
T="10000000,10000000,10000000,10000000,10000000,10000000"
COMMON="mode=pc games=[$G,$G] max_n_tasks=12 total_timesteps=[$T,$T] num_envs=16 compress_n_rounds=4 compress_n_states=16384 ewc_lambda=10000 fisher_clip=1e-3 compress_max_grad_norm=0.5 eval_mode=greedy save_teachers=true wandb=true"
source experiments/parallel.sh

for seed in 0 1 2; do
    run "pc_s${seed}" seed=$seed
done

JOBS=3 launch
