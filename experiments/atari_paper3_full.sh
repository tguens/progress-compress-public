# Task 3, step 5b: the full runs with the online compress. atari_paper2.sh (the paper's
# 6 games, 2 visits of 10 M PPO steps, 3 seeds) with compress_online=true at the budget
# and lambda that won step 4 (the pilot of 2026-10-21): 5 M online steps per compress phase,
# lambda 3e4, Fisher clip 3e-4 (10 M did not beat 5 M on seed 0: 0.645 against 0.663).
# In a chain the teachers were trained against the buffer KB; here the active column
# trains against the online KB, so these runs also measure forward transfer.
# About 35 h per run, 3 in parallel. Teachers saved, wandb on.
cd "$(dirname "$0")/.."
MAIN="src/run_atari.py"
SWEEP_DIR="outputs/atari/paper3"
G="SpaceInvaders-v5,Krull-v5,BeamRider-v5,Hero-v5,StarGunner-v5,MsPacman-v5"
T="10000000,10000000,10000000,10000000,10000000,10000000"
COMMON="mode=pc games=[$G,$G] max_n_tasks=12 total_timesteps=[$T,$T] num_envs=16 compress_n_states=16384 compress_online=true compress_online_steps=5000000 compress_online_epochs=1 ewc_lambda=30000 fisher_clip=3e-4 compress_max_grad_norm=0.5 eval_mode=greedy save_teachers=true wandb=true"
source experiments/parallel.sh

for seed in 0 1 2; do
    run "pc_online_s${seed}" seed=$seed
done

JOBS=3 launch
