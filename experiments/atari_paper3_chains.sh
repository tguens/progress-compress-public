# Task 3, step 4: the pilot of the online compress phase. Compress-only chains over the
# 12-task sequence (6 games, 2 visits) on the teachers saved by atari_paper3_teachers.sh.
# Round 0 as before (16384 teacher states, 10 epochs), then the online phase: the KB policy
# acts, the teacher labels, one SGD pass per rollout of 16 x 128 states.
# Two budgets (2 M and 5 M env steps per compress phase) and two lambda/clip pairs that keep
# lambda * clip * n_tasks * lr near 1. Baseline: the buffer chains pc_s{0,1} of step 1.
cd "$(dirname "$0")/.."
MAIN="src/run_atari.py"
SWEEP_DIR="outputs/atari/paper3"
G="SpaceInvaders-v5,Krull-v5,BeamRider-v5,Hero-v5,StarGunner-v5,MsPacman-v5"
T="10000000,10000000,10000000,10000000,10000000,10000000"
COMMON="mode=pc games=[$G,$G] max_n_tasks=12 total_timesteps=[$T,$T] num_envs=16 compress_n_states=16384 compress_online=true compress_online_epochs=1 compress_max_grad_norm=0.5 eval_mode=greedy wandb=true"
source experiments/parallel.sh

for seed in 0 1; do
    run "on2M_lam1e4_s${seed}" teachers_dir=outputs/atari/paper3/pc_s${seed} seed=$seed compress_online_steps=2000000 ewc_lambda=10000 fisher_clip=1e-3
    run "on2M_lam3e4_s${seed}" teachers_dir=outputs/atari/paper3/pc_s${seed} seed=$seed compress_online_steps=2000000 ewc_lambda=30000 fisher_clip=3e-4
done
run "on5M_lam1e4_s0" teachers_dir=outputs/atari/paper3/pc_s0 seed=0 compress_online_steps=5000000 ewc_lambda=10000 fisher_clip=1e-3
run "on5M_lam3e4_s0" teachers_dir=outputs/atari/paper3/pc_s0 seed=0 compress_online_steps=5000000 ewc_lambda=30000 fisher_clip=3e-4

JOBS=6 launch
