# Task 3, step 4 continued and step 5a: the online compress at lambda 3e4, clip 3e-4 won
# the pilot (experiments/atari_paper3_chains.sh, 2026-10-21): +0.12 to +0.22 on
# the mean normalised KB return after visit 2 over the buffer chains, retention kept. 5 M
# online steps beat 2 M by 0.10 on seed 0, so the step 4 rule asks for a 10 M run on seed 0
# before the budget is fixed. Alongside it, the 5 M chains of seeds 1 and 2 (step 5a for the
# 5 M budget; seed 0 exists). 3 chains in parallel, about 4 h (5 M) and 8 h (10 M).
cd "$(dirname "$0")/.."
MAIN="src/run_atari.py"
SWEEP_DIR="outputs/atari/paper3"
G="SpaceInvaders-v5,Krull-v5,BeamRider-v5,Hero-v5,StarGunner-v5,MsPacman-v5"
T="10000000,10000000,10000000,10000000,10000000,10000000"
COMMON="mode=pc games=[$G,$G] max_n_tasks=12 total_timesteps=[$T,$T] num_envs=16 compress_n_states=16384 compress_online=true compress_online_epochs=1 compress_max_grad_norm=0.5 eval_mode=greedy wandb=true ewc_lambda=30000 fisher_clip=3e-4"
source experiments/parallel.sh

run "on10M_lam3e4_s0" teachers_dir=outputs/atari/paper3/pc_s0 seed=0 compress_online_steps=10000000
run "on5M_lam3e4_s1" teachers_dir=outputs/atari/paper3/pc_s1 seed=1 compress_online_steps=5000000
run "on5M_lam3e4_s2" teachers_dir=outputs/atari/paper3/pc_s2 seed=2 compress_online_steps=5000000

JOBS=3 launch
