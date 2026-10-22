# Paper reproduction: Progress & Compress on the paper's 6 games (Space Invaders, Krull,
# Beamrider, Hero, Stargunner, Ms. Pac-man), 2 visits of 10 M PPO steps (40 M frames) per
# game, 3 seeds. The paper: 5 visits of 50 M frames, 4 seeds. Compress: 4 DAgger rounds of
# 16384 states, SGD momentum 0.9, lr 0.01, lambda 1e4, Fisher clip 1e-3, gradient norm
# clip 0.5 (without it the Hero compress phase diverged on 3 of 3 seeds), greedy evaluation.
# 16 envs (5197 steps per second alone against 3582 with 8). Teachers saved.
cd "$(dirname "$0")/.."
MAIN="src/run_atari.py"
SWEEP_DIR="outputs/atari/paper2"
G="SpaceInvaders-v5,Krull-v5,BeamRider-v5,Hero-v5,StarGunner-v5,MsPacman-v5"
T="10000000,10000000,10000000,10000000,10000000,10000000"
COMMON="mode=pc games=[$G,$G] max_n_tasks=12 total_timesteps=[$T,$T] num_envs=16 compress_n_rounds=4 compress_n_states=16384 ewc_lambda=10000 fisher_clip=1e-3 compress_max_grad_norm=0.5 eval_mode=greedy save_teachers=true"
source experiments/parallel.sh

for seed in 0 1 2; do
    run "pc_s${seed}" seed=$seed
done

JOBS=3 launch
