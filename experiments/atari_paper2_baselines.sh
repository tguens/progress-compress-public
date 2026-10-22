# Paper reproduction: the single-network baselines on the same 12-task sequence as
# atari_paper2.sh (6 games, 2 visits of 10 M steps). mode=finetune: PPO with no
# penalty. mode=ewc: PPO with the online EWC penalty (lambda from the probe,
# experiments/atari_ewc_probe.sh). 16 envs, greedy evaluation, 3 seeds.
cd "$(dirname "$0")/.."
MAIN="src/run_atari.py"
SWEEP_DIR="outputs/atari/paper2"
G="SpaceInvaders-v5,Krull-v5,BeamRider-v5,Hero-v5,StarGunner-v5,MsPacman-v5"
T="10000000,10000000,10000000,10000000,10000000,10000000"
COMMON="games=[$G,$G] max_n_tasks=12 total_timesteps=[$T,$T] num_envs=16 eval_mode=greedy compress_n_states=16384 fisher_n_samples=1024 fisher_clip=1e-3"
source experiments/parallel.sh

# Online EWC lambda 1e5: the probe's best mean final score over its 3 games normalised by
# the single-task runs (0.20 against 0.08 to 0.13 for lambda 1e3, 1e4, 3e4), the paper's
# criterion ("maximum final score across all games"). It keeps Space Invaders at 0.70 of
# single-task and learns little of the new games, the negative transfer the paper reports.
for seed in 0 1 2; do
    run "finetune_s${seed}" mode=finetune seed=$seed
    run "ewc_lam1e5_s${seed}" mode=ewc ewc_lambda=100000 seed=$seed
done

JOBS=6 launch
