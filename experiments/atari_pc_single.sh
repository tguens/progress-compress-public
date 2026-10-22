# The matched architecture control for forward transfer.
# P&C on one game at a time with a frozen random KB: the active column and the adaptors
# train by PPO as in the full runs, against a KB that knows nothing. Same budget, envs and
# evaluation as the paper runs. The first-visit active returns of pc_s* and pc_online_s*
# are then compared with these, game by game. 6 games x 3 seeds = 18 runs, about 1 h each.
# Runs on the refactored code (E6 passed).
cd "$(dirname "$0")/.."
MAIN="src/run_atari.py"
SWEEP_DIR="outputs/atari/pc_single"
COMMON="mode=pc max_n_tasks=1 total_timesteps=[10000000] num_envs=16 num_steps=128 eval_mode=greedy compress_n_states=16384 fisher_n_samples=1024 wandb=true"
source experiments/parallel.sh

for seed in 0 1 2; do
    for game in SpaceInvaders Krull BeamRider Hero StarGunner MsPacman; do
        run "pcsingle_${game}_s${seed}" "games=[${game}-v5]" seed=$seed
    done
done

JOBS=${JOBS:-4} launch
