# Paper reproduction: single-task runs, the normaliser of every Atari plot (the paper
# normalises by "a single model trained on each task"). One Nature CNN trained by PPO on
# one game (mode=finetune, 1 game), 10 M steps, 16 envs, greedy evaluation, 3 seeds.
cd "$(dirname "$0")/.."
MAIN="src/run_atari.py"
SWEEP_DIR="outputs/atari/single"
COMMON="mode=finetune max_n_tasks=1 total_timesteps=[10000000] num_envs=16 eval_mode=greedy compress_n_states=16384 fisher_n_samples=1024"
source experiments/parallel.sh

for seed in 0 1 2; do
    for game in SpaceInvaders Krull BeamRider Hero StarGunner MsPacman; do
        run "${game}_s${seed}" "games=[${game}-v5]" seed=$seed
    done
done

JOBS=4 launch
