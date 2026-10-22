# The compress budget and penalty controls, on the
# archived implementation (src/main_atari.py on dev) and the step 1 teachers.
# 2 penalty settings (lambda 1e4 / clip 1e-3, the original; lambda 3e4 / clip 3e-4, the
# selected online setting) x 2 buffer budgets (10 and 30 epochs per accumulated buffer,
# 6400 and 19200 SGD updates per task) x 3 seeds, plus the online protocol at the
# original penalty on seeds 1 and 2 (seed 0 exists: on5M_lam1e4_s0). The lambda 1e4,
# 10-epoch chains repeat the archived pc_s* compress phases and give the rerun spread.
# 3 chains in parallel beside the full runs; about 25 to 30 min per buffer chain, 4 h
# per online chain.
cd "$(dirname "$0")/.."
MAIN="src/run_atari.py"  # the archived chains ran with main_atari.py, the entry point before the cleanup
SWEEP_DIR="outputs/atari/paper3"
G="SpaceInvaders-v5,Krull-v5,BeamRider-v5,Hero-v5,StarGunner-v5,MsPacman-v5"
T="10000000,10000000,10000000,10000000,10000000,10000000"
COMMON="mode=pc games=[$G,$G] max_n_tasks=12 total_timesteps=[$T,$T] num_envs=16 num_steps=128 compress_n_rounds=4 compress_n_states=16384 compress_batch_size=256 sgd_momentum=0.9 compress_max_grad_norm=0.5 eval_mode=greedy wandb=true"
source experiments/parallel.sh

for seed in 0 1 2; do
    TD="teachers_dir=outputs/atari/paper3/pc_s${seed}"
    run "buf_lam1e4_ep10_s${seed}" $TD seed=$seed compress_online=false ewc_lambda=10000 fisher_clip=1e-3 n_epochs_compress=10
    run "buf_lam3e4_ep10_s${seed}" $TD seed=$seed compress_online=false ewc_lambda=30000 fisher_clip=3e-4 n_epochs_compress=10
    run "buf_lam1e4_ep30_s${seed}" $TD seed=$seed compress_online=false ewc_lambda=10000 fisher_clip=1e-3 n_epochs_compress=30
    run "buf_lam3e4_ep30_s${seed}" $TD seed=$seed compress_online=false ewc_lambda=30000 fisher_clip=3e-4 n_epochs_compress=30
done
for seed in 1 2; do
    run "on5M_lam1e4_s${seed}" teachers_dir=outputs/atari/paper3/pc_s${seed} seed=$seed compress_online=true compress_online_steps=5000000 compress_online_epochs=1 n_epochs_compress=10 ewc_lambda=10000 fisher_clip=1e-3
done

JOBS=3 launch
