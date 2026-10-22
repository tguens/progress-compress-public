# The final Omniglot configuration (search of 2026-10-16): 3 seeds, 10 tasks.
# Adam progress lr 1e-3 for 400 epochs; SGD momentum 0.9 compress lr 0.01 for
# 120 epochs; EWC lambda 500 with the Fisher clipped at 0.01; gamma 0.99; tau 2;
# multi-head; augmentation; frozen teacher. Lambda 1000 won the 5-task search
# but diverged in 1 of 3 seeds at 10 tasks (lr * lambda * clip * n_tasks = 1);
# so this script runs the final configuration only.
cd "$(dirname "$0")/.."
SWEEP_DIR="outputs/experiments/final"
COMMON="max_n_tasks=10 multi_head=true reinit=true compress_teacher=frozen_teacher"
source experiments/parallel.sh

BEST="progress_optimizer=adam progress_lr=1e-3 n_epochs_progress=400 compress_optimizer=sgd sgd_momentum=0.9 compress_lr=0.01 n_epochs_compress=120 ewc_lambda=500 fisher_clip=0.01 gamma=0.99 tau=2 normalize_fisher=false data_augmentation=true"

for seed in 0 1 2; do
    run "best_s${seed}" $BEST seed=$seed
done

launch
