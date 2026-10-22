# Fine-tuning baseline for the Omniglot final figure: experiments/best.sh with
# ewc_lambda=0 (no penalty), 10 tasks, 3 seeds. Same everything else.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
SWEEP_DIR="outputs/experiments/final"
COMMON="max_n_tasks=10 multi_head=true reinit=true compress_teacher=frozen_teacher"
source experiments/parallel.sh

BEST="progress_optimizer=adam progress_lr=1e-3 n_epochs_progress=400 compress_optimizer=sgd sgd_momentum=0.9 compress_lr=0.01 n_epochs_compress=120 ewc_lambda=0 fisher_clip=0.01 gamma=0.99 tau=2 normalize_fisher=false data_augmentation=true"

for seed in 0 1 2; do
    run "ft_lam0_s${seed}" $BEST seed=$seed
done

JOBS=3 launch
