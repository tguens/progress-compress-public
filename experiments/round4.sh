# Round 4 of the Task 1 search: the progress phase, 3 seeds, 5 tasks.
# Rounds 2 and 3 showed that the teacher is underfit: at 50 progress epochs the
# train accuracy still rises and there is no train/eval gap. 100 epochs give
# 0.10 to 0.15 more teacher accuracy. Forgetting per task scales with how poorly
# the task was learned. Base = round 3 base: clip 0.01, lambda 1000, 120 compress epochs.
cd "$(dirname "$0")/.."
SWEEP_DIR="outputs/experiments/round4"
COMMON="max_n_tasks=5 multi_head=true reinit=true compress_teacher=frozen_teacher"
source experiments/parallel.sh

BASE="progress_optimizer=adam progress_lr=1e-3 n_epochs_progress=50 compress_optimizer=sgd sgd_momentum=0.9 compress_lr=0.01 ewc_lambda=1000 fisher_clip=0.01 gamma=0.99 tau=2 n_epochs_compress=120 normalize_fisher=false data_augmentation=true"

cfg() {
    local tag="$1"; shift
    for seed in 0 1 2; do
        run "${tag}_s${seed}" $BASE "$@" seed=$seed
    done
}

# Progress epochs
cfg "ep100"                   n_epochs_progress=100
cfg "ep200"                   n_epochs_progress=200
# Progress learning rate
cfg "plr3e-3"                 progress_lr=3e-3
cfg "plr3e-3_ep100"           progress_lr=3e-3 n_epochs_progress=100
# Longer compress for the sharper teacher
cfg "ep100_ec200"             n_epochs_progress=100 n_epochs_compress=200
cfg "ep200_ec200"             n_epochs_progress=200 n_epochs_compress=200
cfg "plr3e-3_ep100_ec200"     progress_lr=3e-3 n_epochs_progress=100 n_epochs_compress=200
# EWC strength for the sharper teacher (larger Fisher)
cfg "ep100_lam300"            n_epochs_progress=100 ewc_lambda=300
cfg "ep100_lam3000"           n_epochs_progress=100 ewc_lambda=3000
cfg "ep100_clip0.1_lam100"    n_epochs_progress=100 fisher_clip=0.1 ewc_lambda=100

launch
