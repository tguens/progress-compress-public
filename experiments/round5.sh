# Round 5 of the Task 1 search: more progress epochs, 3 seeds, 5 tasks.
# Round 4: 50 -> 100 -> 200 progress epochs gives all_avg_acc 0.66 -> 0.75 -> 0.81.
# Base = round 4 best: 200 progress epochs, clip 0.01, lambda 1000, 120 compress epochs.
cd "$(dirname "$0")/.."
SWEEP_DIR="outputs/experiments/round5"
COMMON="max_n_tasks=5 multi_head=true reinit=true compress_teacher=frozen_teacher"
source experiments/parallel.sh

BASE="progress_optimizer=adam progress_lr=1e-3 n_epochs_progress=200 compress_optimizer=sgd sgd_momentum=0.9 compress_lr=0.01 ewc_lambda=1000 fisher_clip=0.01 gamma=0.99 tau=2 n_epochs_compress=120 normalize_fisher=false data_augmentation=true"

cfg() {
    local tag="$1"; shift
    for seed in 0 1 2; do
        run "${tag}_s${seed}" $BASE "$@" seed=$seed
    done
}

cfg "ep300"                   n_epochs_progress=300
cfg "ep400"                   n_epochs_progress=400
cfg "ep600"                   n_epochs_progress=600
cfg "ep400_ec200"             n_epochs_progress=400 n_epochs_compress=200
cfg "ep300_lam3000"           n_epochs_progress=300 ewc_lambda=3000

launch
