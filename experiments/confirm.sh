# Confirmation of the Task 1 search: the top 5 configurations of rounds 4 and 5
# on new seeds 3, 4 and 5, 5 tasks. Seeds 0 to 2 are in round4/ and round5/.
cd "$(dirname "$0")/.."
SWEEP_DIR="outputs/experiments/confirm"
COMMON="max_n_tasks=5 multi_head=true reinit=true compress_teacher=frozen_teacher"
source experiments/parallel.sh

BASE="progress_optimizer=adam progress_lr=1e-3 compress_optimizer=sgd sgd_momentum=0.9 compress_lr=0.01 ewc_lambda=1000 fisher_clip=0.01 gamma=0.99 tau=2 n_epochs_compress=120 normalize_fisher=false data_augmentation=true"

cfg() {
    local tag="$1"; shift
    for seed in 3 4 5; do
        run "${tag}_s${seed}" $BASE "$@" seed=$seed
    done
}

cfg "ep400"          n_epochs_progress=400
cfg "ep600"          n_epochs_progress=600
cfg "ep400_ec200"    n_epochs_progress=400 n_epochs_compress=200
cfg "ep200"          n_epochs_progress=200
cfg "ep300"          n_epochs_progress=300

launch
