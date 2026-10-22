# Round 2 of the Task 1 search: Fisher clipping x ewc_lambda, 2 seeds, 5 tasks.
# Round 1 showed that lambda >= 300 diverges at compress_lr 0.01 and that
# lambda in 10..100 gives the same retention. The Fisher is heavy-tailed, so
# clip it and raise lambda. Base = README best run, as in round1.sh.
cd "$(dirname "$0")/.."
SWEEP_DIR="outputs/experiments/round2"
COMMON="max_n_tasks=5 multi_head=true reinit=true compress_teacher=frozen_teacher progress_optimizer=adam progress_lr=1e-3"
source experiments/parallel.sh

BASE="compress_optimizer=sgd sgd_momentum=0.9 compress_lr=0.01 ewc_lambda=100 gamma=0.99 tau=2 n_epochs_compress=60 normalize_fisher=false data_augmentation=true"

cfg() {
    local tag="$1"; shift
    for seed in 0 1; do
        run "${tag}_s${seed}" $BASE "$@" seed=$seed
    done
}

# Reference without clipping: the two round 1 winners together
cfg "g1.0_ec120"                 gamma=1.0 n_epochs_compress=120
# Clip x lambda grid. Stability needs lr * lambda * clip * n_tasks < about 4.
cfg "clip1_lam100"               fisher_clip=1 ewc_lambda=100
cfg "clip1_lam300"               fisher_clip=1 ewc_lambda=300
cfg "clip0.1_lam100"             fisher_clip=0.1 ewc_lambda=100
cfg "clip0.1_lam300"             fisher_clip=0.1 ewc_lambda=300
cfg "clip0.1_lam1000"            fisher_clip=0.1 ewc_lambda=1000
cfg "clip0.01_lam100"            fisher_clip=0.01 ewc_lambda=100
cfg "clip0.01_lam300"            fisher_clip=0.01 ewc_lambda=300
cfg "clip0.01_lam1000"           fisher_clip=0.01 ewc_lambda=1000
cfg "clip0.01_lam3000"           fisher_clip=0.01 ewc_lambda=3000
cfg "clip0.01_lam1e4"            fisher_clip=0.01 ewc_lambda=1e4
cfg "clip0.001_lam1000"          fisher_clip=0.001 ewc_lambda=1000
cfg "clip0.001_lam3000"          fisher_clip=0.001 ewc_lambda=3000
cfg "clip0.001_lam1e4"           fisher_clip=0.001 ewc_lambda=1e4
cfg "clip0.001_lam3e4"           fisher_clip=0.001 ewc_lambda=3e4
# Clipping with the round 1 winners
cfg "clip0.01_lam1000_g1.0"      fisher_clip=0.01 ewc_lambda=1000 gamma=1.0
cfg "clip0.01_lam1000_ec120"     fisher_clip=0.01 ewc_lambda=1000 n_epochs_compress=120
cfg "clip0.01_lam1000_g1.0_ec120" fisher_clip=0.01 ewc_lambda=1000 gamma=1.0 n_epochs_compress=120
cfg "clip0.01_lam3000_g1.0_ec120" fisher_clip=0.01 ewc_lambda=3000 gamma=1.0 n_epochs_compress=120
cfg "clip0.001_lam1e4_g1.0_ec120" fisher_clip=0.001 ewc_lambda=1e4 gamma=1.0 n_epochs_compress=120

launch
