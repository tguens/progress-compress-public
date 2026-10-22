# Round 1 of the Task 1 search: 27 configurations x 2 seeds, 5 tasks.
# Base = README best run. Each tag names only what differs from the base.
cd "$(dirname "$0")/.."
SWEEP_DIR="outputs/experiments/round1"
COMMON="max_n_tasks=5 multi_head=true reinit=true compress_teacher=frozen_teacher progress_optimizer=adam progress_lr=1e-3"
source experiments/parallel.sh

BASE="compress_optimizer=sgd sgd_momentum=0.9 compress_lr=0.01 ewc_lambda=100 gamma=0.99 tau=2 n_epochs_compress=60 normalize_fisher=false data_augmentation=true"

# cfg <tag> <overrides that replace base values>
cfg() {
    local tag="$1"; shift
    for seed in 0 1; do
        run "${tag}_s${seed}" $BASE "$@" seed=$seed
    done
}

# Baselines
cfg "base"
cfg "lam0"              ewc_lambda=0
# Compress optimizer and learning rate
cfg "clr0.001"          compress_lr=0.001
cfg "clr0.003"          compress_lr=0.003
cfg "clr0.03"           compress_lr=0.03
cfg "mom0.5"            sgd_momentum=0.5
cfg "adam_clr5e-4_lam100"   compress_optimizer=adam compress_lr=5e-4
cfg "adam_clr5e-4_lam1000"  compress_optimizer=adam compress_lr=5e-4 ewc_lambda=1000
cfg "adam_clr1e-4_lam1000"  compress_optimizer=adam compress_lr=1e-4 ewc_lambda=1000
cfg "adam_clr5e-4_lam1e4"   compress_optimizer=adam compress_lr=5e-4 ewc_lambda=10000
# ewc_lambda
cfg "lam10"             ewc_lambda=10
cfg "lam30"             ewc_lambda=30
cfg "lam300"            ewc_lambda=300
cfg "lam1000"           ewc_lambda=1000
cfg "clr0.003_lam300"   compress_lr=0.003 ewc_lambda=300
# gamma
cfg "g0.5"              gamma=0.5
cfg "g0.9"              gamma=0.9
cfg "g1.0"              gamma=1.0
# tau
cfg "tau1"              tau=1
cfg "tau4"              tau=4
# Epochs
cfg "ec30"              n_epochs_compress=30
cfg "ec120"             n_epochs_compress=120
cfg "ep100"             n_epochs_progress=100
# Fisher normalization (total mass 1, so lambda must be large)
cfg "nf_lam1e4"         normalize_fisher=true ewc_lambda=1e4
cfg "nf_lam1e5"         normalize_fisher=true ewc_lambda=1e5
cfg "nf_lam1e6"         normalize_fisher=true ewc_lambda=1e6
# Augmentation
cfg "noaug"             data_augmentation=false

launch
