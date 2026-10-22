# Round 3 of the Task 1 search: compress head warm-up, 3 seeds, 5 tasks.
# Round 2 showed that the Fisher clip removes the NaN but retention stays at
# prev_avg_acc 0.52 to 0.59 for lambda 300 to 3000. The initial KL of every
# compress phase is 8 to 9 because the KB head is new and random, and the first
# epochs distort the backbone. compress_head_warmup trains the head alone first.
# Base = round 2 best: clip 0.01, lambda 1000, 120 compress epochs.
cd "$(dirname "$0")/.."
SWEEP_DIR="outputs/experiments/round3"
COMMON="max_n_tasks=5 multi_head=true reinit=true compress_teacher=frozen_teacher progress_optimizer=adam progress_lr=1e-3"
source experiments/parallel.sh

BASE="compress_optimizer=sgd sgd_momentum=0.9 compress_lr=0.01 ewc_lambda=1000 fisher_clip=0.01 gamma=0.99 tau=2 n_epochs_compress=120 normalize_fisher=false data_augmentation=true"

cfg() {
    local tag="$1"; shift
    for seed in 0 1 2; do
        run "${tag}_s${seed}" $BASE "$@" seed=$seed
    done
}

# Warm-up length
cfg "w0"                      compress_head_warmup=0
cfg "w10"                     compress_head_warmup=10
cfg "w20"                     compress_head_warmup=20
cfg "w40"                     compress_head_warmup=40
# Warm-up with more compress epochs
cfg "w20_ec200"               compress_head_warmup=20 n_epochs_compress=200
cfg "w40_ec200"               compress_head_warmup=40 n_epochs_compress=200
# Lower compress lr with more epochs, with and without warm-up
cfg "clr0.003_ec240"          compress_lr=0.003 n_epochs_compress=240
cfg "clr0.003_ec240_w40"      compress_lr=0.003 n_epochs_compress=240 compress_head_warmup=40
# Warm-up with other EWC strengths and the README base
cfg "w20_lam0"                compress_head_warmup=20 ewc_lambda=0
cfg "w20_lam300"              compress_head_warmup=20 ewc_lambda=300
cfg "w20_lam3000"             compress_head_warmup=20 ewc_lambda=3000
cfg "w20_clip0.1_lam300"      compress_head_warmup=20 fisher_clip=0.1 ewc_lambda=300
cfg "w20_noclip_lam100"       compress_head_warmup=20 fisher_clip=null ewc_lambda=100
cfg "w20_g1.0"                compress_head_warmup=20 gamma=1.0

launch
