cd "$(dirname "$0")/.."
SWEEP_DIR="outputs/experiments"
COMMON="max_n_tasks=5 seed=0 multi_head=true reinit=true data_augmentation=true compress_teacher=frozen_teacher normalize_fisher=false"

run() {
    tag="$1"; shift
    if [ -f "$SWEEP_DIR/$tag/results.json" ]; then
        echo "SKIP $tag"; return
    fi
    echo ">>> $tag"
    uv run src/run_omniglot.py $COMMON "$@" \
        "hydra.run.dir=$SWEEP_DIR/$tag" 2>&1 | tail -5
}

# Base config: SGD compress, momentum=0.9, lambda=100
run "sgd_mom0.9_lam100" \
    progress_optimizer=adam compress_optimizer=sgd sgd_momentum=0.9 \
    progress_lr=1e-3 compress_lr=0.01 ewc_lambda=100

# Ablation: Adam compress (needs higher lambda)
run "adam_lam1000" \
    progress_optimizer=adam compress_optimizer=adam \
    progress_lr=1e-3 compress_lr=5e-4 ewc_lambda=1000

# Ablation: no data augmentation
run "sgd_mom0.9_lam100_noaug" \
    progress_optimizer=adam compress_optimizer=sgd sgd_momentum=0.9 \
    progress_lr=1e-3 compress_lr=0.01 ewc_lambda=100 \
    data_augmentation=false

# Ablation: no multi-head
run "sgd_mom0.9_lam100_singlehead" \
    progress_optimizer=adam compress_optimizer=sgd sgd_momentum=0.9 \
    progress_lr=1e-3 compress_lr=0.01 ewc_lambda=100 \
    multi_head=false
