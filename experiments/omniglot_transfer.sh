# The Omniglot transfer control: the active column reads a learned KB (the normal P&C run)
# or a frozen random KB (kb_random=true: no compress, no Fisher). The configuration of
# experiments/best.sh, 10 tasks, seeds 0 to 3. Both variants of a seed draw the new
# columns, the heads, the progress data order and the augmentation of task k from the
# stream of (seed, k), so a pair differs only in the KB. Each task evaluates the active
# column after the completed epochs 25, 50, 100, 200 and 400, plus every 5 epochs.
# Measure one run before you set the concurrency beside Atari work: JOBS=<n> overrides
# the job limit of experiments/parallel.sh. The learned run of seed 0 diverged at task 3
# (lambda 500 sits at the stability edge); the published pairs are seeds 1, 2 and 3, and
# seed 0 stays here as the failed attempt.
cd "$(dirname "$0")/.."
SWEEP_DIR="outputs/experiments/final"
MAIN="src/run_omniglot.py"
COMMON="max_n_tasks=10 multi_head=true reinit=true compress_teacher=frozen_teacher eval_epochs_progress=[25,50,100,200,400]"
source experiments/parallel.sh

BEST="progress_optimizer=adam progress_lr=1e-3 n_epochs_progress=400 compress_optimizer=sgd sgd_momentum=0.9 compress_lr=0.01 n_epochs_compress=120 ewc_lambda=500 fisher_clip=0.01 gamma=0.99 tau=2 normalize_fisher=false data_augmentation=true"

for seed in 0 1 2 3; do
    run "transfer_learned_s${seed}" $BEST seed=$seed
    run "transfer_random_s${seed}" $BEST kb_random=true seed=$seed
done

launch
