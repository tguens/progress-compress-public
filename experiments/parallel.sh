# Parallel sweep runner. Source this file from a sweep script.
#
# Usage inside a sweep script:
#     source experiments/parallel.sh
#     run "tag_a" ewc_lambda=100
#     run "tag_b" ewc_lambda=0
#     launch
#
# `run` queues one configuration. It skips the tag if results.json exists.
# `launch` starts the queued runs in parallel, one Hydra run directory and
# one log file per run. The job limit is the cgroup CPU quota minus one,
# because one run takes one core.
# Set SWEEP_DIR and COMMON before you source this file.

SWEEP_DIR="${SWEEP_DIR:-outputs/experiments}"
COMMON="${COMMON:-}"
MAIN="${MAIN:-src/run_omniglot.py}"  # src/run_atari.py for Task 2
JOBS_FILE="$(mktemp)"

job_limit() {
    # CPU quota minus one, at least 1. cgroup quota on a pod, else the core count.
    local cores
    if [ -r /sys/fs/cgroup/cpu.max ]; then
        read -r quota period < /sys/fs/cgroup/cpu.max
        if [ "$quota" != "max" ]; then
            cores=$(( quota / period ))
        fi
    fi
    if [ -z "$cores" ]; then
        cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || python3 -c "import os; print(os.cpu_count() or 1)")
    fi
    local n=$(( cores - 1 ))
    [ "$n" -lt 1 ] && n=1
    echo "$n"
}

run() {
    local tag="$1"; shift
    if [ -f "$SWEEP_DIR/$tag/results.json" ]; then
        echo "SKIP $tag"
        return
    fi
    printf '%s\n' "$tag $*" >> "$JOBS_FILE"
}

run_one() {
    local tag="$1"; shift
    mkdir -p "$SWEEP_DIR/$tag"
    echo ">>> $tag $(date +%H:%M:%S)"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run "$MAIN" $COMMON "$@" \
        "hydra.run.dir=$SWEEP_DIR/$tag" > "$SWEEP_DIR/$tag/run.log" 2>&1
    local status=$?
    echo "<<< $tag exit $status $(date +%H:%M:%S)"
    return $status
}
export -f run_one
export SWEEP_DIR COMMON MAIN

launch() {
    n_jobs="${JOBS:-$(job_limit)}"  # JOBS=<n> overrides the job limit
    echo "Launching $(wc -l < "$JOBS_FILE" | tr -d " ") runs, $n_jobs in parallel"
    xargs -P "$n_jobs" -L 1 bash -c 'run_one "$@"' _ < "$JOBS_FILE"
    local status=$?
    rm -f "$JOBS_FILE"
    if [ "$status" -ne 0 ]; then
        echo "Sweep done with failed runs (see the exit lines above) $(date +%H:%M:%S)"
        return 1
    fi
    echo "Sweep done $(date +%H:%M:%S)"
}
