#!/bin/bash

# Run N parallel crawler instances with identical config, bundled into one folder.
# Usage: ./scripts/run_ensemble.sh --num-runs N [--tmux] model=... crawler=... [hydra overrides]
#
# Examples:
#   ./scripts/run_ensemble.sh --num-runs 5 model=ds-v32_remote crawler=default
#   ./scripts/run_ensemble.sh --num-runs 3 --tmux model=ds-v32_remote crawler=default crawler.num_crawl_steps=10

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT
cd "$PROJECT_ROOT"

# --- Parse arguments ---
NUM_RUNS=""
USE_TMUX=false
HYDRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-runs)
            NUM_RUNS="$2"; shift 2 ;;
        --tmux)
            USE_TMUX=true; shift ;;
        *)
            HYDRA_ARGS+=("$1"); shift ;;
    esac
done

if [ -z "$NUM_RUNS" ] || [ "$NUM_RUNS" -lt 1 ] 2>/dev/null; then
    echo "Error: --num-runs N required (N >= 1)"
    echo ""
    echo "Usage: ./scripts/run_ensemble.sh --num-runs N [--tmux] model=<name> crawler=<name> [overrides]"
    exit 1
fi

# Validate model= and crawler= present
if ! printf '%s\n' "${HYDRA_ARGS[@]}" | grep -q "^model="; then
    echo "Error: missing required argument 'model=<name>'"
    echo "Available: $(ls configs/model/*.yaml | xargs -n1 basename | sed 's/\.yaml//' | tr '\n' ' ')"
    exit 1
fi
if ! printf '%s\n' "${HYDRA_ARGS[@]}" | grep -q "^crawler="; then
    echo "Error: missing required argument 'crawler=<name>'"
    echo "Available: $(ls configs/crawler/*.yaml | xargs -n1 basename | sed 's/\.yaml//' | tr '\n' ' ')"
    exit 1
fi

# Extract model name for folder naming
MODEL_NAME=$(printf '%s\n' "${HYDRA_ARGS[@]}" | grep -oP '^model=\K\S+')

# --- Create ensemble directory ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ENSEMBLE_DIR="$PROJECT_ROOT/artifacts/out/ensemble_${TIMESTAMP}_${MODEL_NAME}"
mkdir -p "$ENSEMBLE_DIR"

# Save ensemble metadata for reproducibility
cat > "$ENSEMBLE_DIR/ensemble_meta.json" <<EOF
{
    "num_runs": $NUM_RUNS,
    "timestamp": "$TIMESTAMP",
    "model": "$MODEL_NAME",
    "hydra_args": "$(printf '%s ' "${HYDRA_ARGS[@]}")"
}
EOF

echo "Ensemble dir: $ENSEMBLE_DIR"
echo "Launching $NUM_RUNS parallel runs..."
echo ""

# --- Launch N processes ---
PIDS=()
for i in $(seq 1 "$NUM_RUNS"); do
    RUN_TAG="run$(printf '%02d' "$i")"
    CMD="python src/crawler/run_crawler.py ${HYDRA_ARGS[*]} crawler.output_dir=$ENSEMBLE_DIR crawler.run_tag=$RUN_TAG"

    if [ "$USE_TMUX" = true ]; then
        SESSION="ensemble_${TIMESTAMP}_${RUN_TAG}"
        LOG_FILE="$ENSEMBLE_DIR/${RUN_TAG}.log"
        tmux new-session -d -s "$SESSION" \
            "export OPENROUTER_API_KEY='$OPENROUTER_API_KEY' && cd $PROJECT_ROOT && export PYTHONPATH=$PYTHONPATH && $CMD 2>&1 | tee '$LOG_FILE'"
        echo "  [$RUN_TAG] tmux session: $SESSION, log: $LOG_FILE"
    else
        LOG_FILE="$ENSEMBLE_DIR/${RUN_TAG}.log"
        $CMD > "$LOG_FILE" 2>&1 &
        PIDS+=($!)
        echo "  [$RUN_TAG] PID=${PIDS[-1]}, log: $LOG_FILE"
    fi
done

# --- Wait and report (non-tmux mode) ---
if [ "$USE_TMUX" = false ]; then
    echo ""
    echo "Waiting for all $NUM_RUNS runs to complete..."
    FAILURES=0
    for i in "${!PIDS[@]}"; do
        PID=${PIDS[$i]}
        RUN_TAG="run$(printf '%02d' $((i+1)))"
        if wait "$PID"; then
            echo "  [$RUN_TAG] PID=$PID completed successfully"
        else
            EXIT_CODE=$?
            echo "  [$RUN_TAG] PID=$PID FAILED (exit code $EXIT_CODE)"
            FAILURES=$((FAILURES + 1))
        fi
    done

    echo ""
    echo "Ensemble complete: $((NUM_RUNS - FAILURES))/$NUM_RUNS succeeded"
    echo "Results in: $ENSEMBLE_DIR"
    [ "$FAILURES" -gt 0 ] && exit 1
else
    echo ""
    echo "All runs launched in tmux. List sessions: tmux ls | grep ensemble_${TIMESTAMP}"
    echo "Results will appear in: $ENSEMBLE_DIR"
fi
