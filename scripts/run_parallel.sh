#!/bin/bash

# Run parallel crawler instances: either N identical runs (ensemble) or one run per model config.
#
# Ensemble mode (same config, multiple runs):
#   ./scripts/run_parallel.sh --num-runs N [--tmux] model=... crawler=... [hydra overrides]
#
# Multi-model mode (different models, one run each):
#   ./scripts/run_parallel.sh --models "a,b,c" [--tmux] crawler=... [hydra overrides]
#
# Kill running sessions:
#   ./scripts/run_parallel.sh --kill <timestamp_or_dir>
#
# Examples:
#   ./scripts/run_parallel.sh --num-runs 5 model=ds-v32_remote crawler=default
#   ./scripts/run_parallel.sh --num-runs 3 --tmux model=ds-v32_remote crawler=default crawler.num_crawl_steps=10
#   ./scripts/run_parallel.sh --models "ds-r1_remote,sonnet-45_remote,grok-41_remote" crawler=default
#   ./scripts/run_parallel.sh --models "ds-r1_remote,sonnet-45_remote" --tmux crawler=default
#   ./scripts/run_parallel.sh --kill 20260324_062619

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT
cd "$PROJECT_ROOT"

# --- Handle --kill ---
if [ "$1" = "--kill" ]; then
    PATTERN="$2"
    if [ -z "$PATTERN" ]; then
        echo "Usage: ./scripts/run_parallel.sh --kill <timestamp_or_dir>"
        exit 1
    fi
    TIMESTAMP_PAT=$(echo "$PATTERN" | grep -oP '\d{8}_\d{6}' | head -1)
    if [ -z "$TIMESTAMP_PAT" ]; then
        echo "Error: could not extract timestamp pattern from '$PATTERN'"
        exit 1
    fi
    SESSIONS=$(tmux ls 2>/dev/null | grep -E "(ensemble|multi)_${TIMESTAMP_PAT}" | cut -d: -f1)
    if [ -z "$SESSIONS" ]; then
        echo "No tmux sessions found matching *_${TIMESTAMP_PAT}"
        exit 0
    fi
    echo "Killing sessions:"
    for S in $SESSIONS; do
        tmux kill-session -t "$S" && echo "  killed $S" || echo "  failed to kill $S"
    done
    exit 0
fi

# --- Parse arguments ---
NUM_RUNS=""
MODELS=""
USE_TMUX=false
HYDRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-runs)
            NUM_RUNS="$2"; shift 2 ;;
        --models)
            MODELS="$2"; shift 2 ;;
        --tmux)
            USE_TMUX=true; shift ;;
        *)
            HYDRA_ARGS+=("$1"); shift ;;
    esac
done

# --- Validate mode ---
if [ -n "$NUM_RUNS" ] && [ -n "$MODELS" ]; then
    echo "Error: --num-runs and --models are mutually exclusive"
    exit 1
fi

if [ -z "$NUM_RUNS" ] && [ -z "$MODELS" ]; then
    echo "Error: specify either --num-runs N or --models \"a,b,c\""
    echo ""
    echo "Usage: ./scripts/run_parallel.sh --num-runs N [--tmux] model=<name> crawler=<name> [overrides]"
    echo "       ./scripts/run_parallel.sh --models \"a,b,c\" [--tmux] crawler=<name> [overrides]"
    echo "       ./scripts/run_parallel.sh --kill <timestamp_or_dir>"
    exit 1
fi

# Validate crawler= present (required for both modes)
if ! printf '%s\n' "${HYDRA_ARGS[@]}" | grep -q "^crawler="; then
    echo "Error: missing required argument 'crawler=<name>'"
    echo "Available: $(ls configs/crawler/*.yaml | xargs -n1 basename | sed 's/\.yaml//' | tr '\n' ' ')"
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# --- Ensemble mode ---
if [ -n "$NUM_RUNS" ]; then
    if [ "$NUM_RUNS" -lt 1 ] 2>/dev/null; then
        echo "Error: --num-runs N required (N >= 1)"
        exit 1
    fi

    if ! printf '%s\n' "${HYDRA_ARGS[@]}" | grep -q "^model="; then
        echo "Error: missing required argument 'model=<name>' for ensemble mode"
        echo "Available: $(ls configs/model/*.yaml | xargs -n1 basename | sed 's/\.yaml//' | tr '\n' ' ')"
        exit 1
    fi

    MODEL_NAME=$(printf '%s\n' "${HYDRA_ARGS[@]}" | grep -oP '^model=\K\S+')
    OUTDIR="$PROJECT_ROOT/artifacts/out/ensemble_${TIMESTAMP}_${MODEL_NAME}"
    mkdir -p "$OUTDIR"

    cat > "$OUTDIR/ensemble_meta.json" <<EOF
{
    "num_runs": $NUM_RUNS,
    "timestamp": "$TIMESTAMP",
    "model": "$MODEL_NAME",
    "hydra_args": "$(printf '%s ' "${HYDRA_ARGS[@]}")"
}
EOF

    echo "Ensemble dir: $OUTDIR"
    echo "Launching $NUM_RUNS parallel runs..."
    echo ""

    # Collect all *_API_KEY env vars to forward into tmux sessions
    ENV_EXPORTS=""
    while IFS='=' read -r key val; do
        ENV_EXPORTS+="export ${key}='${val}' && "
    done < <(env | grep '_API_KEY=' | sort)

    PIDS=()
    for i in $(seq 1 "$NUM_RUNS"); do
        RUN_TAG="run$(printf '%02d' "$i")"
        CMD="python src/crawler/run_crawler.py ${HYDRA_ARGS[*]} crawler.output_dir=$OUTDIR crawler.run_tag=$RUN_TAG"

        if [ "$USE_TMUX" = true ]; then
            SESSION="ensemble_${TIMESTAMP}_${RUN_TAG}"
            LOG_FILE="$OUTDIR/${RUN_TAG}.log"
            tmux new-session -d -s "$SESSION" \
                "${ENV_EXPORTS}cd $PROJECT_ROOT && export PYTHONPATH=$PYTHONPATH && $CMD 2>&1 | tee '$LOG_FILE'"
            echo "  [$RUN_TAG] tmux session: $SESSION, log: $LOG_FILE"
        else
            LOG_FILE="$OUTDIR/${RUN_TAG}.log"
            $CMD > "$LOG_FILE" 2>&1 &
            PIDS+=($!)
            echo "  [$RUN_TAG] PID=${PIDS[-1]}, log: $LOG_FILE"
        fi
    done

# --- Multi-model mode ---
else
    if printf '%s\n' "${HYDRA_ARGS[@]}" | grep -q "^model="; then
        echo "Error: do not pass model= in overrides when using --models (each model is set per-run)"
        exit 1
    fi

    IFS=',' read -ra MODEL_LIST <<< "$MODELS"
    if [ "${#MODEL_LIST[@]}" -lt 1 ]; then
        echo "Error: --models requires at least one model config name"
        exit 1
    fi

    OUTDIR="$PROJECT_ROOT/artifacts/out/multi_${TIMESTAMP}"
    mkdir -p "$OUTDIR"

    MODELS_JSON=$(printf '%s\n' "${MODEL_LIST[@]}" | sed 's/.*/"&"/' | paste -sd, | sed 's/^/[/;s/$/]/')
    cat > "$OUTDIR/multi_meta.json" <<EOF
{
    "num_models": ${#MODEL_LIST[@]},
    "timestamp": "$TIMESTAMP",
    "models": $MODELS_JSON,
    "hydra_args": "$(printf '%s ' "${HYDRA_ARGS[@]}")"
}
EOF

    echo "Multi-model dir: $OUTDIR"
    echo "Launching ${#MODEL_LIST[@]} parallel runs: ${MODEL_LIST[*]}"
    echo ""

    PIDS=()
    RUN_TAGS=()
    for MODEL in "${MODEL_LIST[@]}"; do
        RUN_TAG="$MODEL"
        RUN_TAGS+=("$RUN_TAG")
        CMD="python src/crawler/run_crawler.py model=$MODEL ${HYDRA_ARGS[*]} crawler.output_dir=$OUTDIR crawler.run_tag=$RUN_TAG"

        if [ "$USE_TMUX" = true ]; then
            SESSION="multi_${TIMESTAMP}_${MODEL}"
            LOG_FILE="$OUTDIR/${MODEL}.log"
            tmux new-session -d -s "$SESSION" \
                "${ENV_EXPORTS}cd $PROJECT_ROOT && export PYTHONPATH=$PYTHONPATH && $CMD 2>&1 | tee '$LOG_FILE'"
            echo "  [$MODEL] tmux session: $SESSION, log: $LOG_FILE"
        else
            LOG_FILE="$OUTDIR/${MODEL}.log"
            $CMD > "$LOG_FILE" 2>&1 &
            PIDS+=($!)
            echo "  [$MODEL] PID=${PIDS[-1]}, log: $LOG_FILE"
        fi
    done
fi

# --- Wait and report (non-tmux mode) ---
if [ "$USE_TMUX" = false ]; then
    echo ""
    NUM_TOTAL=${#PIDS[@]}
    echo "Waiting for all $NUM_TOTAL runs to complete..."
    FAILURES=0
    for i in "${!PIDS[@]}"; do
        PID=${PIDS[$i]}
        if [ -n "$NUM_RUNS" ]; then
            TAG="run$(printf '%02d' $((i+1)))"
        else
            TAG="${RUN_TAGS[$i]}"
        fi
        if wait "$PID"; then
            echo "  [$TAG] PID=$PID completed successfully"
        else
            EXIT_CODE=$?
            echo "  [$TAG] PID=$PID FAILED (exit code $EXIT_CODE)"
            FAILURES=$((FAILURES + 1))
        fi
    done

    echo ""
    echo "Complete: $((NUM_TOTAL - FAILURES))/$NUM_TOTAL succeeded"
    echo "Results in: $OUTDIR"
    [ "$FAILURES" -gt 0 ] && exit 1
else
    echo ""
    if [ -n "$NUM_RUNS" ]; then
        echo "All runs launched in tmux. List sessions: tmux ls | grep ensemble_${TIMESTAMP}"
    else
        echo "All runs launched in tmux. List sessions: tmux ls | grep multi_${TIMESTAMP}"
    fi
    echo "Results will appear in: $OUTDIR"
fi
