#!/bin/bash

# Post-crawl topic aggregation via iterative merge
# Usage: ./scripts/run_aggregation.sh [--tmux] [aggregation=<name>] [overrides...]
#
# Config selection:
#   aggregation=default      # aggregation parameters (default: default)
#   model=local_ds8b         # only needed if aggregation_model="local"
#
# Field overrides (dot notation):
#   aggregation.batch_size=40
#   aggregation.max_clusters=20
#   aggregation.aggregation_model="anthropic/claude-3.5-haiku"
#
# Flags:
#   --tmux         Run in a tmux session with logging
#   --all          Run aggregation for every config in configs/aggregation/
#
# Examples:
#   ./scripts/run_aggregation.sh
#   ./scripts/run_aggregation.sh aggregation=olmo3_default
#   ./scripts/run_aggregation.sh --all
#   ./scripts/run_aggregation.sh --tmux model=local_ds8b aggregation.aggregation_model=local

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

cd $PROJECT_ROOT

# Parse flags
NO_TMUX=true
RUN_ALL=false
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --tmux) NO_TMUX=false ;;
        --all)  RUN_ALL=true ;;
        *)      ARGS+=("$arg") ;;
    esac
done

run_single() {
    # Extract experiment name from args for naming
    local EXP_NAME="default"
    for a in "$@"; do
        case "$a" in aggregation=*) EXP_NAME="${a#aggregation=}" ;; esac
    done

    local PYTHON_CMD="python src/aggregation/run_aggregation.py $@"

    if [ "$NO_TMUX" = true ]; then
        echo "Running aggregation directly in terminal"
        echo ""
        $PYTHON_CMD
    else
        mkdir -p "$PROJECT_ROOT/artifacts/log"

        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        LOG_FILE="$PROJECT_ROOT/artifacts/log/aggregation_${EXP_NAME}_${TIMESTAMP}.log"
        SESSION_NAME="aggregation_${EXP_NAME}_${TIMESTAMP}"

        echo "Log File: $LOG_FILE"
        echo "Tmux Session: $SESSION_NAME"
        echo ""

        # Collect all *_API_KEY env vars to forward into the tmux session
        ENV_EXPORTS=""
        while IFS='=' read -r key val; do
            ENV_EXPORTS+="export ${key}='${val}' && "
        done < <(env | grep '_API_KEY=' | sort)

        tmux new-session -d -s "$SESSION_NAME" \
            "${ENV_EXPORTS}cd $PROJECT_ROOT && $PYTHON_CMD 2>&1 | tee '$LOG_FILE'"

        echo "Aggregation started in tmux session: $SESSION_NAME"
        echo "Attach to session with: tmux attach-session -t $SESSION_NAME"
        echo "View logs with: tail -f $LOG_FILE"
    fi
}

if [ "$RUN_ALL" = true ]; then
    for yaml in "$PROJECT_ROOT/configs/aggregation/"*.yaml; do
        name="$(basename "$yaml" .yaml)"
        echo "========================================"
        echo "Running aggregation=$name"
        echo "========================================"
        run_single "aggregation=$name" "${ARGS[@]}"
        echo ""
    done
else
    run_single "${ARGS[@]}"
fi
