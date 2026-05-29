#!/bin/bash

# Post-hoc refusal-rate computation over aggregated cluster heads.
# Usage:
#   ./scripts/run_refusal_rates.sh \
#     model=local_ds70b \
#     +aggregation_dir=artifacts/aggregation/<ts> \
#     [+out_dir=artifacts/refusal_rates/<ts>] \
#     [crawler.num_refusal_checks_per_topic=10] \
#     [crawler.is_refusal_threshold=0.25]
#
# Flags:
#   --tmux  Run in a tmux session with tee'd logging.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

cd "$PROJECT_ROOT"

NO_TMUX=true
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --tmux) NO_TMUX=false ;;
        *)      ARGS+=("$arg") ;;
    esac
done

PYTHON_CMD=(python src/run_refusal_rates.py "${ARGS[@]}")

if [ "$NO_TMUX" = true ]; then
    "${PYTHON_CMD[@]}"
else
    mkdir -p "$PROJECT_ROOT/artifacts/log"
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$PROJECT_ROOT/artifacts/log/refusal_rates_${TIMESTAMP}.log"
    SESSION_NAME="refusal_rates_${TIMESTAMP}"

    echo "Log File: $LOG_FILE"
    echo "Tmux Session: $SESSION_NAME"
    echo

    ENV_EXPORTS=""
    while IFS='=' read -r key val; do
        ENV_EXPORTS+="export ${key}='${val}' && "
    done < <(env | grep '_API_KEY=' | sort)

    tmux new-session -d -s "$SESSION_NAME" \
        "${ENV_EXPORTS}cd $PROJECT_ROOT && ${PYTHON_CMD[*]} 2>&1 | tee '$LOG_FILE'"

    echo "Refusal-rate computation started in tmux: $SESSION_NAME"
    echo "Attach with: tmux attach-session -t $SESSION_NAME"
    echo "Tail with:   tail -f $LOG_FILE"
fi
