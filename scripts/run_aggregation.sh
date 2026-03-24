#!/bin/bash

# Post-crawl topic aggregation via iterative merge
# Usage: ./scripts/run_aggregation.sh [--tmux] [experiments=<name>] [overrides...]
#
# Config selection:
#   experiments=default      # aggregation parameters (default: default)
#   model=local_ds8b         # only needed if aggregation_model="local"
#
# Field overrides (dot notation):
#   experiments.batch_size=40
#   experiments.max_clusters=20
#   experiments.aggregation_model="anthropic/claude-3.5-haiku"
#
# Examples:
#   ./scripts/run_aggregation.sh
#   ./scripts/run_aggregation.sh experiments.batch_size=40
#   ./scripts/run_aggregation.sh --tmux model=local_ds8b experiments.aggregation_model=local

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

cd $PROJECT_ROOT

# Check for --tmux flag
NO_TMUX=true
if [ "$1" = "--tmux" ]; then
    NO_TMUX=false
    shift
fi

# Note: model= only needed if experiments.aggregation_model="local"
# input_paths defaults from experiments YAML; override with experiments.input_paths='[...]'

PYTHON_CMD="python src/aggregation/run_aggregation.py $@"

if [ "$NO_TMUX" = true ]; then
    echo "Running aggregation directly in terminal"
    echo ""
    exec $PYTHON_CMD
else
    mkdir -p "$PROJECT_ROOT/artifacts/log"

    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$PROJECT_ROOT/artifacts/log/aggregation_${TIMESTAMP}.log"
    SESSION_NAME="aggregation_${TIMESTAMP}"

    echo "Log File: $LOG_FILE"
    echo "Tmux Session: $SESSION_NAME"
    echo ""

    tmux new-session -d -s "$SESSION_NAME" \
        "export OPENAI_API_KEY='$OPENAI_API_KEY' && cd $PROJECT_ROOT && $PYTHON_CMD 2>&1 | tee '$LOG_FILE'"

    echo "Aggregation started in tmux session: $SESSION_NAME"
    echo "Attach to session with: tmux attach-session -t $SESSION_NAME"
    echo "View logs with: tail -f $LOG_FILE"
fi
