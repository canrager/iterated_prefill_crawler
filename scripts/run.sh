#!/bin/bash

# Unified script to run crawler with Hydra configs in tmux
# Usage: ./scripts/run.sh [--no-tmux] [hydra_args]
# Example: ./scripts/run.sh model_path="allenai/Llama-3.1-Tulu-3-8B-SFT" prompt_injection_location=assistant_prefix
# Example: ./scripts/run.sh --config-name=debug
# Example: ./scripts/run.sh --no-tmux --config-name=debug  # Run directly in terminal without logging

# Get the directory of this script and the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Set the Python path to include the project root
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

# Change to the project root directory
cd $PROJECT_ROOT

# Check for --tmux flag
NO_TMUX=true
if [ "$1" = "--tmux" ]; then
    NO_TMUX=false
    shift
fi

# Build the full python command with all arguments
PYTHON_CMD="python exp/run_crawler.py $@"

if [ "$NO_TMUX" = true ]; then
    # Run directly in terminal without logging
    echo "Running directly in terminal (no tmux, no logging)"
    echo ""
    exec $PYTHON_CMD
else
    # Create artifacts/log directory if it doesn't exist
    mkdir -p "$PROJECT_ROOT/artifacts/log"

    # Generate timestamp for log filename and session name
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$PROJECT_ROOT/artifacts/log/crawler_${TIMESTAMP}.log"
    SESSION_NAME="crawler_${TIMESTAMP}"

    echo "Log File: $LOG_FILE"
    echo "Tmux Session: $SESSION_NAME"
    echo ""

    # Run the crawler script in a tmux session
    tmux new-session -d -s "$SESSION_NAME" \
        "cd $PROJECT_ROOT && $PYTHON_CMD 2>&1 | tee '$LOG_FILE'"

    echo "Crawler started in tmux session: $SESSION_NAME"
    echo "Attach to session with: tmux attach-session -t $SESSION_NAME"
    echo "View logs with: tail -f $LOG_FILE"
fi
