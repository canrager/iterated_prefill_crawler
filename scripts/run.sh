#!/bin/bash

# Unified script to run crawler with Hydra configs in tmux
# Usage: ./scripts/run.sh [--tmux] [hydra_overrides]
#
# Config selection (corresponds to configs/model/*.yaml and configs/crawler/*.yaml):
#   model=haiku              # use configs/model/haiku.yaml (default: local_ds8b)
#   model=local_tulu8b       # use configs/model/local_tulu8b.yaml
#   model=local_meta8b       # use configs/model/local_meta8b.yaml
#   crawler=debug            # use configs/crawler/debug.yaml (default: default)
#
# Field overrides (dot notation into the nested config):
#   model.temperature=0.9
#   crawler.num_crawl_steps=5
#
# Examples:
#   ./scripts/run.sh model=haiku crawler=default
#   ./scripts/run.sh model=haiku crawler=debug
#   ./scripts/run.sh --tmux model=local_tulu8b crawler=default crawler.num_crawl_steps=10

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

# Require model= and crawler= arguments
if ! echo "$@" | grep -q "model="; then
    echo "Error: missing required argument 'model=<name>'"
    echo "Available: $(ls configs/model/*.yaml | xargs -n1 basename | sed 's/\.yaml//' | tr '\n' ' ')"
    exit 1
fi
if ! echo "$@" | grep -q "crawler="; then
    echo "Error: missing required argument 'crawler=<name>'"
    echo "Available: $(ls configs/crawler/*.yaml | xargs -n1 basename | sed 's/\.yaml//' | tr '\n' ' ')"
    exit 1
fi

# Build the full python command with all arguments
PYTHON_CMD="python src/crawler/run_crawler.py $@"

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
    # Export OPENAI_API_KEY to the tmux environment
    tmux new-session -d -s "$SESSION_NAME" \
        "export OPENAI_API_KEY='$OPENAI_API_KEY' && cd $PROJECT_ROOT && $PYTHON_CMD 2>&1 | tee '$LOG_FILE'"

    echo "Crawler started in tmux session: $SESSION_NAME"
    echo "Attach to session with: tmux attach-session -t $SESSION_NAME"
    echo "View logs with: tail -f $LOG_FILE"
fi
