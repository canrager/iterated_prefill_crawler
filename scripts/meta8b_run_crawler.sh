#!/bin/bash

# Script to run the thought crawler in debug mode
# Created for running exp/run_crawler.py with debug configuration

# Get the directory of this script and the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
# Set the Python path to include the project root
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT
# Change to the project root directory
cd $PROJECT_ROOT

# Create artifacts/log directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/artifacts/log"
# Generate timestamp for log filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_ROOT/artifacts/log/crawler_debug_${TIMESTAMP}.log"
SESSION_NAME="crawler_meta8b_${TIMESTAMP}"

echo "Log Dir: $LOG_FILE"
echo "Tmux Session: $SESSION_NAME"

# Run the crawler script in a tmux session
tmux new-session -d -s "$SESSION_NAME" \
    "cd $PROJECT_ROOT && python exp/run_crawler.py \
    --device 'cuda:0' \
    --cache_dir '/home/can/models/' \
    --model_path 'meta-llama/Llama-3.1-8B-Instruct' \
    --quantization_bits 'none' \
    --prefill_mode 'thought_prefix' \
    $@ \
    2>&1 | tee '$LOG_FILE'"

echo "Attach to session with: tmux attach-session -t $SESSION_NAME"

# Add any additional arguments as needed
# Example: --load_fname "path/to/saved/state" if you want to resume from a saved state