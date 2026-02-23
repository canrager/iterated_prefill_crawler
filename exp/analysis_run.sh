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
LOG_FILE="$PROJECT_ROOT/artifacts/log/analysis_${TIMESTAMP}.log"

echo "Log Dir: $LOG_FILE"

# Run the crawler script with nohup and write to the log file
nohup python exp/evaluate_crawler.py \
    > "$LOG_FILE" 2>&1 &

# Store the process ID
PID=$!
echo "PID: $PID"

# Add any additional arguments as needed
# Example: --load_fname "path/to/saved/state" if you want to resume from a saved state