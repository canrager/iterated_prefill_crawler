#!/usr/bin/env bash
# Pod-side aggregation driver for the reviewer-ready 2x2 follow-on.
# Mirrors scripts/runpod_reviewer_ready.sh — env-var driven, tees run.log,
# writes the latest marker for runpod_control.py status/tail/fetch.
#
# Either pass INPUT_PATHS (verbatim Hydra JSON list of pod-relative paths)
# OR pass REVIEWER_OUT_DIR (a pod-relative 2x2 output dir) and let the
# driver auto-discover one crawler_out_*_<cell>.json per cell.
#
# Required env vars (one of):
#   REVIEWER_OUT_DIR   pod-relative path to a 2x2 output dir
#   INPUT_PATHS        Hydra JSON list, e.g. '["a.json","b.json","c.json","d.json"]'
# Optional env vars:
#   AGG_MODEL_CONFIG   Hydra model= override (default: gemini-31fl_remote)
#   AGG_LLM            experiments.aggregation_model (default: moonshotai/kimi-k2-0905)
#   MAX_FINAL_TOPICS   experiments.max_final_topics (default: 80)
#   INPUT_BATCH_SIZE   experiments.input_batch_size (default: 50)
#   OUTPUT_BATCH_SIZE  experiments.output_batch_size (default: 25)
#   EXTRA_OVERRIDES    Whitespace-separated Hydra overrides pass-through
#   RUNPOD_LATEST_FILE override for the latest marker path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${RUNPOD_REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

cd "$PROJECT_ROOT"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

AGG_MODEL_CONFIG="${AGG_MODEL_CONFIG:-gemini-31fl_remote}"
AGG_LLM="${AGG_LLM:-moonshotai/kimi-k2-0905}"
MAX_FINAL_TOPICS="${MAX_FINAL_TOPICS:-80}"
INPUT_BATCH_SIZE="${INPUT_BATCH_SIZE:-50}"
OUTPUT_BATCH_SIZE="${OUTPUT_BATCH_SIZE:-25}"
REVIEWER_OUT_DIR="${REVIEWER_OUT_DIR:-}"
INPUT_PATHS="${INPUT_PATHS:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
LATEST_MARKER="${RUNPOD_LATEST_FILE:-artifacts/out/runpod_latest_aggregation.txt}"

OUT_DIR="${OUT_DIR:-artifacts/aggregation/$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR" "$(dirname "$LATEST_MARKER")"
printf '%s\n' "$OUT_DIR" > "$LATEST_MARKER"

RUN_LOG="$OUT_DIR/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT"
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ -x "$PYTHON_BIN" ]; then
    PYTHON_CMD=("$PYTHON_BIN")
else
    PYTHON_CMD=(uv run python)
fi

# Build the Hydra input_paths argument.
if [ -z "$INPUT_PATHS" ]; then
    if [ -z "$REVIEWER_OUT_DIR" ]; then
        echo "Error: pass either INPUT_PATHS or REVIEWER_OUT_DIR." >&2
        exit 2
    fi
    cells=(direct prefill_only iter_no_prefill ipc)
    paths=()
    for cell in "${cells[@]}"; do
        match=$(ls "$REVIEWER_OUT_DIR"/crawler_out_*_${cell}.json 2>/dev/null | head -1 || true)
        if [ -z "$match" ]; then
            echo "Error: no crawler_out_*_${cell}.json under $REVIEWER_OUT_DIR" >&2
            exit 2
        fi
        paths+=("$match")
    done
    # Hydra expects the form '["a","b","c","d"]' with double quotes
    INPUT_PATHS=$(printf '"%s",' "${paths[@]}")
    INPUT_PATHS="[${INPUT_PATHS%,}]"
fi

ARGS=(
    "model=$AGG_MODEL_CONFIG"
    "experiments.aggregation_model=$AGG_LLM"
    "experiments.input_paths=$INPUT_PATHS"
    "experiments.max_final_topics=$MAX_FINAL_TOPICS"
    "experiments.input_batch_size=$INPUT_BATCH_SIZE"
    "experiments.output_batch_size=$OUTPUT_BATCH_SIZE"
    "hydra.run.dir=$OUT_DIR"
)

if [ -n "$EXTRA_OVERRIDES" ]; then
    read -r -a EXTRA_OVERRIDE_ITEMS <<< "$EXTRA_OVERRIDES"
    ARGS+=("${EXTRA_OVERRIDE_ITEMS[@]}")
fi

echo "Runpod aggregation"
echo "Project root:       $PROJECT_ROOT"
echo "Aggregation model:  $AGG_MODEL_CONFIG / $AGG_LLM"
echo "Input paths:        $INPUT_PATHS"
echo "Max final topics:   $MAX_FINAL_TOPICS"
echo "Input batch size:   $INPUT_BATCH_SIZE"
echo "Output batch size:  $OUTPUT_BATCH_SIZE"
echo "Out dir:            $OUT_DIR"
echo "Extra overrides:    ${EXTRA_OVERRIDES:-none}"
echo "Latest marker:      $LATEST_MARKER"
echo "Timestamp:          $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "Error: OPENROUTER_API_KEY is not set on the pod."
    echo "Set it before starting the run because the aggregation model is remote."
    exit 1
fi

echo
echo "Starting aggregation"
"${PYTHON_CMD[@]}" src/aggregation/run_aggregation.py "${ARGS[@]}"

# run_aggregation.py writes under artifacts/aggregation/<ts>/ by default,
# but hydra.run.dir is unused there — it always builds its own timestamped
# subdir. Resolve the actual output dir and refresh the marker.
ACTUAL_DIR=$(ls -td "$PROJECT_ROOT/artifacts/aggregation"/*/ 2>/dev/null | head -1 | sed "s|^$PROJECT_ROOT/||; s|/$||")
if [ -n "$ACTUAL_DIR" ]; then
    printf '%s\n' "$ACTUAL_DIR" > "$LATEST_MARKER"
    OUT_DIR="$ACTUAL_DIR"
fi

echo
echo "Aggregation complete."
echo "Out dir: $OUT_DIR"
