#!/usr/bin/env bash
# Compute post-hoc refusal rates over an aggregated cluster set on a Runpod
# checkout. Mirrors runpod_reviewer_ready.sh — env-var driven so the local
# controller can launch it inside one tmux session.
#
# Required env vars:
#   AGGREGATION_DIR   Pod-relative path to artifacts/aggregation/<ts>
# Optional env vars:
#   MODEL_CONFIG      Hydra model= override (default: local_ds70b)
#   OUT_DIR           Output dir (default: artifacts/refusal_rates/<UTC ts>)
#   PROBES_PER_TOPIC  -> crawler.num_refusal_checks_per_topic (default: 10)
#   THRESHOLD         -> crawler.is_refusal_threshold (default: 0.25)
#   SPECIFICITY_LEVEL Probe only topics at these levels (e.g. L5, or "L4,L5"
#                     / "L4 L5" for several) from specificity_scores.csv
#                     instead of final_topics.txt.
#   SPECIFICITY_CSV   Override the specificity csv path (default:
#                     $AGGREGATION_DIR/specificity_scores.csv).
#   EXTRA_OVERRIDES   Whitespace-separated Hydra overrides pass-through
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

MODEL_CONFIG="${MODEL_CONFIG:-local_ds70b}"
AGGREGATION_DIR="${AGGREGATION_DIR:?AGGREGATION_DIR is required (pod-relative path to artifacts/aggregation/<ts>)}"
OUT_DIR="${OUT_DIR:-artifacts/refusal_rates/$(date -u +%Y%m%d_%H%M%S)}"
PROBES_PER_TOPIC="${PROBES_PER_TOPIC:-10}"
THRESHOLD="${THRESHOLD:-0.25}"
SPECIFICITY_LEVEL="${SPECIFICITY_LEVEL:-}"
SPECIFICITY_CSV="${SPECIFICITY_CSV:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
LATEST_MARKER="${RUNPOD_LATEST_FILE:-artifacts/out/runpod_latest_refusal_rates.txt}"

mkdir -p "$OUT_DIR" "$(dirname "$LATEST_MARKER")"
printf '%s\n' "$OUT_DIR" > "$LATEST_MARKER"

RUN_LOG="$OUT_DIR/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export LLM_API_TIMEOUT_SECONDS="${LLM_API_TIMEOUT_SECONDS:-90}"
export LLM_API_MAX_RETRIES="${LLM_API_MAX_RETRIES:-1}"
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT"
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ -x "$PYTHON_BIN" ]; then
    PYTHON_CMD=("$PYTHON_BIN")
else
    PYTHON_CMD=(uv run python)
fi

ARGS=(
    "model=$MODEL_CONFIG"
    "+aggregation_dir=$AGGREGATION_DIR"
    "+out_dir=$OUT_DIR"
    "crawler.num_refusal_checks_per_topic=$PROBES_PER_TOPIC"
    "crawler.is_refusal_threshold=$THRESHOLD"
)

if [ -n "$SPECIFICITY_LEVEL" ]; then
    # Accept comma/space separated levels; pass as a Hydra list so commas are
    # not parsed as a (multirun) sweep. Single level -> [L5], multi -> [L4,L5].
    LEVELS_CSV=$(printf '%s' "$SPECIFICITY_LEVEL" | tr ' ' ',')
    ARGS+=("+specificity_level=[$LEVELS_CSV]")
fi

if [ -n "$SPECIFICITY_CSV" ]; then
    ARGS+=("+specificity_csv=$SPECIFICITY_CSV")
fi

if [ -n "$EXTRA_OVERRIDES" ]; then
    read -r -a EXTRA_OVERRIDE_ITEMS <<< "$EXTRA_OVERRIDES"
    ARGS+=("${EXTRA_OVERRIDE_ITEMS[@]}")
fi

echo "Runpod refusal-rate computation"
echo "Project root:     $PROJECT_ROOT"
echo "Model config:     $MODEL_CONFIG"
echo "Aggregation dir:  $AGGREGATION_DIR"
echo "Out dir:          $OUT_DIR"
echo "Probes per topic: $PROBES_PER_TOPIC"
echo "Threshold:        $THRESHOLD"
echo "Specificity level: ${SPECIFICITY_LEVEL:-none (all final_topics)}"
echo "Specificity csv:  ${SPECIFICITY_CSV:-default ($AGGREGATION_DIR/specificity_scores.csv)}"
echo "Extra overrides:  ${EXTRA_OVERRIDES:-none}"
echo "Latest marker:    $LATEST_MARKER"
echo "Timestamp:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "Error: OPENROUTER_API_KEY is not set on the pod."
    echo "Set it before starting the run because local_ds70b uses OpenRouter helper models."
    exit 1
fi

if [ -n "$SPECIFICITY_LEVEL" ]; then
    CSV_PATH="${SPECIFICITY_CSV:-$AGGREGATION_DIR/specificity_scores.csv}"
    if [ ! -f "$CSV_PATH" ]; then
        echo "Error: specificity csv not found for level $SPECIFICITY_LEVEL."
        echo "Checked: $CSV_PATH"
        exit 1
    fi
elif [ ! -f "$AGGREGATION_DIR/final_topics.txt" ] || [ ! -f "$AGGREGATION_DIR/reduction_log.json" ]; then
    echo "Error: AGGREGATION_DIR is missing final_topics.txt or reduction_log.json."
    echo "Checked: $AGGREGATION_DIR"
    exit 1
fi

echo
echo "Starting refusal-rate computation"
"${PYTHON_CMD[@]}" src/run_refusal_rates.py "${ARGS[@]}"

echo
echo "Refusal-rate computation complete."
echo "Out dir: $OUT_DIR"
echo "Table:   $OUT_DIR/refusal_rates.md"
