#!/usr/bin/env bash
# Run the reviewer 2x2 ablation sequentially on a remote GPU checkout.
#
# Configuration is via environment variables so the local controller can launch
# this script inside one tmux session without rewriting command strings:
#   MODEL_CONFIG=local_ds70b
#   CRAWLER_CONFIG=default
#   OUT_DIR=artifacts/out/reviewer_ablation_runpod_<timestamp>
#   SAMPLES=2
#   VALIDATE_ALL_DISCOVERED=1
#   EXTRA_OVERRIDES='model.vllm_tensor_parallel_size=1'
#   LLM_API_TIMEOUT_SECONDS=90
#   LLM_API_MAX_RETRIES=1

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
CRAWLER_CONFIG="${CRAWLER_CONFIG:-default}"
OUT_DIR="${OUT_DIR:-artifacts/out/reviewer_ablation_runpod_$(date -u +%Y%m%d_%H%M%S)}"
SAMPLES="${SAMPLES:-}"
VALIDATE_ALL_DISCOVERED="${VALIDATE_ALL_DISCOVERED:-0}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
LATEST_MARKER="${RUNPOD_LATEST_FILE:-artifacts/out/runpod_latest_reviewer_ablation.txt}"

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
    --models "$MODEL_CONFIG"
    --crawler "$CRAWLER_CONFIG"
    --out-dir "$OUT_DIR"
)

if [ -n "$SAMPLES" ]; then
    ARGS+=(--samples "$SAMPLES")
fi

if [ -n "$EXTRA_OVERRIDES" ]; then
    read -r -a EXTRA_OVERRIDE_ITEMS <<< "$EXTRA_OVERRIDES"
    for override in "${EXTRA_OVERRIDE_ITEMS[@]}"; do
        ARGS+=(--override "$override")
    done
fi

case "$(printf '%s' "$VALIDATE_ALL_DISCOVERED" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes)
        ARGS+=(--validate-all-discovered)
        ;;
esac

echo "Runpod reviewer-ready 2x2 run"
echo "Project root: $PROJECT_ROOT"
echo "Model config: $MODEL_CONFIG"
echo "Crawler config: $CRAWLER_CONFIG"
echo "Out dir: $OUT_DIR"
echo "Extra overrides: ${EXTRA_OVERRIDES:-none}"
echo "Helper API timeout seconds: $LLM_API_TIMEOUT_SECONDS"
echo "Helper API max retries: $LLM_API_MAX_RETRIES"
echo "Latest marker: $LATEST_MARKER"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "Error: OPENROUTER_API_KEY is not set on the pod."
    echo "Set it before starting the run because local_ds70b uses OpenRouter helper models."
    exit 1
fi

echo
echo "Writing command plan to $OUT_DIR/plan.md"
"${PYTHON_CMD[@]}" scripts/reviewer_ablation.py plan "${ARGS[@]}" | tee "$OUT_DIR/plan.md"

echo
echo "Starting sequential 2x2 run"
"${PYTHON_CMD[@]}" scripts/reviewer_ablation.py run "${ARGS[@]}"

echo
echo "Writing reviewer summary to $OUT_DIR/summary.md"
"${PYTHON_CMD[@]}" scripts/reviewer_ablation.py summarize --out-dir "$OUT_DIR" | tee "$OUT_DIR/summary.md"

echo
echo "Reviewer-ready run complete."
echo "Out dir: $OUT_DIR"
echo "Summary: $OUT_DIR/summary.md"
