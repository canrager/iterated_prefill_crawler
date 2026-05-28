#!/usr/bin/env bash
# Prepare a synced Runpod checkout for reviewer-ablation runs.
#
# This script is intended to run on the pod from the repository root. The local
# controller calls it over SSH after rsyncing the checkout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${RUNPOD_REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

cd "$PROJECT_ROOT"
mkdir -p artifacts/log artifacts/out

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

BOOTSTRAP_LOG="artifacts/log/runpod_bootstrap_$(date -u +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$BOOTSTRAP_LOG") 2>&1

echo "Runpod bootstrap"
echo "Project root: $PROJECT_ROOT"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "GPU visibility:"
    nvidia-smi
else
    echo
    echo "Warning: nvidia-smi is not available. Check the Runpod image/GPU mount."
fi

if ! command -v uv >/dev/null 2>&1; then
    echo
    echo "uv is not installed; installing with the official installer."
    if ! command -v curl >/dev/null 2>&1; then
        echo "Error: curl is required to install uv automatically."
        exit 1
    fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is still unavailable after installation."
    exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo
    echo "Warning: tmux is not installed. The local controller needs tmux for start/status."
fi

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

echo
echo "Python and dependency setup:"
uv --version
uv sync
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"

if ! command -v ninja >/dev/null 2>&1; then
    echo
    echo "ninja is not installed; installing the Python-packaged ninja CLI for vLLM/FlashInfer JIT builds."
    uv pip install ninja
fi

if ! command -v ninja >/dev/null 2>&1; then
    echo "Error: ninja is still unavailable after installation."
    exit 1
fi

echo "ninja version: $(ninja --version)"

echo
echo "Environment reminders:"
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "- OPENROUTER_API_KEY is not set; helper model calls will fail until it is set."
else
    echo "- OPENROUTER_API_KEY is set."
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "- HF_TOKEN is not set; this is fine unless Hugging Face authentication is required."
else
    echo "- HF_TOKEN is set."
fi

echo
echo "Bootstrap complete. Log: $BOOTSTRAP_LOG"
