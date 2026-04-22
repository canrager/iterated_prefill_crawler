set -a ; source /home/avery/Development/iterated_prefill_crawler/.env ; set +a
set -x PYTHONPATH /tmp/ipc-main
/home/avery/Development/iterated_prefill_crawler/.venv/bin/python -u scripts/bench_extractor_models.py | tee "artifacts/log/bench_$(date +%Y-%m-%d_%H-%M-%S).log"
