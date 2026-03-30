#!/usr/bin/env python3
"""Run parallel crawler instances with output directed to log files.

Ensemble mode (same config, multiple runs):
    python scripts/run_parallel.py --num-runs 5 model=ds-v32_remote crawler=default

Multi-model mode (different models, one run each):
    python scripts/run_parallel.py --models ds-r1_remote,sonnet-45_remote crawler=default

Extra hydra overrides are passed through:
    python scripts/run_parallel.py --num-runs 3 model=ds-v32_remote crawler=default crawler.num_crawl_steps=10
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run parallel crawler instances",
        # Allow unknown args to pass through as hydra overrides
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--num-runs", type=int, help="Number of identical ensemble runs")
    group.add_argument("--models", type=str, help="Comma-separated list of model config names")
    return parser.parse_known_args()


def find_hydra_arg(hydra_args: list[str], prefix: str) -> str | None:
    for arg in hydra_args:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def launch_run(cmd: list[str], log_path: Path, env: dict) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=PROJECT_ROOT,
        env=env,
    )
    proc._log_file = log_file  # keep reference so it isn't GC'd
    return proc


def main():
    args, hydra_args = parse_args()

    # Validate crawler= is present
    if find_hydra_arg(hydra_args, "crawler=") is None:
        configs = sorted(p.stem for p in (PROJECT_ROOT / "configs" / "crawler").glob("*.yaml"))
        print(f"Error: missing required argument 'crawler=<name>'")
        print(f"Available: {' '.join(configs)}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + f":{PROJECT_ROOT}"

    # Load .env if present
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()

    python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    base_cmd = [python, "src/crawler/run_crawler.py"]

    # Build list of (tag, cmd) pairs
    runs: list[tuple[str, list[str]]] = []

    if args.num_runs is not None:
        # Ensemble mode
        model_name = find_hydra_arg(hydra_args, "model=")
        if model_name is None:
            configs = sorted(p.stem for p in (PROJECT_ROOT / "configs" / "model").glob("*.yaml"))
            print(f"Error: missing required argument 'model=<name>' for ensemble mode")
            print(f"Available: {' '.join(configs)}")
            sys.exit(1)

        out_dir = PROJECT_ROOT / "artifacts" / "out" / f"ensemble_{timestamp}_{model_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "num_runs": args.num_runs,
            "timestamp": timestamp,
            "model": model_name,
            "hydra_args": " ".join(hydra_args),
        }
        (out_dir / "ensemble_meta.json").write_text(json.dumps(meta, indent=4))

        for i in range(1, args.num_runs + 1):
            tag = f"run{i:02d}"
            cmd = base_cmd + hydra_args + [
                f"crawler.output_dir={out_dir}",
                f"crawler.run_tag={tag}",
            ]
            runs.append((tag, cmd))

        print(f"Ensemble dir: {out_dir}")
        print(f"Launching {args.num_runs} parallel runs...\n")

    else:
        # Multi-model mode
        if find_hydra_arg(hydra_args, "model=") is not None:
            print("Error: do not pass model= in overrides when using --models")
            sys.exit(1)

        model_list = [m.strip() for m in args.models.split(",") if m.strip()]
        if not model_list:
            print("Error: --models requires at least one model config name")
            sys.exit(1)

        out_dir = PROJECT_ROOT / "artifacts" / "out" / f"multi_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "num_models": len(model_list),
            "timestamp": timestamp,
            "models": model_list,
            "hydra_args": " ".join(hydra_args),
        }
        (out_dir / "multi_meta.json").write_text(json.dumps(meta, indent=4))

        for model in model_list:
            cmd = base_cmd + [f"model={model}"] + hydra_args + [
                f"crawler.output_dir={out_dir}",
                f"crawler.run_tag={model}",
            ]
            runs.append((model, cmd))

        print(f"Multi-model dir: {out_dir}")
        print(f"Launching {len(model_list)} parallel runs: {', '.join(model_list)}\n")

    # Launch all processes
    procs: list[tuple[str, subprocess.Popen]] = []
    for tag, cmd in runs:
        log_path = out_dir / f"{tag}.log"
        proc = launch_run(cmd, log_path, env)
        procs.append((tag, proc))
        print(f"  [{tag}] PID={proc.pid}, log: {log_path}")

    # Wait for completion
    print(f"\nWaiting for all {len(procs)} runs to complete...")
    failures = 0
    for tag, proc in procs:
        ret = proc.wait()
        proc._log_file.close()
        if ret == 0:
            print(f"  [{tag}] PID={proc.pid} completed successfully")
        else:
            print(f"  [{tag}] PID={proc.pid} FAILED (exit code {ret})")
            failures += 1

    total = len(procs)
    print(f"\nComplete: {total - failures}/{total} succeeded")
    print(f"Results in: {out_dir}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
