from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.crawler_shape_bench import format_scoreboard_diff, run_bench


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline replay sweep over crawler-shape hyperparameters."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--fixture-dir",
        help="Path to artifacts/crawler_shape_fixtures/<run_id>.",
    )
    mode.add_argument(
        "--diff",
        help="Existing scoreboard JSON to render as a compact baseline diff report.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional scoreboard output path. Defaults to artifacts/bench/crawler_shape_<timestamp>.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.diff:
        with Path(args.diff).open(encoding="utf-8") as handle:
            scoreboard = json.load(handle)
        print(format_scoreboard_diff(scoreboard))
        return

    fixture_dir = Path(args.fixture_dir)
    output_path = (
        Path(args.output)
        if args.output
        else Path("artifacts/bench")
        / f"crawler_shape_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    scoreboard = run_bench(fixture_dir=fixture_dir, output_path=output_path)
    valid = sum(1 for cell in scoreboard["cells"] if cell["valid"])
    invalid = len(scoreboard["cells"]) - valid
    print(f"Bench scoreboard: {output_path}")
    print(f"Cells: {len(scoreboard['cells'])} total, {valid} valid, {invalid} invalid")


if __name__ == "__main__":
    main()
