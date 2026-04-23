"""Convert an existing live-run transcript JSONL into a crawler-shape fixture.

The offline bench's ``capture_crawler_fixture.py`` monkey-patches the
transcript logger to emit fixture records during a live run. But the logger's
output at ``artifacts/out/crawler_out_*.jsonl`` already carries the fields
we need -- ``call_type``, ``model``, ``inputs``, ``outputs``, ``temperature``,
``max_tokens``. This converter repackages those existing records into the
fixture format so an older live run can seed an offline sweep without a
fresh API spend.

Usage:
    python scripts/convert_transcript_to_fixture.py \\
        --transcript artifacts/out/crawler_out_20260423_020441_deepseek-v3.2_5samples_2crawls_Truefilter.jsonl \\
        --run-id run3_ds_v32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from src.crawler_shape_bench import (
    iter_fixture_records_from_model_call,
    write_fixture_records,
)


SUPPORTED_CALL_TYPES = {"batch_generate_api", "async_query_openrouter"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transcript", required=True, type=Path,
                   help="Path to crawler_out_*.jsonl from a prior live run.")
    p.add_argument("--run-id", required=True,
                   help="Fixture run identifier. Output dir will be "
                        "artifacts/crawler_shape_fixtures/<run-id>/.")
    p.add_argument("--fixture-root", type=Path,
                   default=Path("artifacts/crawler_shape_fixtures"),
                   help="Parent directory for fixture bundles.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.fixture_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    responses_path = out_dir / "responses.jsonl"
    # Fresh file -- don't accidentally append onto an existing fixture.
    if responses_path.exists():
        responses_path.unlink()

    # Write a config.json matching the shape FixtureCaptureWriter.finalize
    # produces, so load_fixture_bundle can consume it.  The crawler's own
    # output JSON carries the live CrawlerConfig under "config" -- reuse it.
    sibling = args.transcript.with_suffix(".json")
    crawler_config = None
    if sibling.exists():
        data = json.loads(sibling.read_text())
        crawler_config = data.get("config") or data
    payload = {
        "schema_version": 1,
        "captured_at": None,
        "responses_path": str(responses_path),
        "records_written": 0,  # filled in below after writing
        "hydra_overrides": [],
        "crawler_config": crawler_config,
        "live_summary": {
            "source": "converted from transcript",
            "transcript": str(args.transcript),
        },
    }

    total_lines = 0
    skipped_call_type = 0
    skipped_shape = 0
    written = 0

    with args.transcript.open() as handle:
        for line in handle:
            total_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped_shape += 1
                continue
            call_type = rec.get("call_type")
            if call_type not in SUPPORTED_CALL_TYPES:
                skipped_call_type += 1
                continue
            try:
                records = list(iter_fixture_records_from_model_call(
                    call_type=call_type,
                    model=rec.get("model") or "",
                    inputs=rec.get("inputs"),
                    outputs=rec.get("outputs"),
                    temperature=rec.get("temperature"),
                    max_tokens=rec.get("max_tokens"),
                ))
            except (ValueError, TypeError):
                skipped_shape += 1
                continue
            written += write_fixture_records(responses_path, records)

    payload["records_written"] = written
    (out_dir / "config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2)
    )

    print(f"Input:    {args.transcript} ({total_lines} lines)")
    print(f"Output:   {responses_path}")
    print(f"Config:   {out_dir / 'config.json'}")
    print(f"Wrote:    {written} fixture records")
    print(f"Skipped:  {skipped_call_type} other-call-type, {skipped_shape} malformed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
