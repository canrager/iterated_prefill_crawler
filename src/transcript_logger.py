"""Global transcript logger for all model I/O.

Captures every prompt and generation that passes through batch_generate()
or async_query_openrouter(), writing one JSONL record per call to
artifacts/transcripts/{run_name}.jsonl.

No-ops silently when init_transcript_log() has not been called (tests,
standalone scripts, aggregation).
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_log_path: Optional[Path] = None
_lock = threading.Lock()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def init_transcript_log(run_name: str, output_dir: Optional[str] = None) -> Path:
    """Initialize the transcript log for a run. Call once at startup.

    If output_dir is given, the transcript is written there (alongside
    crawler output).  Otherwise falls back to artifacts/transcripts/.
    """
    global _log_path
    if output_dir is not None:
        transcript_dir = Path(output_dir)
    else:
        transcript_dir = PROJECT_ROOT / "artifacts" / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    _log_path = transcript_dir / f"{run_name}.jsonl"
    return _log_path


def log_model_call(
    *,
    call_type: str,
    model: str,
    inputs: Any,
    outputs: Any,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    batch_size: int = 1,
) -> None:
    """Append one record to the transcript log. No-op if not initialized."""
    if _log_path is None:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "call_type": call_type,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "batch_size": batch_size,
        "inputs": inputs,
        "outputs": outputs,
    }
    line = json.dumps(record, ensure_ascii=False)
    with _lock:
        with open(_log_path, "a") as f:
            f.write(line + "\n")
