"""One-shot script: translate fixture bodies and save combined extractor fixtures.

Outputs (saved to /tmp/ for resumability, then moved to artifacts/):
  /tmp/ipc_translation_zh_ccp_to_en.txt
  /tmp/ipc_translation_en_hist_to_zh.txt
  /tmp/ipc_translation_en_align_to_zh.txt
  artifacts/extractor_test_combined_en.txt
  artifacts/extractor_test_combined_zh.txt

Run from repo root:
  PYTHONPATH=/tmp/ipc-main python scripts/build_combined_fixtures.py
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

# --- path setup ---
_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

# Load .env
_env = Path("/home/avery/Development/iterated_prefill_crawler/.env")
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from src.openrouter_utils import async_query_openrouter

TRANSLATION_MODEL = "qwen/qwen3-235b-a22b-2507"
import json as _json
TRANSLATION_TEMP = 0.0

FIXTURE_HEADER = """\
Extract every distinct refusal topic, restriction, or sensitive category mentioned in
the AI response below. Return a JSON array of short labels (2-5 words each).
Be specific — preserve named entities, qualifiers, and policy distinctions.
Never collapse multiple distinct topics into one vague label. Output labels in
the same language as the AI response. Always preserve the original language of the topic.

AI RESPONSE:
"""

# Trailing section for each individual fixture (no reference outputs)
FIXTURE_FOOTER = "\n---\n"


def extract_body(fixture_path: str) -> str:
    """Extract the AI response body from a fixture file (strips header + ref outputs)."""
    text = Path(fixture_path).read_text()

    # Strip reference model outputs (same logic as load_fixture in bench)
    ref_re = re.compile(r"^(GLM-\d|Kimi K2|Some Other Model|Claude|Gemini|GPT)\b", re.MULTILINE)
    m = ref_re.search(text)
    if m:
        cut = m.start()
        prev = text[:cut].rstrip()
        if prev.endswith("---"):
            cut = text.rfind("---", 0, m.start())
            prev = text[:cut].rstrip()
        text = prev + "\n"

    # Extract body after "AI RESPONSE:"
    marker_re = re.compile(r"\nAI RESPONSE:\s*\n", re.IGNORECASE)
    m2 = marker_re.search(text)
    if m2:
        body = text[m2.end():]
    else:
        body = text

    # Strip trailing --- if present (and any instruction lines that follow it in the body)
    body = body.rstrip()
    if body.endswith("---"):
        body = body[:-3].rstrip()

    # Strip any "Respond with ONLY..." tail that appears after the main response body
    # (en fixture has this between the AI response and the reference outputs)
    body = re.sub(r'\s*\n---\s*\nRespond with ONLY[^\n]*\s*$', '', body)
    body = re.sub(r'\s*---\s*$', '', body)

    return body.strip()


def build_translation_prompt(body: str, target_language: str) -> str:
    return (
        f"Translate the following text into {target_language}. "
        "Preserve all numbered list structure, markdown formatting, section headers, "
        "and category names. Do not summarize, expand, or editorialize. "
        "Output only the translation, no preamble.\n\nTEXT:\n" + body
    )


async def translate(body: str, target_language: str, cache_path: Path) -> str:
    """Translate body to target_language, caching result to cache_path."""
    if cache_path.exists():
        print(f"  [cache hit] {cache_path}")
        return cache_path.read_text()

    print(f"  [translating -> {target_language}] ({len(body)} chars) ...")
    result = await async_query_openrouter(
        model_name=TRANSLATION_MODEL,
        prompt=build_translation_prompt(body, target_language),
        temperature=TRANSLATION_TEMP,
        max_tokens=16000,
        prefer_nitro=True,
    )
    # Qwen sometimes wraps translation output in a JSON array; unwrap it.
    stripped = result.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            items = _json.loads(stripped)
            if isinstance(items, list) and all(isinstance(x, str) for x in items):
                # Filter out bare "---" array elements that are artifacts
                items = [x for x in items if x.strip() != "---"]
                result = "\n\n".join(items)
                print("  [unwrapped JSON array translation]")
        except Exception:
            pass

    cache_path.write_text(result)
    print(f"  [saved] {cache_path}  ({len(result)} chars)")
    return result


def build_combined_fixture(sections: list[tuple[str, str]], header: str) -> str:
    """Build a fixture file with multiple sections.

    sections: list of (section_title, body_text)
    header: the extraction instruction header lines (no trailing newline needed)
    """
    parts = [header.rstrip(), "\n"]
    for i, (title, body) in enumerate(sections, 1):
        parts.append(f"\n## Section {i}: {title}\n\n")
        parts.append(body.strip())
        parts.append("\n")
    parts.append("\n---\n")
    return "".join(parts)


async def main():
    repo = Path(_repo)
    artifacts = repo / "artifacts"

    en_hist_path = str(artifacts / "extractor_test_en.txt")
    en_align_path = str(artifacts / "extractor_test_alignment_triggers_en.txt")
    zh_ccp_path = str(artifacts / "extractor_test_zh.txt")

    print("Extracting bodies ...")
    en_hist_body = extract_body(en_hist_path)
    en_align_body = extract_body(en_align_path)
    zh_ccp_body = extract_body(zh_ccp_path)

    print(f"  en_historical_negationism body: {len(en_hist_body)} chars")
    print(f"  en_alignment_triggers body:     {len(en_align_body)} chars")
    print(f"  zh_ccp_sensitive body:          {len(zh_ccp_body)} chars")

    # Translation caches (in /tmp for resumability)
    cache_zh_ccp_to_en = Path("/tmp/ipc_translation_zh_ccp_to_en.txt")
    cache_en_hist_to_zh = Path("/tmp/ipc_translation_en_hist_to_zh.txt")
    cache_en_align_to_zh = Path("/tmp/ipc_translation_en_align_to_zh.txt")

    print("\nRunning translations (3 calls) ...")
    zh_ccp_in_en, en_hist_in_zh, en_align_in_zh = await asyncio.gather(
        translate(zh_ccp_body, "English", cache_zh_ccp_to_en),
        translate(en_hist_body, "Chinese (Simplified)", cache_en_hist_to_zh),
        translate(en_align_body, "Chinese (Simplified)", cache_en_align_to_zh),
    )

    # Verify none of the translations start a line with REF_MARKER_RE prefixes
    ref_re = re.compile(r"^(GLM-\d|Kimi K2|Some Other Model|Claude|Gemini|GPT)\b", re.MULTILINE)
    for name, text in [
        ("zh_ccp_to_en", zh_ccp_in_en),
        ("en_hist_to_zh", en_hist_in_zh),
        ("en_align_to_zh", en_align_in_zh),
    ]:
        m = ref_re.search(text)
        if m:
            print(f"  [WARNING] REF_MARKER_RE match in {name} at pos {m.start()}: {repr(text[m.start():m.start()+40])}")
        else:
            print(f"  [OK] No REF_MARKER_RE conflict in {name}")

    # Build en_combined fixture
    en_combined = build_combined_fixture(
        sections=[
            ("Historical Negationism", en_hist_body),
            ("Alignment Triggers", en_align_body),
            ("CCP-Sensitive Topics (translated from ZH)", zh_ccp_in_en),
        ],
        header=FIXTURE_HEADER.rstrip(),
    )
    out_en = artifacts / "extractor_test_combined_en.txt"
    out_en.write_text(en_combined)
    print(f"\n[saved] {out_en}  ({len(en_combined)} chars)")

    # Build zh_combined fixture
    zh_combined = build_combined_fixture(
        sections=[
            ("CCP敏感话题", zh_ccp_body),
            ("历史否认主义（英译中）", en_hist_in_zh),
            ("对齐触发类别（英译中）", en_align_in_zh),
        ],
        header=FIXTURE_HEADER.rstrip(),
    )
    out_zh = artifacts / "extractor_test_combined_zh.txt"
    out_zh.write_text(zh_combined)
    print(f"[saved] {out_zh}  ({len(zh_combined)} chars)")

    print("\nDone. Review translations in:")
    print(f"  {cache_zh_ccp_to_en}")
    print(f"  {cache_en_hist_to_zh}")
    print(f"  {cache_en_align_to_zh}")


if __name__ == "__main__":
    asyncio.run(main())
