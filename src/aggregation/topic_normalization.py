import re
import unicodedata


def normalize_topic_key(topic: str) -> str:
    """Normalize deterministic topic variants for aggregation bookkeeping."""
    stripped = str(topic).strip()
    stripped = stripped.strip(
        "".join(
            ch for ch in stripped if unicodedata.category(ch).startswith("P")
        )
    )
    return re.sub(r"\s+", " ", stripped).casefold()
