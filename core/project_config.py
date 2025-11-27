from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
INPUT_DIR = ROOT_DIR / "artifacts" / "input"
INTERIM_DIR = ROOT_DIR / "artifacts" / "interim"
RESULT_DIR = ROOT_DIR / "artifacts" / "result"

MODELS_DIR = ROOT_DIR.parent
DEVICE = "cuda"


def resolve_cache_dir(cache_dir: str = None) -> Path:
    """
    Resolve cache_dir path relative to ROOT_DIR.parent.
    If cache_dir is None or empty, defaults to MODELS_DIR / "models".
    Creates the directory if it doesn't exist.

    Args:
        cache_dir: Relative path (e.g., "models") or None for default

    Returns:
        Path: Absolute path to the cache directory
    """
    if cache_dir is None or cache_dir == "":
        cache_path = MODELS_DIR / "models"
    else:
        # If it's already an absolute path, use it as-is
        cache_path = Path(cache_dir)
        if not cache_path.is_absolute():
            # Resolve relative to ROOT_DIR.parent
            cache_path = MODELS_DIR / cache_dir

    # Create directory if it doesn't exist
    cache_path.mkdir(parents=True, exist_ok=True)

    return cache_path
