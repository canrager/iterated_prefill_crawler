from pathlib import Path


# Project directories
ROOT_DIR = Path(__file__).parent.parent
INPUT_DIR = ROOT_DIR / "artifacts" / "input"
INTERIM_DIR = ROOT_DIR / "artifacts" / "out"
RESULT_DIR = ROOT_DIR / "artifacts" / "result"
CONFIG_DIR = ROOT_DIR / "configs"

for _dir in (INPUT_DIR, INTERIM_DIR, RESULT_DIR, CONFIG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# Project directory helper functions
def resolve_cache_dir(cache_dir: str) -> Path:
    """
    Resolve cache_dir to an absolute Path and create it if it doesn't exist.
    Relative paths are resolved against ROOT_DIR.

    Args:
        cache_dir: Path to the model cache directory (absolute or relative to project root)

    Returns:
        Path: Absolute path to the cache directory
    """
    cache_path = Path(cache_dir)
    if not cache_path.is_absolute():
        cache_path = ROOT_DIR / cache_path
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path
