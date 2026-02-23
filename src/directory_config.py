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

    Args:
        cache_dir: Absolute path to the model cache directory

    Returns:
        Path: Absolute path to the cache directory
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path
