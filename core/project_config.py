from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
INPUT_DIR = ROOT_DIR / "artifacts" / "input"
INTERIM_DIR = ROOT_DIR / "artifacts" / "interim"
RESULT_DIR = ROOT_DIR / "artifacts" / "result"

MODELS_DIR = ROOT_DIR.parent.parent / "models"