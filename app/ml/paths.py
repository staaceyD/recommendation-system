import os
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parents[2] / "instance" / "mf"


def model_dir() -> Path:
    """Directory the trained matrix-factorization artifact is saved to / loaded from."""
    return Path(os.environ.get("MODEL_DIR", _DEFAULT))
