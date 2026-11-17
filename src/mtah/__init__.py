"""VJ Intelligence entry for More than a Hack 2027, Phase 1 (ball possession)."""
import os
from pathlib import Path

def data_dir() -> Path:
    p = os.environ.get("MTAH_DATA_DIR")
    if not p:
        raise SystemExit("Set MTAH_DATA_DIR to the unpacked match bundle (outside this repo).")
    return Path(p).expanduser()
