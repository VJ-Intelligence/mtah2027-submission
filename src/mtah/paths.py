"""Filesystem locations. Match data and external data both live outside the repo."""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OFFICIAL = REPO / "vendor" / "official"
_BASE = Path.home() / "Documents" / "Business" / "VJ Intelligence" / "INTERNAL"


def ext_dir() -> Path:
    """External (openly licensed) training data. Safe to inspect."""
    p = Path(os.environ.get("MTAH_EXT_DIR", _BASE / "mtah2027-external")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def official_on_path() -> None:
    """Make the official repo importable (evaluation.*, starter.*, games)."""
    for p in (OFFICIAL, OFFICIAL / "evaluation", OFFICIAL / "starter"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
