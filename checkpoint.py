"""Checkpoint load/save for resume support."""

import json
from pathlib import Path

from config import CHECKPOINT_FILE


def load_checkpoint() -> dict:
    """Load checkpoint; returns empty dict if file does not exist."""
    path = Path(CHECKPOINT_FILE)
    if not path.exists():
        return {"results": {}, "model": None}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(checkpoint: dict) -> None:
    """Persist checkpoint to disk."""
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)
