"""Checkpoint load/save for resume support. Per-provider sections."""

import json
from pathlib import Path

from config import CHECKPOINT_FILE


def load_checkpoint(provider: str) -> dict:
    """Load checkpoint for a given provider; returns empty dict if not found."""
    path = Path(CHECKPOINT_FILE)
    if not path.exists():
        return {"results": {}, "model": None}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get(provider, {"results": {}, "model": None})


def save_checkpoint(provider: str, checkpoint: dict) -> None:
    """Persist checkpoint for a given provider to disk."""
    path = Path(CHECKPOINT_FILE)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    data[provider] = checkpoint
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
