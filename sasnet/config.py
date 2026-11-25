"""JSON run configs.  ``"base"`` names another config (relative path) whose keys are merged recursively underneath."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = json.loads(path.read_text())
    base = cfg.pop("base", None)
    if base is not None:
        cfg = _merge(load_config(path.parent / base), cfg)
    return cfg
