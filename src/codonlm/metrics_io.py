#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping


def read_metrics(path: str | Path) -> Dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def write_merge_metrics(path: str | Path, updates: Mapping) -> Dict:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    base = read_metrics(p)
    merged = dict(base)
    merged.update(dict(updates))
    p.write_text(json.dumps(merged, indent=2) + "\n")
    return merged


__all__ = ["read_metrics", "write_merge_metrics"]

