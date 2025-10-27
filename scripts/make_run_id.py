#!/usr/bin/env python3
"""
Generate a RUN_ID from a YAML config path and current date.

Format: YYYY-MM-DD_<model>_<nL><nH>_d<n_embd>_e<epochs>
Example: 2025-10-06_tiny_2L4H_d128_e5

Usage:
  python -m scripts.make_run_id path/to/config.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import yaml


def infer_model_tag(config_path: Path) -> str:
    stem = config_path.stem  # e.g., tiny_mps_v2 -> 'tiny_mps_v2'
    # take the first segment before an underscore as the short model tag
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def make_run_id(config_path: Path, cfg: dict) -> str:
    today = dt.date.today().strftime("%Y-%m-%d")
    model = infer_model_tag(config_path)
    n_layer = int(cfg.get("n_layer", 0) or 0)
    n_head = int(cfg.get("n_head", 0) or 0)
    n_embd = int(cfg.get("n_embd", 0) or 0)
    epochs = int(cfg.get("epochs", 0) or 0)
    return f"{today}_{model}_{n_layer}L{n_head}H_d{n_embd}_e{epochs}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    path = Path(args.config)
    cfg = yaml.safe_load(path.read_text()) or {}
    print(make_run_id(path, cfg))


if __name__ == "__main__":
    main()

