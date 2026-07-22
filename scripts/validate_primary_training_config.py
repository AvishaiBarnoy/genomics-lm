#!/usr/bin/env python3
"""Validate an immutable corrected primary-training YAML contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.codonlm.training.primary_contract import (
    load_and_validate_primary_training_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    result = load_and_validate_primary_training_config(args.config)
    print(json.dumps({"status": "passed", **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
