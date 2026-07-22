#!/usr/bin/env python3
"""Verify a completed corrected dataset against its tracked release contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.freeze_corrected_datasets import (
    load_and_validate_freeze,
    validate_freeze_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    args = parser.parse_args()

    index = load_and_validate_freeze(args.freeze)
    contract = json.loads(args.expected.read_text())
    validate_freeze_contract(index, contract)
    print(
        f"[freeze] verified release={contract['release']} "
        f"freeze_id={index['freeze_id']}"
    )


if __name__ == "__main__":
    main()
