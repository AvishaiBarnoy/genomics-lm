#!/usr/bin/env python3
"""Validate a versioned CodonLM dataset manifest and its artifacts."""

import argparse
import json

from src.codonlm.dataset_manifest import load_dataset_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--structure-only", action="store_true")
    args = parser.parse_args()
    manifest = load_dataset_manifest(
        args.manifest, verify_artifacts=not args.structure_only
    )
    print(
        json.dumps(
            {
                "dataset_id": manifest["dataset"]["id"],
                "schema": manifest["schema"],
                "scientific_valid": manifest["dataset"]["scientific_valid"],
                "artifact_count": len(manifest["artifacts"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
