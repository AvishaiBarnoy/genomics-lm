#!/usr/bin/env python3
import argparse
from pathlib import Path
from src.codonlm.sample import main as sample_main

# Thin wrapper to expose generation with stop flags.

if __name__ == "__main__":
    sample_main()

