import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from scripts.cleanup_runs import main, PROTECTED_FILENAMES


def test_cleanup_runs_keep_only_best(tmp_path: Path):
    # Set up mock runs directory structure
    runs_dir = tmp_path / "runs"
    run_folder = runs_dir / "mock_run"
    chk_dir = run_folder / "checkpoints"
    chk_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock checkpoint files
    best_file = chk_dir / "best.pt"
    last_file = chk_dir / "last.pt"
    other_file = chk_dir / "epoch_5.pt"
    
    best_file.write_text("dummy best")
    last_file.write_text("dummy last")
    other_file.write_text("dummy other")
    
    # Mock repo_dir and sys.argv / argparse
    mock_args = MagicMock()
    mock_args.dry_run = False
    mock_args.keep_only_best = True
    mock_args.older_than_days = None
    mock_args.force = True
    
    with patch("scripts.cleanup_runs.REPO_DIR", tmp_path), \
         patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        main()
        
    # Check that best.pt is preserved, but last.pt and epoch_5.pt are deleted
    assert best_file.exists()
    assert not last_file.exists()
    assert not other_file.exists()
