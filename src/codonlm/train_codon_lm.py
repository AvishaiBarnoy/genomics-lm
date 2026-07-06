#!/usr/bin/env python3
"""
Lightweight CLI wrapper for CodonLM training.
Forwards CLI arguments to the modular training loop implementation.
"""

from __future__ import annotations

import argparse
import yaml

from src.codonlm.training.config import (
    write_meta,
    _ensure_path_list,
    _normalize_run_id,
    _auto_run_id,
    _prepare_output_dirs,
    _normalize_offset_weights,
)
from src.codonlm.training.checkpoint import _read_itos, _load_transfer_state_dict
from src.codonlm.training.objectives import (
    multi_offset_lm_loss,
    offset_target_mask,
    termination_aux_loss,
    termination_distance_bucket_labels,
)
from src.codonlm.training.loop import dev, run_training
from src.codonlm.data_loading import PackedDataset, dataset_length_audit

# Re-export variables for backwards-compatibility
RUN_ID_ENV = "RUN_ID"
PAD_ID = 0
DEFAULT_BOUNDARY_IDS = (2, 3)  # <EOS_CDS>, <SEP>


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_id", default=None, help=f"Unique run id; falls back to $RUN_ID or config.run_id")
    ap.add_argument("--resume", default=None, help="Path to checkpoint to resume training from")
    ap.add_argument("--transfer_from", default=None, help="Path to pre-trained weights to initialize model from (ignores optimizer/step state)")
    ap.add_argument("--train_npz", action="append", default=None, help="Training NPZ file (repeatable)")
    ap.add_argument("--val_npz", action="append", default=None, help="Validation NPZ file (repeatable)")
    ap.add_argument("--test_npz", action="append", default=None, help="Test NPZ file (repeatable)")
    ap.add_argument("--save_epochs", action="store_true", help="Save checkpoint at every epoch")
    ap.add_argument("--max_time_minutes", type=float, default=None, help="Override config max_time_minutes")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)) or {}
    if "data" in cfg and isinstance(cfg["data"], dict):
        for k, v in cfg["data"].items():
            cfg.setdefault(k, v)
    cfg["save_epochs"] = args.save_epochs or cfg.get("save_epochs", False)
    if args.max_time_minutes is not None:
        cfg["max_time_minutes"] = float(args.max_time_minutes)

    run_training(cfg, args)


if __name__ == "__main__":
    main()
