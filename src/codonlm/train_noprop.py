#!/usr/bin/env python3
"""Train NoPropTinyGPT through the shared model-agnostic engine."""

from __future__ import annotations

import argparse
import sys
import torch
import yaml
from torch.utils.data import DataLoader

from src.training.contracts import EngineEvent
from src.training.engine import EngineConfig, TrainingEngine
from src.training.run_lifecycle import TrainingRun, configuration_fingerprint

from .data_loading import PackedDataset, dynamic_lm_collate_fn
from .model_tiny_gpt import NoPropTinyGPT
from .noprop_task import (
    NoPropTask,
    NoPropUpdateStrategy,
    adapt_noprop_checkpoint,
    decode_noprop_checkpoint,
)
from .train_codon_lm import _auto_run_id, _ensure_path_list, _normalize_run_id
from .training.vocabulary import (
    resolve_vocabulary_contract,
    snapshot_vocabulary,
    write_vocabulary_manifest,
)


class _NoPropConsole:
    def __init__(self, curves_path) -> None:
        self.curves_path = curves_path

    def on_event(self, event: EngineEvent) -> None:
        if event.name != "epoch_completed":
            return
        epoch = int(event.metadata["epoch"])
        train = event.metadata["training_metrics"]
        validation = event.metrics
        with self.curves_path.open("a") as handle:
            handle.write(
                f"{epoch},{train['loss'].total:.6f},"
                f"{validation['loss'].total:.6f}\n"
            )
        train_blocks = _format_blocks(train)
        validation_blocks = _format_blocks(validation)
        print(f"Epoch {epoch}:")
        print(f"  Train Block MSEs: {train_blocks} | CE Loss: {train['loss'].total:.4f}")
        print(
            f"  Val Block MSEs:   {validation_blocks} | "
            f"CE Loss: {validation['loss'].total:.4f}"
        )


def _format_blocks(metrics) -> str:
    names = sorted(name for name in metrics if name.startswith("block_") and name.endswith("_mse"))
    return ", ".join(
        f"B{name.removeprefix('block_').removesuffix('_mse')}:{metrics[name].total:.4f}"
        for name in names
    )


def train(config_path: str, *, run_id=None, device_name=None, noise_sigma=0.1, resume=None):
    with open(config_path) as handle:
        cfg = yaml.safe_load(handle)
    epochs = int(cfg.get("epochs", 5))
    device = (
        torch.device(device_name)
        if device_name
        else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    )
    requested_run_id = _normalize_run_id(run_id) or _auto_run_id(cfg, config_path)
    fingerprint = configuration_fingerprint({**cfg, "noise_sigma": noise_sigma})
    training_run = TrainingRun.open(
        "runs",
        requested_run_id,
        resume=resume,
        target_epochs=epochs,
        config_fingerprint=fingerprint,
    )
    logger = training_run.logger()
    logger.__enter__()
    try:
        print(f"[noprop] using device: {device}")
        print(f"[noprop] run_id: {training_run.run_dir.name}")
        train_paths = _ensure_path_list(None, cfg.get("train_npz"), "train_npz")
        val_paths = _ensure_path_list(None, cfg.get("val_npz"), "val_npz")
        contract = resolve_vocabulary_contract(
            [*train_paths, *val_paths],
            configured_path=cfg.get("itos_path"),
            configured_size=cfg.get("vocab_size"),
        )
        cfg["vocab_size"] = contract.size
        vocabulary_snapshot = snapshot_vocabulary(
            contract, training_run.run_dir / "itos.txt"
        )
        cfg["itos_path"] = str(vocabulary_snapshot)
        cfg["vocabulary"] = contract.provenance(vocabulary_snapshot)
        write_vocabulary_manifest(
            cfg["vocabulary"], training_run.run_dir / "vocabulary.json"
        )

        train_ds = PackedDataset(train_paths)
        val_ds = PackedDataset(val_paths)
        collate_fn = dynamic_lm_collate_fn if train_ds.is_dynamic else None
        seed = int(cfg.get("seed", 42))
        train_generator = torch.Generator()
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg["batch_size"]),
            shuffle=True,
            collate_fn=collate_fn,
            generator=train_generator,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg["batch_size"]),
            collate_fn=collate_fn,
        )
        model = NoPropTinyGPT(
            vocab_size=cfg["vocab_size"],
            block_size=cfg["block_size"],
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            n_embd=cfg["n_embd"],
            dropout=cfg.get("dropout", 0.1),
            sep_id=3 if cfg.get("sep_mask_enabled", True) else None,
            n_kv_head=cfg.get("n_kv_head"),
            use_sdpa=cfg.get("use_sdpa", False),
        ).to(device)
        lr = float(cfg.get("learning_rate", 5e-4))
        opt_emb = torch.optim.AdamW(
            [*model.tok_emb.parameters(), *model.pos_emb.parameters()], lr=lr
        )
        opts_blocks = [torch.optim.AdamW(block.parameters(), lr=lr) for block in model.blocks]
        opt_head = torch.optim.AdamW(
            [*model.ln_f.parameters(), *model.head.parameters()], lr=lr
        )
        task = NoPropTask(
            model=model,
            train_loader=train_loader,
            validation_loader=val_loader,
            device=device,
            noise_sigma=noise_sigma,
            train_generator=train_generator,
            seed=seed,
        )
        curves_path = training_run.scores / "curves.csv"
        if not curves_path.exists():
            curves_path.write_text("epoch,train_ce,val_ce\n")
        engine = TrainingEngine(
            task=task,
            strategy=NoPropUpdateStrategy(opt_emb, opts_blocks, opt_head),
            run=training_run,
            config=EngineConfig(
                epochs=epochs,
                grad_accum_steps=1,
                monitor="loss",
                best_checkpoint_pattern="best_epoch_{epoch:03d}.pt",
            ),
            device=device,
            callbacks=[_NoPropConsole(curves_path)],
            run_fingerprint=fingerprint,
            checkpoint_decoder=decode_noprop_checkpoint,
            checkpoint_payload_adapter=adapt_noprop_checkpoint,
        )
        return engine.fit()
    finally:
        training_run.close()
        logger.__exit__(*sys.exc_info())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--noise_sigma", type=float, default=0.1)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train(
        args.config,
        run_id=args.run_id,
        device_name=args.device,
        noise_sigma=args.noise_sigma,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
