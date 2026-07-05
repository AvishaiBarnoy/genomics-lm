#!/usr/bin/env python3
"""
Generate markdown summaries for all existing and completed run directories under runs/.
Can also be called for a single run directory at the end of training.
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Any, Dict

def parse_run(run_dir: Path) -> Dict[str, Any]:
    run_id = run_dir.name
    summary_data = {
        "run_id": run_id,
        "status": "unknown",
        "epochs": "unknown",
        "best_epoch": "unknown",
        "best_val_loss": "unknown",
        "last_perplexity": "unknown",
        "wall_time": "unknown",
        "model_spec": {},
        "config": {},
    }

    # Try loading config
    config_path = run_dir / "checkpoints" / "config.yaml"
    if not config_path.exists():
        config_path = run_dir / "config.yaml"

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                summary_data["config"] = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error reading config for {run_id}: {e}", file=sys.stderr)

    # Try loading metrics
    metrics_path = run_dir / "scores" / "metrics.json"
    if not metrics_path.exists():
        metrics_path = run_dir / "checkpoints" / "meta.json"
    if not metrics_path.exists():
        metrics_path = run_dir / "meta.json"

    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                meta = json.load(f)
                summary_data["status"] = meta.get("status", "unknown")
                summary_data["epochs"] = meta.get("last_epoch", meta.get("epochs", "unknown"))
                summary_data["best_epoch"] = meta.get("best_epoch", "unknown")
                summary_data["best_val_loss"] = meta.get("best_val_loss", "unknown")
                summary_data["last_perplexity"] = meta.get("last_perplexity", "unknown")
                
                wall_sec = meta.get("train_wall_sec")
                if wall_sec is not None:
                    h = int(wall_sec // 3600)
                    m = int((wall_sec % 3600) // 60)
                    s = int(wall_sec % 60)
                    if h > 0:
                        summary_data["wall_time"] = f"{h}h {m}m {s}s"
                    else:
                        summary_data["wall_time"] = f"{m}m {s}s"
                
                summary_data["model_spec"] = meta.get("model_spec", {})
        except Exception as e:
            print(f"Error reading metrics for {run_id}: {e}", file=sys.stderr)

    # Fallback status if missing but log file or checkpoints exist
    if summary_data["status"] == "unknown":
        if (run_dir / "checkpoints" / "best.pt").exists() or (run_dir / "checkpoints" / "last.pt").exists():
            summary_data["status"] = "stopped"
        log_file = run_dir / "logs" / "train.log"
        if log_file.exists():
            try:
                log_content = log_file.read_text()
                if "training failed" in log_content.lower() or "error" in log_content.lower():
                    summary_data["status"] = "failed"
                elif "completed" in log_content.lower():
                    summary_data["status"] = "completed"
            except Exception:
                pass

    return summary_data

def make_markdown(data: Dict[str, Any]) -> str:
    cfg = data["config"]
    spec = data["model_spec"]
    
    # Merge spec keys into cfg if empty/missing
    for k, v in spec.items():
        if k not in cfg or cfg[k] is None:
            cfg[k] = v

    run_id = data["run_id"]
    status = data["status"]
    
    # Determine emoji for status
    status_emoji = "⚪"
    if status == "completed":
        status_emoji = "✅"
    elif status == "failed":
        status_emoji = "❌"
    elif status == "stopped":
        status_emoji = "⏹️"
        
    md = []
    md.append(f"# Run Summary: `{run_id}`")
    md.append("")
    md.append("## 📊 Status & Key Performance Indicators (KPIs)")
    md.append(f"- **Status:** {status_emoji} {status.capitalize()}")
    md.append(f"- **Epochs Trained:** {data['epochs']}")
    md.append(f"- **Best Epoch:** {data['best_epoch']}")
    
    best_loss = data['best_val_loss']
    if isinstance(best_loss, float):
        md.append(f"- **Best Validation Loss:** {best_loss:.4f}")
    else:
        md.append(f"- **Best Validation Loss:** {best_loss}")
        
    perp = data['last_perplexity']
    if isinstance(perp, float):
        md.append(f"- **Final Perplexity:** {perp:.4f}")
    else:
        md.append(f"- **Final Perplexity:** {perp}")
        
    md.append(f"- **Wall Time:** {data['wall_time']}")
    md.append("")
    
    # Heuristics for intent / focus
    focus = []
    
    # Check for stop codon placement / termination auxiliary head
    if cfg.get("termination_loss_enabled") or spec.get("termination_aux"):
        weight = cfg.get("termination_loss_weight", "N/A")
        focus.append(f"- **Stop Codon Placement:** Adds an auxiliary head (loss weight: `{weight}`) to predict distance-to-stop thresholds for training stability and structural awareness.")
    
    # Check for long/short range prediction offsets
    offsets = cfg.get("multi_offset_targets") or spec.get("multi_offset_targets")
    if offsets:
        focus.append(f"- **Long/Short Range Predictions:** Adds auxiliary prediction heads for targets at offsets `{offsets}` to encourage encoding of multi-range contexts.")
        
    # Check for transfer learning
    transfer_from = cfg.get("transfer_from") or spec.get("transfer")
    if transfer_from:
        focus.append(f"- **Transfer Learning:** Initializes model weights from pre-trained checkpoint `{transfer_from}`.")
        
    # Check for GQA
    n_kv_head = cfg.get("n_kv_head") or spec.get("n_kv_head")
    n_head = cfg.get("n_head") or spec.get("n_head")
    if n_kv_head and n_head and n_kv_head < n_head:
        focus.append(f"- **Grouped-Query Attention (GQA):** Configured with `{n_kv_head}` KV heads vs `{n_head}` query heads to reduce memory footprint and parameter count.")
        
    # Check for memory mapping
    if cfg.get("use_mmap"):
        focus.append("- **Memory-Mapped Dataset:** Uses disk-backed `MmapPackedDataset` to stream data on-demand, reducing peak startup RAM.")
        
    # Check for gradient checkpointing
    if cfg.get("use_checkpoint") or cfg.get("grad_checkpointing"):
        focus.append("- **Gradient Checkpointing:** Trades computation for memory by recalculating activations during the backward pass.")
        
    # Check for NoProp
    if "noprop" in run_id.lower():
        focus.append("- **NoProp Integration:** Testing forward-only / local feedback training updates without global backpropagation.")

    if not focus:
        focus.append("- **Standard Baseline:** Baseline CodonLM training.")

    md.append("## 🎯 Key Focus & Tested Features")
    md.extend(focus)
    md.append("")
    
    # Dataset information
    md.append("## 🧬 Datasets")
    train_npz = cfg.get("train_npz") or cfg.get("train_paths")
    if train_npz:
        md.append(f"- **Training Data (NPZ):** `{train_npz}`")
    val_npz = cfg.get("val_npz") or cfg.get("val_paths")
    if val_npz:
        md.append(f"- **Validation Data (NPZ):** `{val_npz}`")
        
    datasets = cfg.get("datasets")
    if datasets and isinstance(datasets, list):
        md.append("- **GenBank Sources:**")
        for ds in datasets:
            if isinstance(ds, dict):
                name = ds.get("name", "unknown")
                min_len = ds.get("min_len", "N/A")
                md.append(f"  - `{name}` (min_len: {min_len})")
    md.append("")
    
    # Model Specs
    md.append("## 🧠 Model Architecture & Settings")
    md.append(f"- **Layers (L):** {cfg.get('n_layer', 'N/A')} | **Heads (H):** {cfg.get('n_head', 'N/A')} | **Embedding Dim (D):** {cfg.get('n_embd', 'N/A')}")
    md.append(f"- **Attention Type:** {'SDPA (Fused Causal)' if cfg.get('use_sdpa') else 'Standard MHA'}")
    md.append(f"- **Context Block Size:** {cfg.get('block_size', 'N/A')} codons")
    md.append(f"- **Vocab Size:** {cfg.get('vocab_size', 'N/A')}")
    md.append("")
    
    # Training Parameters
    md.append("## ⚙️ Training Optimizations")
    md.append(f"- **Optimizer:** `{cfg.get('optimizer', 'adamw')}`")
    bsz = cfg.get("batch_size")
    gacc = cfg.get("grad_accum_steps")
    if bsz and gacc:
        md.append(f"- **Batch Size:** {bsz} | **Grad Accumulation:** {gacc} (Effective batch: {bsz * gacc})")
    elif bsz:
        md.append(f"- **Batch Size:** {bsz}")
    md.append(f"- **AMP Enabled:** {cfg.get('amp', False)}")
    md.append(f"- **Gradient Checkpointing:** {cfg.get('use_checkpoint', False) or cfg.get('grad_checkpointing', False)}")
    md.append("")
    
    return "\n".join(md)

def generate_summary(run_dir: Path) -> None:
    data = parse_run(run_dir)
    md_content = make_markdown(data)
    summary_path = run_dir / "summary.md"
    summary_path.write_text(md_content + "\n")
    print(f"Generated summary for run {run_dir.name} -> {summary_path}")

def main():
    if len(sys.argv) > 1:
        # Run for a specific run directory passed as argument
        for arg in sys.argv[1:]:
            run_path = Path(arg).resolve()
            if run_path.is_dir():
                generate_summary(run_path)
            else:
                print(f"Directory not found: {arg}", file=sys.stderr)
    else:
        # Run for all directories in runs/
        runs_dir = Path(__file__).resolve().parent.parent / "runs"
        if not runs_dir.is_dir():
            print(f"Runs directory not found at {runs_dir}", file=sys.stderr)
            sys.exit(1)
            
        for child in runs_dir.iterdir():
            # Skip hidden files/folders and special non-run folders
            if child.is_dir() and not child.name.startswith(".") and child.name not in ("_summary", "archive", "protein_critic", "protein_critic_12L8H"):
                # Make sure it's a run folder (has checkpoints or logs or scores)
                if (child / "checkpoints").exists() or (child / "scores").exists() or (child / "logs").exists():
                    generate_summary(child)

if __name__ == "__main__":
    main()
