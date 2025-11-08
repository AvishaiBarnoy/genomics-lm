#!/usr/bin/env python3
"""
Training with:
- MPS autocast float16 (Apple Silicon)
- optional gradient checkpointing
- cosine or plateau LR schedules with early stopping
- CSV logging

Config highlights:
  amp: true/false
  use_checkpoint: true/false
  optimizer: "adamw" or "adafactor"
  scheduler: "cosine" (default) or "plateau"
  warmup_steps: linear warmup period
  min_lr: final LR for cosine schedule / floor for plateau
  early_stop_patience: epochs without improvement before stopping
  log_csv: where to append training curves (relative to scores_dir by default)
"""

import argparse, yaml, math, csv, time, torch, numpy as np, json, os, logging
from pathlib import Path
from typing import Optional, Tuple
from torch.utils.data import DataLoader, Dataset
from .model_tiny_gpt import TinyGPT

RUN_ID_ENV = "RUN_ID"


def _ensure_path_list(arg_value, cfg_value, key: str):
    source = arg_value if arg_value is not None else cfg_value
    if source is None:
        raise ValueError(f"Missing {key} specification (provide in config or CLI)")
    if isinstance(source, (str, os.PathLike)):
        return [str(source)]
    if isinstance(source, (list, tuple)):
        return [str(p) for p in source]
    raise TypeError(f"Unsupported {key} type: {type(source)}")


def _normalize_run_id(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    run_id = str(value).strip()
    return run_id or None


def _auto_run_id(cfg: dict, config_path: Optional[str]) -> str:
    try:
        from . import __package__  # noqa: F401
    except Exception:
        pass
    # Build RUN_ID like YYYY-MM-DD_<model>_<nL><nH>_d<n_embd>_e<epochs>
    from datetime import date
    from pathlib import Path
    today = date.today().strftime("%Y-%m-%d")
    tag = "run"
    if config_path:
        stem = Path(config_path).stem
        tag = stem.split("_", 1)[0] if "_" in stem else stem
    return f"{today}_{tag}_{int(cfg.get('n_layer',0))}L{int(cfg.get('n_head',0))}H_d{int(cfg.get('n_embd',0))}_e{int(cfg.get('epochs',0))}"


def _prepare_output_dirs(base_ckpt_dir: str, base_scores_dir: str, run_id: Optional[str]) -> Tuple[Path, Path]:
    ckpt_root = Path(base_ckpt_dir)
    if run_id:
        ckpt_root = ckpt_root / run_id
    ckpt_root.mkdir(parents=True, exist_ok=True)

    scores_root = Path(base_scores_dir)
    if run_id:
        scores_root = scores_root / run_id
    scores_root.mkdir(parents=True, exist_ok=True)
    return ckpt_root, scores_root


class PackedDataset(Dataset):
    def __init__(self, paths):
        if isinstance(paths, (str, os.PathLike)):
            paths = [paths]
        else:
            paths = list(paths)
        totals = []
        tail_shape = None
        y_tail_shape = None
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                X = data["X"]
                Y = data["Y"]
                if tail_shape is None:
                    tail_shape = X.shape[1:]
                    y_tail_shape = Y.shape[1:]
                totals.append(X.shape[0])
        total_rows = sum(totals)
        if total_rows == 0:
            self.X = torch.empty((0,) + (tail_shape or (0,)), dtype=torch.long)
            self.Y = torch.empty((0,) + (y_tail_shape or (0,)), dtype=torch.long)
            return

        X_agg = np.empty((total_rows,) + tail_shape, dtype=np.int64)
        Y_agg = np.empty((total_rows,) + y_tail_shape, dtype=np.int64)

        offset = 0
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                X = np.asarray(data["X"], dtype=np.int64)
                Y = np.asarray(data["Y"], dtype=np.int64)
                rows = X.shape[0]
                X_agg[offset:offset + rows] = X
                Y_agg[offset:offset + rows] = Y
                offset += rows
        self.X = torch.from_numpy(X_agg)
        self.Y = torch.from_numpy(Y_agg)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

def dev():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_id", default=None, help=f"Unique run id; falls back to ${RUN_ID_ENV} or config.run_id")
    ap.add_argument("--resume", default=None, help="Path to checkpoint to resume training from")
    ap.add_argument("--train_npz", action="append", default=None, help="Training NPZ file (repeatable)")
    ap.add_argument("--val_npz", action="append", default=None, help="Validation NPZ file (repeatable)")
    ap.add_argument("--test_npz", action="append", default=None, help="Test NPZ file (repeatable)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    resume_path = args.resume or cfg.pop("resume", None)
    if resume_path is not None:
        resume_path = str(resume_path)

    device = dev()
    torch.manual_seed(cfg.get("seed", 1337))
    amp = bool(cfg.get("amp", True)) and (device.type == "mps")

    default_train = f"data/processed/train_bs{cfg['block_size']}.npz"
    default_val = f"data/processed/val_bs{cfg['block_size']}.npz"
    default_test = f"data/processed/test_bs{cfg['block_size']}.npz"
    cfg.setdefault("train_npz", default_train)
    cfg.setdefault("val_npz", default_val)
    cfg.setdefault("test_npz", default_test)

    train_paths = _ensure_path_list(args.train_npz, cfg.get("train_npz"), "train_npz")
    val_paths = _ensure_path_list(args.val_npz, cfg.get("val_npz"), "val_npz")
    test_paths = _ensure_path_list(args.test_npz, cfg.get("test_npz"), "test_npz")
    cfg["train_npz"] = train_paths
    cfg["val_npz"] = val_paths
    cfg["test_npz"] = test_paths

    if resume_path and not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    matmul_precision = cfg.get("matmul_precision")
    if matmul_precision:
        setter = getattr(torch, "set_float32_matmul_precision", None)
        if callable(setter):
            try:
                setter(str(matmul_precision))
                print(f"[matmul] float32 precision set to {matmul_precision}")
            except Exception as exc:
                print(f"[matmul] failed to set precision '{matmul_precision}': {exc}")
        else:
            print("[matmul] torch.set_float32_matmul_precision unavailable in this build.")

    # Optional: derive embedding size from per-head dimension
    if "d_head" in cfg and cfg.get("n_head"):
        try:
            cfg["n_embd"] = int(cfg["d_head"]) * int(cfg["n_head"])
            print(f"[dims] using d_head={cfg['d_head']} × n_head={cfg['n_head']} → n_embd={cfg['n_embd']}")
        except Exception as exc:
            print(f"[dims] failed to derive n_embd from d_head: {exc}")

    train_ds = PackedDataset(train_paths)
    val_ds = PackedDataset(val_paths)
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", False))
    prefetch_factor = int(cfg.get("prefetch_factor", 2)) if num_workers > 0 else None
    persistent_workers = bool(cfg.get("persistent_workers", True)) if num_workers > 0 else False
    dl_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)
    if prefetch_factor is not None:
        dl_kwargs["prefetch_factor"] = prefetch_factor
    if persistent_workers:
        dl_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, **dl_kwargs)
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"], **dl_kwargs)

    sep_mask_enabled = bool(cfg.get("sep_mask_enabled", True))
    model = TinyGPT(
        cfg["vocab_size"],
        cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg["dropout"],
        use_checkpoint=bool(cfg.get("use_checkpoint", cfg.get("grad_checkpointing", False))),
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
        sep_id=(3 if sep_mask_enabled else None),
        tie_embeddings=bool(cfg.get("tie_embeddings", True)),
        n_kv_head=int(cfg.get("n_kv_head")) if cfg.get("n_kv_head") is not None else None,
        use_sdpa=bool(cfg.get("use_sdpa", False)),
    ).to(device)

    compile_requested = bool(cfg.get("compile", False))
    compile_mode = cfg.get("compile_mode", "default")

    if compile_requested:
        # Be resilient: if Dynamo hits a backend quirk, fall back to eager without crashing
        try:
            import torch._dynamo as dynamo
            dynamo.config.suppress_errors = True
            try:
                dynamo.config.log_level = logging.ERROR
            except Exception:
                pass
        except Exception:
            pass
        # Compatibility shim for older transformers packages that break Dynamo's ModelOutput probe
        try:
            import importlib
            fu = importlib.import_module("transformers.file_utils")
            if not hasattr(fu, "ModelOutput"):
                utils_mod = importlib.import_module("transformers.utils")
                if hasattr(utils_mod, "ModelOutput"):
                    setattr(fu, "ModelOutput", getattr(utils_mod, "ModelOutput"))
        except Exception:
            # transformers not installed or different layout; ignore
            pass
        torch_compile = getattr(torch, "compile", None)
        if torch_compile:
            try:
                model = torch_compile(model, mode=compile_mode)
                print(f"[compile] torch.compile enabled (mode={compile_mode})")
                # Probe whether a graph was actually captured; if not, we likely fell back to eager.
                try:
                    from torch._dynamo.utils import counters as _dynamo_counters  # type: ignore
                    before_ok = int(_dynamo_counters["frames"].get("ok", 0)) if isinstance(_dynamo_counters, dict) else 0
                    probe_T = max(1, min(8, int(cfg.get("block_size", 8))))
                    with torch.no_grad():
                        _ = model(torch.zeros((1, probe_T), dtype=torch.long, device=device))
                    after_ok = int(_dynamo_counters["frames"].get("ok", 0)) if isinstance(_dynamo_counters, dict) else before_ok
                    captured = max(0, after_ok - before_ok)
                    if captured == 0:
                        print("[compile] no graphs captured; running in eager (fallback).")
                    else:
                        print(f"[compile] graphs_captured={captured}")
                except Exception:
                    # If counters are unavailable or the probe fails, stay silent
                    pass
            except Exception as exc:
                print(f"[compile] torch.compile failed ({exc}); continuing without compilation.")
        else:
            print("[compile] torch.compile not available in this PyTorch build.")

    # Optimizer selection (lazy import to avoid importing transformers unless needed)
    if cfg.get("optimizer", "adamw").lower() == "adafactor":
        try:
            from transformers.optimization import Adafactor  # type: ignore
        except Exception:
            raise RuntimeError("transformers not installed; pip install transformers to use Adafactor")
        optim = Adafactor(
            model.parameters(),
            lr=cfg.get("lr", 3e-4),
            scale_parameter=False,
            relative_step=False,
            weight_decay=cfg.get("weight_decay", 0.05),
        )
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    scheduler_name = str(cfg.get("scheduler", "cosine")).lower()
    if scheduler_name not in {"cosine", "plateau"}:
        print(f"[warn] Unknown scheduler '{scheduler_name}', defaulting to cosine.")
        scheduler_name = "cosine"

    gacc = cfg.get("grad_accum_steps", 16)
    warmup_steps = int(cfg.get("warmup_steps", 200))
    min_lr = float(cfg.get("min_lr", 1e-5))
    base_lr = float(cfg["lr"])

    # Auto-epochs: target total tokens = tokens_per_param * n_params (heuristic T = 20 * N)
    # tokens_per_epoch ≈ len(train_ds) * block_size
    epochs_cfg = cfg.get("epochs", 5)
    n_params = sum(p.numel() for p in model.parameters())
    tokens_per_param = float(cfg.get("tokens_per_param", 20.0))
    if isinstance(epochs_cfg, str) and epochs_cfg.strip().lower() == "auto":
        tokens_target = max(1.0, tokens_per_param * float(n_params))
        tokens_per_epoch = max(1.0, float(len(train_ds) * cfg["block_size"]))
        est_epochs = int(math.ceil(tokens_target / tokens_per_epoch))
        # clamp to reasonable bounds
        est_epochs = max(int(cfg.get("epochs_min", 1)), min(est_epochs, int(cfg.get("epochs_max", max(1, est_epochs)))))
        max_epochs = est_epochs
        print(
            f"[epochs-auto] tokens_per_param={tokens_per_param} n_params={n_params} → target_tokens={int(tokens_target)}; "
            f"tokens_per_epoch≈{int(tokens_per_epoch)} → epochs={max_epochs}"
        )
    else:
        max_epochs = int(epochs_cfg)
    steps_per_epoch = math.ceil(len(train_loader) / max(1, gacc))
    total_steps = max(1, steps_per_epoch * max_epochs)
    use_cosine = scheduler_name == "cosine"
    if use_cosine:
        warmup_for_lambda = max(1, warmup_steps)
        min_lr_ratio = (min_lr / base_lr) if base_lr > 0 else 0.0

        def lr_lambda(step_idx: int) -> float:
            if step_idx < warmup_for_lambda:
                return float(step_idx + 1) / warmup_for_lambda
            progress = (step_idx - warmup_for_lambda) / max(1, total_steps - warmup_for_lambda)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode="min",
            factor=0.5,
            patience=cfg.get("plateau_patience", 2),
            min_lr=min_lr,
        )

    run_id = _normalize_run_id(args.run_id or cfg.get("run_id") or os.environ.get(RUN_ID_ENV))
    if not run_id:
        run_id = _auto_run_id(cfg, args.config)
    if run_id:
        cfg["run_id"] = run_id
    outdir = cfg["out_dir"]
    scores_base = cfg.get("scores_dir", "outputs/scores")
    ckpt_dir, scores_dir = _prepare_output_dirs(outdir, scores_base, run_id)
    log_csv_cfg = cfg.get("log_csv")
    if log_csv_cfg:
        log_csv_path = Path(log_csv_cfg)
        log_csv = (scores_dir / log_csv_path).resolve() if not log_csv_path.is_absolute() else log_csv_path
    else:
        log_csv = scores_dir / "curves.csv"
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    with log_csv.open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["step","train_loss","val_loss","perplexity","lr"])

    start_epoch = 0
    best = float("inf"); no_improve = 0
    step = 0
    best_epoch = None

    if resume_path:
        print(f"[resume] loading {resume_path}")
        ckpt_resume = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt_resume["model"])
        if "optimizer" in ckpt_resume:
            try:
                optim.load_state_dict(ckpt_resume["optimizer"])
            except Exception as exc:
                print(f"[resume] optimizer state load failed: {exc}")
        if "scheduler" in ckpt_resume and ckpt_resume["scheduler"] is not None:
            try:
                scheduler.load_state_dict(ckpt_resume["scheduler"])
            except Exception as exc:
                print(f"[resume] scheduler state load failed: {exc}")
        start_epoch = int(ckpt_resume.get("epoch", 0))
        step = int(ckpt_resume.get("step", step))
        best = float(ckpt_resume.get("best_val", best))
        best_epoch = ckpt_resume.get("best_epoch", best_epoch)
        no_improve = int(ckpt_resume.get("no_improve", no_improve))

    if start_epoch >= max_epochs:
        print(f"[resume] start_epoch {start_epoch} >= configured epochs {max_epochs}; no new epochs will run unless you increase 'epochs'.")

    scaler = None  # GradScaler is CUDA-only; autocast on MPS may be unsupported in some torch versions

    def one_pass(split, loader):
        nonlocal step
        mps_autocast_ok = True
        model.train(split=="train")
        total, n = 0.0, 0
        optim.zero_grad(set_to_none=True)
        skipped = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            def fwd():
                logits_, loss_ = model(xb, yb)
                return logits_, loss_ / gacc

            if amp and mps_autocast_ok:
                try:
                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                        logits, loss = fwd()
                except RuntimeError as e:
                    msg = str(e).lower()
                    if "unsupported autocast device_type" in msg or "autocast" in msg and device.type == "mps":
                        # Fallback: disable MPS autocast for this run
                        mps_autocast_ok = False
                        logits, loss = fwd()
                    else:
                        raise
            else:
                logits, loss = fwd()
            # Guard against NaNs/infs (e.g., degenerate batches); skip update for this micro-step
            if not torch.isfinite(loss):
                skipped += 1
                continue
            if split=="train":
                loss.backward()
                if (n+1) % gacc == 0:
                    if (not use_cosine) and warmup_steps > 0 and step < warmup_steps:
                        scale = float(step + 1) / max(1, warmup_steps)
                        for pg in optim.param_groups:
                            pg["lr"] = base_lr * scale
                    optim.step()
                    optim.zero_grad(set_to_none=True)
                    step += 1
                    if use_cosine:
                        scheduler.step()
            total += loss.item()*gacc; n += 1
        return (total / max(n,1), skipped)

    history = []
    best_epoch = None
    if run_id:
        print(f"[run] id={run_id}")
    print(f"[paths] ckpts={ckpt_dir} scores={scores_dir} log_csv={log_csv}")
    print(f"[model] params={n_params} sep_mask_enabled={sep_mask_enabled}")
    print(f"[loader] num_workers={num_workers} pin_memory={pin_memory} prefetch_factor={prefetch_factor} persistent_workers={persistent_workers}")
    print(
        f"[train] starting: epochs={max_epochs}, steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, "
        f"batch_size={cfg['batch_size']}, grad_accum={gacc}, scheduler={scheduler_name}"
    )

    train_wall0 = time.perf_counter()
    train_cpu0 = time.process_time()

    for epoch in range(start_epoch, max_epochs):
        ep_wall0 = time.perf_counter()
        ep_cpu0 = time.process_time()
        epoch_idx = epoch + 1
        train_loss, train_skips = one_pass("train", train_loader)
        with torch.no_grad():
            val_loss, val_skips = one_pass("val", val_loader)
        ppl = math.exp(min(20.0, val_loss))
        if not use_cosine:
            scheduler.step(val_loss)
        lr_now = optim.param_groups[0]["lr"]
        msg = f"[epoch {epoch_idx}] train {train_loss:.3f} | val {val_loss:.3f} | ppl {ppl:.2f} | lr {lr_now:.2e}"
        if train_skips or val_skips:
            msg += f" | skips train={train_skips} val={val_skips}"
        print(msg)
        ep_wall1 = time.perf_counter()
        ep_cpu1 = time.process_time()
        print(f"[timing] epoch {epoch_idx} wall_sec={ep_wall1-ep_wall0:.2f} cpu_sec={ep_cpu1-ep_cpu0:.2f}")

        improved = val_loss + 1e-6 < best
        if improved:
            best = val_loss
            best_epoch = epoch_idx
            no_improve = 0
        else:
            no_improve += 1

        ckpt_payload = {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "cfg": cfg,
            "epoch": epoch_idx,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "best_val": best,
            "best_epoch": best_epoch,
            "no_improve": no_improve,
            "step": step,
        }
        torch.save(ckpt_payload, ckpt_dir / "last.pt")
        with log_csv.open("a", newline="") as f:
            csv.writer(f).writerow([epoch_idx, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{ppl:.3f}", f"{lr_now:.3e}"])

        history.append({
            "epoch": epoch_idx,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "perplexity": ppl,
            "lr": lr_now,
        })

        if improved:
            torch.save(ckpt_payload, ckpt_dir / "best.pt")
        elif no_improve >= int(cfg.get("early_stop_patience", 5)):
            print("[early-stopping] no improvement; stopping.")
            break

    if history:
        metrics = {
            "run_id": run_id,
            "best_epoch": best_epoch,
            "best_val_loss": best if best != float("inf") else None,
            "last_epoch": history[-1]["epoch"],
            "last_val_loss": history[-1]["val_loss"],
            "last_train_loss": history[-1]["train_loss"],
            "last_perplexity": history[-1]["perplexity"],
        }
        metrics_path = scores_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    # Overall training timing (wall vs CPU time)
    train_wall1 = time.perf_counter()
    train_cpu1 = time.process_time()
    print(f"[timing] train_wall_sec={train_wall1-train_wall0:.2f} train_cpu_sec={train_cpu1-train_cpu0:.2f}")

if __name__ == "__main__":
    main()
