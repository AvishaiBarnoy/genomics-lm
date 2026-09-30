import os
import sys
import math
import csv
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.codonlm.model_tiny_gpt import TinyGPT
from src.codonlm.dataset_manifest import (
    discover_manifest,
    load_dataset_manifest,
    manifest_artifact_path,
)
from src.codonlm.data_loading import (
    build_codon_lm_dataloaders,
    build_codon_lm_datasets,
    dataset_length_audit,
)
from src.codonlm.replay import GeneratedTerminationReplayDataset
from src.training.runtime import (
    PeriodicCheckpointPolicy,
    WallTimer,
    default_device,
)
from src.training.engine import EngineConfig, TrainingEngine
from src.training.strategies import (
    AccumulatedBackpropStrategy,
    PrecisionPolicy,
)
from src.training.errors import NonFiniteGroupLimitError
from src.training.run_lifecycle import (
    TrainingRun,
    configuration_fingerprint,
)

from src.codonlm.training.config import (
    write_meta,
    _ensure_path_list,
    _normalize_run_id,
    _auto_run_id,
    _normalize_offset_weights,
)
from src.codonlm.training.checkpoint import _read_itos, _load_transfer_state_dict
from src.codonlm.training.vocabulary import (
    resolve_vocabulary_contract,
    snapshot_vocabulary,
    validate_resume_checkpoint,
    write_vocabulary_manifest,
)
from src.codonlm.training.task import (
    CodonLMTask,
    decode_codon_lm_checkpoint,
    make_codon_lm_checkpoint_adapter,
)
from src.codonlm.training.callbacks import CodonLMConsole

RUN_ID_ENV = "RUN_ID"
PAD_ID = 0


NonfiniteGroupLimitError = NonFiniteGroupLimitError


def resolve_warmup_steps(cfg: dict, total_steps: int) -> int:
    """Resolve a fixed or scheduler-relative warmup without ambiguous precedence."""
    if total_steps <= 0:
        raise ValueError("scheduler_total_steps must be positive")
    fraction = cfg.get("warmup_fraction")
    if fraction is None:
        steps = int(cfg.get("warmup_steps", 200))
        if steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        return steps
    if "warmup_steps" in cfg:
        raise ValueError("configure only one of warmup_steps or warmup_fraction")
    fraction = float(fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if fraction == 0.0:
        return 0
    return max(1, int(round(total_steps * fraction)))


@dataclass
class AccumulationHealth:
    """Checkpointable counters for gradient-accumulation group integrity."""

    active_microbatches: int = 0
    nonfinite_microbatches: int = 0
    aborted_groups: int = 0
    discarded_finite_microbatches: int = 0

    def record_finite_microbatch(self) -> None:
        self.active_microbatches += 1

    def complete_group(self) -> None:
        if self.active_microbatches <= 0:
            raise ValueError("cannot complete an empty accumulation group")
        self.active_microbatches = 0

    def abort_group(self, optimizer) -> int:
        discarded = self.active_microbatches
        optimizer.zero_grad(set_to_none=True)
        self.nonfinite_microbatches += 1
        self.aborted_groups += 1
        self.discarded_finite_microbatches += discarded
        self.active_microbatches = 0
        return discarded

    def exceeds_limit(self, max_aborted_groups: int) -> bool:
        if max_aborted_groups < 0:
            return False
        return self.aborted_groups > max_aborted_groups

    def state_dict(self) -> dict[str, int]:
        state = self.metrics_dict()
        # Gradients are not checkpointed; resume replays from the last resolved group.
        state["active_microbatches"] = 0
        return state

    def metrics_dict(self) -> dict[str, int]:
        return {
            "active_microbatches": self.active_microbatches,
            "nonfinite_microbatches": self.nonfinite_microbatches,
            "aborted_groups": self.aborted_groups,
            "discarded_finite_microbatches": self.discarded_finite_microbatches,
        }

    def load_state_dict(self, state: dict | None) -> None:
        state = state or {}
        self.active_microbatches = 0
        self.nonfinite_microbatches = int(state.get("nonfinite_microbatches", 0))
        self.aborted_groups = int(state.get("aborted_groups", 0))
        self.discarded_finite_microbatches = int(
            state.get("discarded_finite_microbatches", 0)
        )


def _average_accumulated_gradients(parameters, microbatch_count: int) -> None:
    if microbatch_count <= 0:
        raise ValueError("microbatch_count must be positive")
    for param in parameters:
        if param.grad is not None:
            param.grad.div_(microbatch_count)

def dev(force_gpu: bool = False, requested: str = "auto"):
    requested = str(requested or "auto").lower()
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"unsupported device {requested!r}; expected auto, cpu, mps, or cuda")
    if requested == "cpu":
        device = torch.device("cpu")
    elif requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("requested device=mps but MPS is not available")
        device = torch.device("mps")
    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested device=cuda but CUDA is not available")
        device = torch.device("cuda")
    else:
        device = default_device()
    if force_gpu and device.type == "cpu":
        raise RuntimeError("force_gpu=true but no CUDA or MPS device is available")
    return device


def run_training(cfg: dict, args) -> None:
    primary_contract = cfg.get("primary_training_contract")
    if primary_contract is not None:
        from src.codonlm.training.primary_contract import (
            validate_primary_training_config,
        )

        contract_result = validate_primary_training_config(cfg)
        requested_run_id = _normalize_run_id(args.run_id)
        if requested_run_id and requested_run_id != contract_result["run_id"]:
            raise ValueError(
                "--run_id cannot override an immutable primary training config"
            )
        for name in ("train_npz", "val_npz", "test_npz", "transfer_from"):
            if getattr(args, name, None) is not None:
                raise ValueError(
                    f"--{name} cannot override an immutable primary training config"
                )
        print(
            "[contract] corrected primary config verified "
            f"role={contract_result['role']} protocol={contract_result['protocol']} "
            f"seed={contract_result['seed']}"
        )

    resume_path = args.resume or cfg.pop("resume", None)
    if resume_path is not None:
        resume_path = str(resume_path)
    fork_from = getattr(args, "fork_from", None) or cfg.pop("fork_from", None)
    if fork_from is not None:
        fork_from = str(fork_from)

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

    manifest_value = cfg.get("dataset_manifest")
    if isinstance(manifest_value, dict):
        manifest_value = manifest_value.get("path")
    manifest_path = (
        Path(manifest_value).expanduser().resolve()
        if manifest_value
        else discover_manifest([*train_paths, *val_paths, *test_paths])
    )
    if manifest_path is not None:
        dataset_manifest = load_dataset_manifest(manifest_path)
        for split, selected in (
            ("train", train_paths), ("val", val_paths), ("test", test_paths)
        ):
            declared = manifest_artifact_path(
                dataset_manifest, manifest_path, f"{split}_tokens"
            ).resolve()
            resolved_selected = [Path(path).expanduser().resolve() for path in selected]
            if resolved_selected != [declared]:
                raise ValueError(
                    f"{split} dataset paths do not match manifest {manifest_path}: "
                    f"selected={resolved_selected}, declared={declared}"
                )
        cfg["dataset_manifest"] = {
            "path": str(manifest_path),
            "dataset_id": dataset_manifest["dataset"]["id"],
            "scientific_valid": dataset_manifest["dataset"]["scientific_valid"],
            "schema": dataset_manifest["schema"],
        }
        current_dataset_id = dataset_manifest["dataset"]["id"]
    else:
        cfg["dataset_manifest"] = {
            "status": "legacy_unverified",
            "scientific_valid": False,
        }
        current_dataset_id = None

    configured_vocab_size = cfg.get("vocab_size")
    vocabulary_contract = resolve_vocabulary_contract(
        [*train_paths, *val_paths, *test_paths],
        configured_path=cfg.get("itos_path"),
        configured_size=configured_vocab_size,
    )
    cfg["vocab_size"] = vocabulary_contract.size

    if resume_path and not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    if resume_path:
        validate_resume_checkpoint(
            resume_path, vocabulary_contract, dataset_id=current_dataset_id
        )
    if fork_from:
        validate_resume_checkpoint(
            fork_from, vocabulary_contract, dataset_id=current_dataset_id
        )

    transfer_path = (
        None
        if resume_path or fork_from
        else (args.transfer_from or cfg.pop("transfer_from", None))
    )
    if transfer_path and not os.path.isfile(transfer_path):
        raise FileNotFoundError(f"Transfer weights not found: {transfer_path}")

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

    if "d_head" in cfg and cfg.get("n_head"):
        try:
            cfg["n_embd"] = int(cfg["d_head"]) * int(cfg["n_head"])
            print(f"[dims] using d_head={cfg['d_head']} × n_head={cfg['n_head']} → n_embd={cfg['n_embd']}")
        except Exception as exc:
            print(f"[dims] failed to derive n_embd from d_head: {exc}")

    base_seed = int(cfg.get("seed", 1337))
    use_mmap = bool(cfg.get("use_mmap", False))
    train_ds, val_ds = build_codon_lm_datasets(train_paths, val_paths, use_mmap=use_mmap)
    if use_mmap:
        train_storage = getattr(train_ds, "storage_mode", "unknown")
        val_storage = getattr(val_ds, "storage_mode", "unknown")
        print(f"[loader] storage train={train_storage} val={val_storage}")
        if train_storage != "npy_mmap" or val_storage != "npy_mmap":
            print(
                "[loader] warning: use_mmap=true requires adjacent *_X.npy and "
                "*_lengths.npy/*_Y.npy sidecars; compressed NPZ data is loaded into memory."
            )
    train_audit = dataset_length_audit(train_ds, int(cfg["block_size"]))
    val_audit = dataset_length_audit(val_ds, int(cfg["block_size"]))
    cfg["dataset_audit"] = {"train": train_audit, "val": val_audit}
    cfg["whole_gene_status"] = (
        "whole-or-chunked"
        if train_audit["at_block_size"] or val_audit["at_block_size"]
        else "whole-under-block-size"
    )
    print(f"[audit] train_lengths={train_audit}")
    print(f"[audit] val_lengths={val_audit}")

    def _loader_cfg_for_epoch(epoch_idx: int) -> dict:
        loader_cfg = dict(cfg)
        loader_cfg["dataloader_seed"] = base_seed + max(0, int(epoch_idx))
        return loader_cfg

    try:
        train_loader, val_loader, train_sampler, dl_kwargs = build_codon_lm_dataloaders(
            train_ds,
            val_ds,
            _loader_cfg_for_epoch(0),
        )
        if train_sampler is not None:
            print(
                f"[loader] BucketBatchSampler: {cfg.get('n_buckets', 8)} buckets, "
                f"{len(train_sampler)} batches, batch_size={cfg['batch_size']}"
            )
    except Exception as exc:
        raise RuntimeError(f"failed to build CodonLM dataloaders: {exc}") from exc

    sep_mask_enabled = bool(cfg.get("sep_mask_enabled", True))
    multi_offset_enabled = bool(cfg.get("multi_offset_loss_enabled", False))
    multi_offset_targets = [int(x) for x in cfg.get("multi_offset_targets", [])]
    multi_offset_weights = (
        _normalize_offset_weights(multi_offset_targets, cfg.get("multi_offset_weights"))
        if multi_offset_enabled
        else {}
    )
    if multi_offset_weights:
        print(f"[loss] multi_offset_weights={multi_offset_weights}")
    termination_loss_enabled = bool(cfg.get("termination_loss_enabled", False))
    termination_loss_weight = float(cfg.get("termination_loss_weight", 0.1))
    termination_stop_ids = tuple(int(x) for x in cfg.get("termination_stop_ids", [2]))
    termination_bucket_edges = tuple(int(x) for x in cfg.get("termination_bucket_edges", [0, 3, 10, 30]))
    termination_n_classes = int(cfg.get("termination_n_classes", len(termination_bucket_edges) + 1))
    if termination_n_classes != len(termination_bucket_edges) + 1:
        raise ValueError("termination_n_classes must equal len(termination_bucket_edges) + 1")
    termination_class_weights_raw = cfg.get("termination_class_weights")
    termination_class_weight_values = None
    if termination_class_weights_raw is not None:
        if len(termination_class_weights_raw) != termination_n_classes:
            raise ValueError(
                "termination_class_weights must contain termination_n_classes values"
            )
        termination_class_weight_values = [
            float(value) for value in termination_class_weights_raw
        ]
        if any(value <= 0 for value in termination_class_weight_values):
            raise ValueError("termination_class_weights values must be positive")
    replay_loss_enabled = bool(cfg.get("replay_loss_enabled", False))
    replay_loss_weight = float(cfg.get("replay_loss_weight", 0.1))
    replay_data = cfg.get("replay_data")
    replay_batch_size = int(cfg.get("replay_batch_size", cfg.get("batch_size", 1)))
    replay_every_microbatches = int(cfg.get("replay_every_microbatches", 1))
    if replay_every_microbatches <= 0:
        raise ValueError("replay_every_microbatches must be positive")
    replay_class_weights_raw = cfg.get("replay_class_weights")
    replay_class_weight_values = None
    if replay_class_weights_raw is not None:
        if len(replay_class_weights_raw) != termination_n_classes:
            raise ValueError(
                "replay_class_weights must contain termination_n_classes values"
            )
        replay_class_weight_values = [
            float(value) for value in replay_class_weights_raw
        ]
        if any(value <= 0 for value in replay_class_weight_values):
            raise ValueError("replay_class_weights values must be positive")
    termination_head_enabled = termination_loss_enabled or replay_loss_enabled
    if termination_loss_enabled:
        print(
            f"[loss] termination_aux weight={termination_loss_weight} "
            f"stop_ids={termination_stop_ids} bucket_edges={termination_bucket_edges} "
            f"class_weights={termination_class_weights_raw}"
        )
    if replay_loss_enabled:
        if not replay_data:
            raise ValueError("replay_loss_enabled=true requires replay_data")
        for checkpoint_path in (resume_path, fork_from):
            if checkpoint_path is None:
                continue
            replay_checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            replay_state = replay_checkpoint.get("task", {}).get("replay")
            if replay_state is None:
                raise ValueError(
                    "Replay-enabled resume/fork requires a shared-engine checkpoint "
                    "with replay iterator state; legacy checkpoints cannot resume "
                    "this objective exactly."
                )
        print(
            f"[loss] replay_termination weight={replay_loss_weight} "
            f"data={replay_data} batch_size={replay_batch_size} "
            f"every_microbatches={replay_every_microbatches} "
            f"class_weights={replay_class_weights_raw}"
        )

    eos_loss_weight = cfg.get("eos_loss_weight", None)
    loss_weights = None
    if eos_loss_weight is not None and float(eos_loss_weight) != 1.0:
        from src.codonlm.codon_tokenize import stoi, STOP_CODONS
        loss_weights = [1.0] * cfg["vocab_size"]
        loss_weights[stoi["<EOS_CDS>"]] = float(eos_loss_weight)
        for codon in STOP_CODONS:
            if codon in stoi:
                loss_weights[stoi[codon]] = float(eos_loss_weight)
        print(f"[weights] upweighting termination tokens by {eos_loss_weight}x")

    run_id = _normalize_run_id(args.run_id or cfg.get("run_id") or os.environ.get(RUN_ID_ENV))
    if not run_id:
        run_id = _auto_run_id(cfg, args.config)
    if run_id:
        cfg["run_id"] = run_id
    configured_epochs = cfg.get("epochs")
    target_epochs = int(configured_epochs) if configured_epochs is not None else None
    run_fingerprint = configuration_fingerprint(cfg)
    training_run = TrainingRun.open(
        "runs",
        run_id,
        resume=resume_path,
        fork_from=fork_from,
        last_checkpoint_name="last.pt",
        target_epochs=target_epochs,
        config_fingerprint=run_fingerprint,
    )
    run_id = training_run.run_dir.name
    cfg["run_id"] = run_id
    ckpt_dir, scores_dir = training_run.checkpoints, training_run.scores
    accumulation_health = AccumulationHealth()

    vocabulary_snapshot = snapshot_vocabulary(
        vocabulary_contract, ckpt_dir.parent / "itos.txt"
    )
    vocabulary_provenance = vocabulary_contract.provenance(vocabulary_snapshot)
    cfg["itos_path"] = str(vocabulary_snapshot)
    cfg["vocabulary"] = vocabulary_provenance
    write_vocabulary_manifest(
        vocabulary_provenance, ckpt_dir.parent / "vocabulary.json"
    )

    shutil.copy2(args.config, ckpt_dir / "config.yaml")
    training_run.start_logging()

    def write_failure_meta(exc: Exception) -> None:
        meta = {
            "run_id": run_id,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "accumulation_health": accumulation_health.metrics_dict(),
            "model_spec": {},
        }
        write_meta(ckpt_dir, meta)

    log_csv_cfg = cfg.get("log_csv")
    if log_csv_cfg:
        log_csv_path = Path(log_csv_cfg)
        log_csv = (scores_dir / log_csv_path).resolve() if not log_csv_path.is_absolute() else log_csv_path
    else:
        log_csv = scores_dir / "curves.csv"
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    # Check if we should append to preserve history when resuming
    is_resume = resume_path is not None and log_csv.exists()
    open_mode = "a" if is_resume else "w"
    
    with log_csv.open(open_mode, newline="") as f:
        if not is_resume:
            writer = csv.writer(f)
            offset_cols = []
            for offset in sorted(multi_offset_weights):
                offset_cols.extend([f"train_offset_{offset}", f"val_offset_{offset}"])
            term_cols = ["train_term_loss", "val_term_loss"] if termination_loss_enabled else []
            replay_cols = ["train_replay_term_loss"] if replay_loss_enabled else []
            writer.writerow([
                "step",
                "train_loss",
                "val_loss",
                "train_next_loss",
                "val_next_loss",
                "perplexity",
                "lr",
                *offset_cols,
                *term_cols,
                *replay_cols,
            ])

    try:
        device = dev(
            force_gpu=bool(cfg.get("force_gpu", False)),
            requested=str(cfg.get("device", "auto")),
        )
    except Exception as exc:
        print(f"[error] training failed: {exc}", file=sys.stderr)
        write_failure_meta(exc)
        raise

    cfg["device"] = str(device)
    print(f"[device] using {device}")
    torch.manual_seed(base_seed)
    amp = bool(cfg.get("amp", True)) and (device.type == "mps")
    termination_class_weights = (
        torch.tensor(
            termination_class_weight_values,
            dtype=torch.float32,
            device=device,
        )
        if termination_class_weight_values is not None
        else None
    )
    replay_class_weights = (
        torch.tensor(
            replay_class_weight_values,
            dtype=torch.float32,
            device=device,
        )
        if replay_class_weight_values is not None
        else None
    )

    use_shape_guidance = bool(cfg.get("use_shape_guidance", False))
    encoder = None
    lookup_table = None
    if use_shape_guidance:
        from src.codonlm.biophysics import NucleotideEncoder, generate_shape_training_data, load_nucleotide_encoder_state
        from scripts.train_biophysics_fusion import build_one_hot_lookup
        
        encoder = NucleotideEncoder(d_shape=3).to(device)
        enc_ckpt = Path(cfg.get("biophysics_encoder_checkpoint", "runs/biophysics_encoder.pt"))
        if enc_ckpt.exists():
            print(f"[biophysics] Loading pre-trained encoder from {enc_ckpt}")
            encoder.load_state_dict(load_nucleotide_encoder_state(enc_ckpt))
        else:
            print(f"[biophysics] {enc_ckpt} not found. Pre-training on-the-fly...")
            train_x, train_y = generate_shape_training_data(num_samples=5000, seq_len_codons=60)
            optimizer_enc = torch.optim.AdamW(encoder.parameters(), lr=0.005)
            criterion_enc = nn.MSELoss()
            encoder.train()
            for _ in range(5):
                for i in range(0, len(train_x), 64):
                    bx = train_x[i : i + 64].to(device)
                    by = train_y[i : i + 64].to(device)
                    optimizer_enc.zero_grad()
                    pred = encoder(bx)
                    loss = criterion_enc(pred, by)
                    loss.backward()
                    optimizer_enc.step()
            print("[biophysics] On-the-fly encoder pre-training completed.")
            
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
            
        itos_file = Path(cfg.get("itos_path", "itos.txt"))
        if itos_file.exists():
            itos = [line.strip() for line in itos_file.read_text().splitlines() if line.strip()]
        else:
            from src.codonlm.codon_tokenize import itos as CODON_ITOS
            itos = CODON_ITOS
        lookup_table = build_one_hot_lookup(itos, device)

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
        loss_weights=loss_weights,
        termination_aux=termination_head_enabled,
        termination_n_classes=termination_n_classes,
        multi_offset_targets=multi_offset_targets if multi_offset_enabled else None,
        use_swiglu=bool(cfg.get("use_swiglu", False)),
        use_rope=bool(cfg.get("use_rope", False)),
        use_shape_guidance=use_shape_guidance,
    ).to(device)
    if model.tok_emb.num_embeddings != vocabulary_contract.size:
        raise RuntimeError("model token embedding rows do not match resolved vocabulary")
    if model.head.out_features != vocabulary_contract.size:
        raise RuntimeError("model output rows do not match resolved vocabulary")

    replay_loader = None
    if replay_loss_enabled:
        replay_path = Path(str(replay_data))
        if not replay_path.is_absolute():
            replay_path = Path.cwd() / replay_path
        replay_ds = GeneratedTerminationReplayDataset(
            replay_path,
            block_size=int(cfg["block_size"]),
            pad_id=PAD_ID,
        )
        replay_generator = torch.Generator()
        replay_generator.manual_seed(base_seed + 17)
        replay_loader = DataLoader(
            replay_ds,
            batch_size=max(1, replay_batch_size),
            shuffle=True,
            num_workers=0,
            drop_last=False,
            generator=replay_generator,
        )
        cfg["replay_data"] = str(replay_path)
        cfg["replay_examples"] = int(len(replay_ds))
        print(f"[replay] loaded {len(replay_ds)} generated-state records from {replay_path}")

    compile_requested = bool(cfg.get("compile", False))
    compile_mode = cfg.get("compile_mode", "default")

    if compile_requested:
        try:
            import torch._dynamo as dynamo
            dynamo.config.suppress_errors = True
            try:
                dynamo.config.log_level = logging.ERROR
            except Exception:
                pass
        except Exception:
            pass
        try:
            import importlib
            fu = importlib.import_module("transformers.file_utils")
            if not hasattr(fu, "ModelOutput"):
                utils_mod = importlib.import_module("transformers.utils")
                if hasattr(utils_mod, "ModelOutput"):
                    setattr(fu, "ModelOutput", getattr(utils_mod, "ModelOutput"))
        except Exception:
            pass
        torch_compile = getattr(torch, "compile", None)
        if torch_compile:
            try:
                model = torch_compile(model, mode=compile_mode)
                print(f"[compile] torch.compile enabled (mode={compile_mode})")
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
                    pass
            except Exception as exc:
                print(f"[compile] torch.compile failed ({exc}); continuing without compilation.")
        else:
            print("[compile] torch.compile not available in this PyTorch build.")

    freeze_backbone = bool(cfg.get("freeze_backbone", False))
    if freeze_backbone:
        frozen_count = 0
        trainable_count = 0
        for name, param in model.named_parameters():
            if "offset_projs" in name or "termination_head" in name:
                param.requires_grad = True
                trainable_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1
        print(f"[freeze] Backbone frozen: {frozen_count} tensors frozen, {trainable_count} tensors trainable (offset_projs and termination_head)")

    # 1. Unfreeze encoder parameters if configured
    unfreeze_encoder = bool(cfg.get("unfreeze_encoder", False))
    if use_shape_guidance and encoder is not None:
        if unfreeze_encoder:
            print("[biophysics] Unfreezing NucleotideEncoder parameters for joint training.")
            for p in encoder.parameters():
                p.requires_grad = True
        else:
            print("[biophysics] Keeping NucleotideEncoder parameters frozen.")
            for p in encoder.parameters():
                p.requires_grad = False

    # 2. Gather model parameters and partition into learning rate groups
    embedding_params = []
    backbone_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Put embeddings, shape projections, and heads in the fast group
        if "transformer.wte" in name or "shape_proj" in name or "offset_projs" in name or "termination_head" in name:
            embedding_params.append(param)
        else:
            backbone_params.append(param)

    # 3. Add encoder parameters to backbone params (slow learning rate group)
    if use_shape_guidance and encoder is not None and unfreeze_encoder:
        backbone_params.extend([p for p in encoder.parameters() if p.requires_grad])

    trainable_params = embedding_params + backbone_params

    # 4. Formulate optimizer parameter groups
    lr_base = float(cfg.get("lr", 5e-6))
    lr_embed = float(cfg.get("lr_embedding", lr_base))
    
    param_groups = []
    if embedding_params:
        param_groups.append({
            "params": embedding_params,
            "lr": lr_embed,
            "weight_decay": 0.0
        })
        print(f"[optim] Group 1 (Fast LR: {lr_embed:.1e}): {len(embedding_params)} tensors (embeddings & projections)")
    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": lr_base,
            "weight_decay": float(cfg.get("weight_decay", 0.05))
        })
        print(f"[optim] Group 2 (Base LR: {lr_base:.1e}): {len(backbone_params)} tensors (backbone & encoder)")

    if cfg.get("optimizer", "adamw").lower() == "adafactor":
        try:
            from transformers.optimization import Adafactor  # type: ignore
        except Exception:
            raise RuntimeError("transformers not installed; pip install transformers to use Adafactor")
        optim = Adafactor(
            param_groups,
            scale_parameter=False,
            relative_step=False,
        )
    else:
        optim = torch.optim.AdamW(param_groups)

    scheduler_name = str(cfg.get("scheduler", "cosine")).lower()
    if scheduler_name not in {"cosine", "plateau"}:
        print(f"[warn] Unknown scheduler '{scheduler_name}', defaulting to cosine.")
        scheduler_name = "cosine"

    gacc = cfg.get("grad_accum_steps", 16)
    max_nonfinite_groups = int(cfg.get("max_nonfinite_accumulation_groups", 3))
    if max_nonfinite_groups < -1:
        raise ValueError("max_nonfinite_accumulation_groups must be -1 or greater")
    min_lr = float(cfg.get("min_lr", 1e-5))
    base_lr = float(cfg["lr"])

    epochs_cfg = cfg.get("epochs", 5)
    n_params = sum(p.numel() for p in model.parameters())
    tokens_per_param = float(cfg.get("tokens_per_param", 20.0))
    if isinstance(epochs_cfg, str) and epochs_cfg.strip().lower() == "auto":
        tokens_target = max(1.0, tokens_per_param * float(n_params))
        tokens_per_epoch = max(1.0, float(len(train_ds) * cfg["block_size"]))
        est_epochs = int(math.ceil(tokens_target / tokens_per_epoch))
        est_epochs = max(int(cfg.get("epochs_min", 1)), min(est_epochs, int(cfg.get("epochs_max", max(1, est_epochs)))))
        max_epochs = est_epochs
        print(
            f"[epochs-auto] tokens_per_param={tokens_per_param} n_params={n_params} → target_tokens={int(tokens_target)}; "
            f"tokens_per_epoch≈{int(tokens_per_epoch)} → epochs={max_epochs}"
        )
    else:
        max_epochs = int(epochs_cfg)
    steps_per_epoch = math.ceil(len(train_loader) / max(1, gacc))
    computed_total_steps = max(1, steps_per_epoch * max_epochs)
    total_steps = int(cfg.get("scheduler_total_steps", computed_total_steps))
    if total_steps <= 0:
        raise ValueError("scheduler_total_steps must be positive")
    warmup_steps = resolve_warmup_steps(cfg, total_steps)
    cfg["resolved_warmup_steps"] = warmup_steps
    if "warmup_fraction" in cfg:
        print(
            f"[scheduler] warmup_fraction={float(cfg['warmup_fraction']):.6f} "
            f"resolved_warmup_steps={warmup_steps}/{total_steps}"
        )
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

    if transfer_path:
        print(f"[transfer] initializing model from {transfer_path}")
        ckpt_transfer = torch.load(transfer_path, map_location=device)
        sd = (
            ckpt_transfer["model"]
            if isinstance(ckpt_transfer, dict) and "model" in ckpt_transfer
            else ckpt_transfer
        )
        transfer_cfg = ckpt_transfer.get("cfg", {}) if isinstance(ckpt_transfer, dict) else {}
        source_itos = _read_itos(transfer_cfg.get("itos_path"), Path.cwd())
        if source_itos is None:
            transfer_checkpoint_path = Path(transfer_path).resolve()
            for candidate in (
                transfer_checkpoint_path.parent / "itos.txt",
                transfer_checkpoint_path.parent.parent / "itos.txt",
            ):
                source_itos = _read_itos(str(candidate))
                if source_itos is not None:
                    break
        transfer_report = _load_transfer_state_dict(
            model,
            sd,
            source_itos=source_itos,
            target_itos=list(vocabulary_contract.tokens),
        )
        source_embedding_rows = (
            int(sd["tok_emb.weight"].shape[0]) if "tok_emb.weight" in sd else None
        )
        vocabulary_provenance["legacy_adaptation"] = bool(
            source_embedding_rows != vocabulary_contract.size
            or transfer_report["loaded_rows"]
        )
        vocabulary_provenance["transfer"] = {
            "checkpoint": str(transfer_path),
            "source_embedding_rows": source_embedding_rows,
            "source_tokenizer_entries": (
                len(source_itos) if source_itos is not None else None
            ),
            "target_vocab_size": vocabulary_contract.size,
            "loaded_rows": transfer_report["loaded_rows"],
            "skipped": transfer_report["skipped"],
        }
        cfg["vocabulary"] = vocabulary_provenance
        write_vocabulary_manifest(vocabulary_provenance, ckpt_dir.parent / "vocabulary.json")
        print(
            "[transfer] loaded_exact="
            f"{len(transfer_report['loaded_exact'])} row_loaded="
            f"{transfer_report['loaded_rows']}"
        )

    def train_loader_for_epoch(epoch: int):
        loader, _, _, _ = build_codon_lm_dataloaders(
            train_ds, val_ds, _loader_cfg_for_epoch(epoch + 1)
        )
        return loader

    task = CodonLMTask(
        model=model,
        train_loader_factory=train_loader_for_epoch,
        validation_loader=val_loader,
        device=device,
        config=cfg,
        encoder=encoder,
        lookup_table=lookup_table,
        replay_loader=replay_loader,
        replay_generator=(replay_generator if replay_loss_enabled else None),
        termination_class_weights=termination_class_weights,
        replay_class_weights=replay_class_weights,
        multi_offset_weights=multi_offset_weights,
    )
    strategy = AccumulatedBackpropStrategy(
        optim,
        scheduler=scheduler,
        parameters=trainable_params,
        precision=PrecisionPolicy(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp,
            scale_gradients=False,
        ),
        scheduler_interval="update" if use_cosine else "epoch",
        scheduler_metric="loss",
        warmup_steps=(warmup_steps if not use_cosine else 0),
        warmup_lrs=([base_lr] * len(optim.param_groups) if not use_cosine else None),
    )
    engine = TrainingEngine(
        task=task,
        strategy=strategy,
        run=training_run,
        config=EngineConfig(
            epochs=max_epochs,
            grad_accum_steps=int(gacc),
            monitor="loss",
            epoch_checkpoint_pattern=("epoch_{epoch}.pt" if cfg.get("save_epochs") else None),
            best_checkpoint_pattern="best_epoch_{epoch:03d}.pt",
            max_aborted_groups=max_nonfinite_groups,
            early_stop_patience=int(cfg.get("early_stop_patience", 5)),
        ),
        device=device,
        callbacks=[
            CodonLMConsole(
                log_csv,
                multi_offset_weights=multi_offset_weights,
                termination_enabled=termination_loss_enabled,
                replay_enabled=replay_loss_enabled,
                optimizer=optim,
            )
        ],
        wall_timer=WallTimer(cfg.get("max_time_minutes")),
        checkpoint_policy=PeriodicCheckpointPolicy(
            every_steps=int(cfg.get("checkpoint_every_steps", 0) or 0),
            every_minutes=float(cfg.get("checkpoint_every_minutes", 0.0) or 0.0),
        ),
        run_fingerprint=run_fingerprint,
        checkpoint_decoder=decode_codon_lm_checkpoint,
        checkpoint_payload_adapter=make_codon_lm_checkpoint_adapter(
            cfg,
            batch_size=int(cfg["batch_size"]),
            grad_accum_steps=int(gacc),
            train_examples=len(train_ds),
            train_batches=len(train_loader),
            max_nonfinite_groups=max_nonfinite_groups,
        ),
    )
    try:
        result = engine.fit()
        meta = {
            "run_id": run_id,
            "status": "completed" if result.status == "complete" else "stopped",
            "best_epoch": engine.best_epoch,
            "best_val_loss": engine.best_metric,
            "accumulation_health": strategy.state_dict()["accumulation_health"],
            "model_spec": model.to_dict() if hasattr(model, "to_dict") else {},
        }
        write_meta(ckpt_dir, meta)
        return result
    except Exception as exc:
        write_failure_meta(exc)
        raise
    finally:
        training_run.close()
