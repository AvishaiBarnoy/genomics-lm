import torch
from pathlib import Path
from typing import Optional

def _read_itos(path_value: Optional[str], base_dir: Path | None = None) -> list[str] | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if not path.exists():
        return None
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _load_transfer_state_dict(
    model: torch.nn.Module,
    source_state: dict,
    *,
    source_itos: list[str] | None = None,
    target_itos: list[str] | None = None,
) -> dict:
    """Load a checkpoint into a model, allowing tokenizer/vocab expansion."""
    target_state = model.state_dict()
    adapted = {}
    loaded_exact: list[str] = []
    loaded_rows: list[str] = []
    skipped: list[str] = []

    source_index = {tok: i for i, tok in enumerate(source_itos or [])}
    target_index = {tok: i for i, tok in enumerate(target_itos or [])}
    vocab_row_names = {"tok_emb.weight", "head.weight", "loss_weights"}

    for name, target_tensor in target_state.items():
        source_tensor = source_state.get(name)
        if source_tensor is None:
            skipped.append(name)
            continue
        requires_token_remap = (
            name in vocab_row_names
            and source_index
            and target_index
            and list(source_itos or []) != list(target_itos or [])
        )
        if tuple(source_tensor.shape) == tuple(target_tensor.shape) and not requires_token_remap:
            adapted[name] = source_tensor
            loaded_exact.append(name)
            continue
        if (
            source_tensor.ndim >= 1
            and target_tensor.ndim >= 1
            and tuple(source_tensor.shape[1:]) == tuple(target_tensor.shape[1:])
            and (
                source_tensor.shape[0] != target_tensor.shape[0]
                or requires_token_remap
            )
        ):
            merged = target_tensor.detach().clone()
            copied = 0
            if source_index and target_index:
                for tok, dst_idx in target_index.items():
                    src_idx = source_index.get(tok)
                    if src_idx is None or src_idx >= source_tensor.shape[0] or dst_idx >= merged.shape[0]:
                        continue
                    merged[dst_idx] = source_tensor[src_idx].to(device=merged.device, dtype=merged.dtype)
                    copied += 1
            else:
                copied = min(int(source_tensor.shape[0]), int(merged.shape[0]))
                merged[:copied] = source_tensor[:copied].to(device=merged.device, dtype=merged.dtype)
            if copied:
                adapted[name] = merged
                loaded_rows.append(f"{name}:{copied}")
            else:
                skipped.append(name)
            continue
        skipped.append(name)

    missing, unexpected = model.load_state_dict(adapted, strict=False)
    return {
        "loaded_exact": loaded_exact,
        "loaded_rows": loaded_rows,
        "skipped": skipped,
        "missing": list(missing),
        "unexpected": list(unexpected),
    }
