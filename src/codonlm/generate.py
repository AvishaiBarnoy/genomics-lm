from __future__ import annotations

from typing import Dict, List, Tuple

import torch


STOP_CODONS = {"TAA", "TAG", "TGA"}


@torch.no_grad()
def _next_token_logits(model, device: torch.device, ids: List[int]) -> torch.Tensor:
    max_T = getattr(model, "block_size", None)
    ctx = ids[-max_T:] if max_T is not None else ids
    x = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
    logits, _ = model(x)
    return logits[0, -1]


@torch.no_grad()
def generate_cds_constrained(
    model,
    device: torch.device,
    ctx_ids: List[int],
    stoi: Dict[str, int],
    itos: List[str],
    target_codons: int,
    hard_cap: int,
    require_terminal_stop: bool = False,
    temperature: float = 1.0,
    topk: int = 0,
) -> Tuple[List[int], Dict[str, object]]:
    """Generate codons up to constraints and return (ids, info).

    info includes: had_terminal_stop, early_stop, hit_hard_cap
    """
    ids = list(ctx_ids)
    had_terminal_stop = False
    early_stop = False
    hit_hard_cap = False

    # Track generated codon count (exclude any BOS and specials in decoding loop)
    # We count codon tokens after the prefix length present in ctx_ids.
    # Here we simply monitor total newly added tokens.
    new_codons = 0
    eos_idx = stoi.get("<EOS_CDS>")

    while new_codons < int(hard_cap):
        logits = _next_token_logits(model, device, ids)
        if temperature != 1.0:
            logits = logits / max(1e-6, float(temperature))
        probs = torch.softmax(logits, dim=-1)
        if topk and topk > 0:
            vals, idxs = torch.topk(probs, k=min(topk, probs.numel()))
            pick = torch.multinomial(vals, 1).item()
            next_id = idxs[pick].item()
        else:
            next_id = torch.multinomial(probs, 1).item()
        ids.append(int(next_id))

        # decode this token to a codon string to check stops
        tok = itos[next_id] if 0 <= next_id < len(itos) else ""
        is_codon = len(tok) == 3 and set(tok) <= set("ACGT")
        if is_codon:
            new_codons += 1
            if tok in STOP_CODONS:
                # Stop codon occurred
                if new_codons < int(target_codons):
                    # early stop relative to target
                    early_stop = True
                    if not require_terminal_stop:
                        had_terminal_stop = True
                        break
                    # else keep going to hit terminal condition or hard cap
                else:
                    had_terminal_stop = True
                    # reached or exceeded target and got terminal stop
                    break

        # EOS handling
        if eos_idx is not None and next_id == eos_idx:
            # End of gene marker; accept if meets length target OR not enforcing terminal stop
            if new_codons >= int(target_codons) or not require_terminal_stop:
                break
            # otherwise continue until hard cap or a biological stop

        # If we hit target length without terminal stop and require it, continue until first stop or hard cap.
        if new_codons >= int(target_codons) and not require_terminal_stop:
            # Not requiring terminal stop: stop at target length
            break

    if new_codons >= int(hard_cap):
        hit_hard_cap = True

    info = {
        "had_terminal_stop": bool(had_terminal_stop),
        "early_stop": bool(early_stop),
        "hit_hard_cap": bool(hit_hard_cap),
        "target_codons": int(target_codons),
        "generated_codons": int(new_codons),
    }
    return ids, info


__all__ = ["generate_cds_constrained", "STOP_CODONS"]

