from __future__ import annotations

from typing import Dict, List, Tuple

import torch


STOP_CODONS = {"TAA", "TAG", "TGA"}


@torch.no_grad()
def _next_token_logits(
    model,
    device: torch.device,
    ids: List[int],
    return_aux: bool = False,
):
    max_T = getattr(model, "block_size", None)
    ctx = ids[-max_T:] if max_T is not None else ids
    x = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
    if return_aux:
        logits, _, aux = model(x, return_aux=True)
        return logits[0, -1], aux
    logits, _ = model(x)
    return logits[0, -1]


def _stop_token_ids(stoi: Dict[str, int]) -> List[int]:
    return [stoi[codon] for codon in sorted(STOP_CODONS) if codon in stoi]


def _cds_token_ids(itos: List[str]) -> List[int]:
    return [
        idx
        for idx, tok in enumerate(itos)
        if len(tok) == 3 and set(tok) <= set("ACGT")
    ]


def _mask_to_allowed_tokens(logits: torch.Tensor, allowed_ids: List[int]) -> torch.Tensor:
    if not allowed_ids:
        return logits
    allowed = torch.tensor(allowed_ids, dtype=torch.long, device=logits.device)
    masked = torch.full_like(logits, float("-inf"))
    masked[allowed] = logits[allowed]
    return masked


def _apply_termination_stop_bias(
    logits: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    stop_ids: List[int],
    stop_bias: float,
    trigger_class_max: int,
) -> tuple[torch.Tensor, int | None]:
    if stop_bias <= 0.0 or not stop_ids:
        return logits, None
    term_logits = aux.get("termination_logits")
    if term_logits is None:
        return logits, None
    pred_class = int(term_logits[0, -1].argmax().item())
    if pred_class <= int(trigger_class_max):
        logits = logits.clone()
        logits[torch.tensor(stop_ids, device=logits.device)] += float(stop_bias)
    return logits, pred_class


def _apply_multi_offset_priors(
    logits: torch.Tensor,
    aux: Dict[str, object],
    ctx_len: int,
    offsets: List[int],
    weights: Dict[int, float],
) -> torch.Tensor:
    offset_logits_dict = aux.get("offset_logits")
    if not offset_logits_dict or not isinstance(offset_logits_dict, dict):
        return logits

    modified_logits = logits.clone()
    for offset in offsets:
        weight = weights.get(offset, 0.0)
        if weight == 0.0:
            continue
        idx = ctx_len - offset
        if idx >= 0 and offset in offset_logits_dict:
            prior = offset_logits_dict[offset][0, idx]
            modified_logits += float(weight) * prior
    return modified_logits


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
    termination_bias_enabled: bool = False,
    termination_stop_bias: float = 0.0,
    termination_trigger_class_max: int = 0,
    termination_bias_window: int = 0,
    cds_only: bool = True,
    multi_offset_prior_enabled: bool = False,
    multi_offset_prior_weights: Dict[int, float] | None = None,
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
    stop_ids = _stop_token_ids(stoi)
    allowed_cds_ids = _cds_token_ids(itos) if cds_only else []
    termination_bias_steps = 0
    last_termination_class = None

    while new_codons < int(hard_cap):
        bias_length_ok = new_codons >= max(0, int(target_codons) - int(termination_bias_window))
        need_aux = (termination_bias_enabled and bias_length_ok) or multi_offset_prior_enabled
        if need_aux:
            logits, aux = _next_token_logits(model, device, ids, return_aux=True)
        else:
            logits = _next_token_logits(model, device, ids)
            aux = {}

        if multi_offset_prior_enabled and aux and multi_offset_prior_weights:
            max_T = getattr(model, "block_size", None)
            ctx_len = min(len(ids), max_T) if max_T is not None else len(ids)
            logits = _apply_multi_offset_priors(
                logits,
                aux,
                ctx_len=ctx_len,
                offsets=list(multi_offset_prior_weights.keys()),
                weights=multi_offset_prior_weights,
            )

        if termination_bias_enabled and bias_length_ok and aux:
            logits, term_class = _apply_termination_stop_bias(
                logits,
                aux,
                stop_ids=stop_ids,
                stop_bias=float(termination_stop_bias),
                trigger_class_max=int(termination_trigger_class_max),
            )
            if term_class is not None:
                last_termination_class = term_class
                if term_class <= int(termination_trigger_class_max) and float(termination_stop_bias) > 0:
                    termination_bias_steps += 1
        if cds_only:
            logits = _mask_to_allowed_tokens(logits, allowed_cds_ids)
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
        "termination_bias_enabled": bool(termination_bias_enabled),
        "termination_bias_steps": int(termination_bias_steps),
        "termination_bias_window": int(termination_bias_window),
        "last_termination_class": last_termination_class,
        "cds_only": bool(cds_only),
    }
    return ids, info


@torch.no_grad()
def generate_cds_red(
    model,
    device: torch.device,
    ctx_ids: List[int],
    stoi: Dict[str, int],
    itos: List[str],
    target_codons: int,
    hard_cap: int,
    max_attempts: int = 5,
    temperature: float = 1.0,
    topk: int = 0,
    termination_bias_enabled: bool = False,
    termination_stop_bias: float = 0.0,
    termination_trigger_class_max: int = 0,
    termination_bias_window: int = 0,
    cds_only: bool = True,
) -> Tuple[List[int], Dict[str, object]]:
    """Simple ReD wrapper for a single prefix: retry until success or max_attempts."""
    total_tokens = 0
    last_ids = []
    last_info = {}
    for i in range(max_attempts):
        ids, info = generate_cds_constrained(
            model, device, ctx_ids, stoi, itos, target_codons, hard_cap,
            require_terminal_stop=True, temperature=temperature, topk=topk,
            termination_bias_enabled=termination_bias_enabled,
            termination_stop_bias=termination_stop_bias,
            termination_trigger_class_max=termination_trigger_class_max,
            termination_bias_window=termination_bias_window,
            cds_only=cds_only,
        )
        total_tokens += info["generated_codons"]
        last_ids, last_info = ids, info
        if info["had_terminal_stop"]:
            last_info["attempts"] = i + 1
            last_info["total_tokens_red"] = total_tokens
            return ids, last_info
    
    last_info["attempts"] = max_attempts
    last_info["total_tokens_red"] = total_tokens
    return last_ids, last_info


@torch.no_grad()
def batch_red_sampler(
    model,
    device: torch.device,
    contexts: List[List[int]],
    stoi: Dict[str, int],
    itos: List[str],
    target_codons: int,
    hard_cap: int,
    global_token_budget: int,
    temperature: float = 1.0,
    topk: int = 0,
    termination_bias_enabled: bool = False,
    termination_stop_bias: float = 0.0,
    termination_trigger_class_max: int = 0,
    termination_bias_window: int = 0,
    cds_only: bool = True,
) -> Tuple[Dict[int, Tuple[List[int], Dict]], List[int], int]:
    """Perform Reset-and-Discard across multiple prefixes.
    
    Returns:
        solved: Dict mapping original index to (ids, info)
        remaining: List of original indices that never reached a terminal stop
        total_tokens: Total tokens spent during the process
    """
    # active_tasks: list of (ids, original_index)
    active_tasks = [(list(ctx), i) for i, ctx in enumerate(contexts)]
    solved = {}
    total_tokens = 0
    
    round_idx = 0
    while active_tasks and total_tokens < global_token_budget:
        round_idx += 1
        next_active = []
        for ctx, idx in active_tasks:
            if total_tokens >= global_token_budget:
                next_active.append((ctx, idx))
                continue
            
            # One attempt (τ=1 is optimal per paper)
            gen_ids, info = generate_cds_constrained(
                model, device, ctx, stoi, itos, target_codons, hard_cap,
                require_terminal_stop=True, temperature=temperature, topk=topk,
                termination_bias_enabled=termination_bias_enabled,
                termination_stop_bias=termination_stop_bias,
                termination_trigger_class_max=termination_trigger_class_max,
                termination_bias_window=termination_bias_window,
                cds_only=cds_only,
            )
            spent = info["generated_codons"]
            total_tokens += spent
            
            if info["had_terminal_stop"]:
                info["round"] = round_idx
                solved[idx] = (gen_ids, info)
            else:
                next_active.append((ctx, idx))
        active_tasks = next_active
        
    remaining = [idx for _, idx in active_tasks]
    return solved, remaining, total_tokens


__all__ = ["generate_cds_constrained", "generate_cds_red", "batch_red_sampler", "STOP_CODONS"]
