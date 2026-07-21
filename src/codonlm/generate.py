from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


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


def _sample_token(logits: torch.Tensor, temperature: float, topk: int) -> int:
    if temperature != 1.0:
        logits = logits / max(1e-6, float(temperature))
    probs = torch.softmax(logits, dim=-1)
    if topk and topk > 0:
        vals, idxs = torch.topk(probs, k=min(topk, probs.numel()))
        pick = torch.multinomial(vals, 1).item()
        return int(idxs[pick].item())
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate_model_raw(
    model,
    device: torch.device,
    ctx_ids: List[int],
    stoi: Dict[str, int],
    itos: List[str],
    max_new_tokens: int,
    temperature: float = 1.0,
    topk: int = 0,
) -> Tuple[List[int], Dict[str, object]]:
    """Sample the model vocabulary without CDS masking or forced termination."""
    ids = list(ctx_ids)
    eos_idx = stoi.get("<EOS_CDS>")
    had_terminal_stop = False
    generated_codons = 0
    stop_reason = "max_new_tokens"

    for _ in range(int(max_new_tokens)):
        logits = _next_token_logits(model, device, ids)
        next_id = _sample_token(logits, temperature=temperature, topk=topk)
        ids.append(next_id)
        tok = itos[next_id] if 0 <= next_id < len(itos) else ""
        if len(tok) == 3 and set(tok) <= set("ACGT"):
            generated_codons += 1
            if tok in STOP_CODONS:
                had_terminal_stop = True
                stop_reason = "biological_stop"
                break
        if eos_idx is not None and next_id == eos_idx:
            stop_reason = "eos"
            break

    generated_tokens = len(ids) - len(ctx_ids)
    return ids, {
        "protocol": "raw_model",
        "cds_only": False,
        "require_terminal_stop": False,
        "guidance_components": [],
        "had_terminal_stop": had_terminal_stop,
        "early_stop": False,
        "hit_hard_cap": stop_reason == "max_new_tokens",
        "generated_codons": generated_codons,
        "generated_tokens": generated_tokens,
        "max_new_tokens": int(max_new_tokens),
        "stop_reason": stop_reason,
    }


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

    total_new_tokens = 0
    while new_codons < int(hard_cap) and total_new_tokens < 3 * int(hard_cap):
        total_new_tokens += 1
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
        next_id = _sample_token(logits, temperature=temperature, topk=topk)
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

    guidance_components = []
    if termination_bias_enabled:
        guidance_components.append("termination_bias")
    if multi_offset_prior_enabled:
        guidance_components.append("multi_offset_prior")
    if require_terminal_stop:
        guidance_components.append("forced_terminal_stop")
    if not cds_only:
        guidance_components.append("non_cds_tokens")
    info = {
        "protocol": "guided" if guidance_components else "cds_constrained",
        "guidance_components": guidance_components,
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
        "require_terminal_stop": bool(require_terminal_stop),
        "generated_tokens": int(total_new_tokens),
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


__all__ = [
    "generate_model_raw",
    "generate_cds_constrained",
    "generate_cds_red",
    "batch_red_sampler",
    "STOP_CODONS",
    "batch_score_critic",
    "generate_cds_critic_guided"
]

@torch.no_grad()
def batch_score_critic(
    critic_model,
    tokenizer,
    aa_seqs: List[str],
    target_task: str,
    target_class_idx: int | None,
    device: torch.device,
    ebm_model = None
) -> torch.Tensor:
    """
    Evaluates a batch of candidate amino acid sequences on the critic or EBM.
    Returns a tensor of scores (log probabilities or negative energy scores) of shape (K,).
    """
    if not aa_seqs:
        return torch.zeros(0, device=device)

    # 1. Tokenize all candidate sequences
    batch_ids = []
    max_len = 0
    for aa_seq in aa_seqs:
        # Standard critic tokenization (BOS + encoded sequence + EOS)
        ids = [tokenizer.bos_token_id] + tokenizer.encode_sequence(aa_seq) + [tokenizer.eos_token_id]
        batch_ids.append(ids)
        max_len = max(max_len, len(ids))

    # 2. Pad sequences to max_len
    padded_batch = []
    pad_token = tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else 0
    for ids in batch_ids:
        pad_len = max_len - len(ids)
        padded_ids = ids + [pad_token] * pad_len
        padded_batch.append(padded_ids)

    # 3. Create input tensor and run forward pass
    input_tensor = torch.tensor(padded_batch, dtype=torch.long, device=device)
    
    if target_task == "ebm" and ebm_model is not None:
        # For EBM, extract sequence latent embedding and pass to EBM
        seq_length = input_tensor.size(1)
        pos_embeds = critic_model.backbone.position_embedding(torch.arange(seq_length, device=device))
        
        x = critic_model.backbone.token_embedding(input_tensor)
        x = critic_model.backbone.dropout(x + pos_embeds)
        
        is_causal = not getattr(critic_model.config, "bidirectional", True)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_length, device=device) if is_causal else None
        
        for block in critic_model.backbone.transformer_blocks:
            x = block(x, src_mask=causal_mask)
            
        if critic_model.pooling_type == "attention":
            pooled_feats, _ = critic_model.pooler(x)
        else:
            pooled_feats = x.mean(dim=1)
            
        latent = critic_model.shared_latent(pooled_feats)
        energy = ebm_model(latent) # (K, 1) or (K,)
        # Return negative energy score (as low energy is preferred, higher is better)
        return -energy.view(-1)
        
    else:
        # Classifier heads
        logits_dict = critic_model(input_tensor)
        if target_task not in logits_dict:
            return torch.zeros(len(aa_seqs), device=device)
            
        logits = logits_dict[target_task] # (K, n_classes)
        probs = torch.softmax(logits, dim=-1)
        
        # Default target class is class 0
        class_idx = target_class_idx if target_class_idx is not None else 0
        if class_idx >= probs.size(-1):
            class_idx = 0
            
        target_probs = probs[:, class_idx]
        return torch.log(target_probs + 1e-10)

def generate_cds_critic_guided(
    model,
    critic_model,
    c_tokenizer,
    device: torch.device,
    ctx_ids: List[int],
    stoi: Dict[str, int],
    itos: List[str],
    target_codons: int,
    hard_cap: int,
    alpha: float = 0.5,
    guide_top_k: int = 5,
    target_task: str = "stability",
    target_class_idx: int | None = None,
    ebm_model = None,
    temperature: float = 1.0,
    cds_only: bool = True,
    require_terminal_stop: bool = False
) -> Tuple[List[int], Dict[str, object]]:
    """
    Generate codons guided token-by-token (codon-by-codon) by the Protein Critic or EBM.
    """
    from src.eval.inference_playground import translate_codons_to_aa

    ids = list(ctx_ids)
    had_terminal_stop = False
    early_stop = False
    hit_hard_cap = False
    new_codons = 0
    eos_idx = stoi.get("<EOS_CDS>")
    allowed_cds_ids = _cds_token_ids(itos) if cds_only else []

    total_new_tokens = 0
    while new_codons < int(hard_cap) and total_new_tokens < 3 * int(hard_cap):
        total_new_tokens += 1
        # 1. Get generator next token logits
        logits = _next_token_logits(model, device, ids)
        if cds_only:
            logits = _mask_to_allowed_tokens(logits, allowed_cds_ids)
        if temperature != 1.0:
            logits = logits / max(1e-6, float(temperature))
            
        probs = torch.softmax(logits, dim=-1)

        # 2. Prune candidates to top-K for critic scoring
        k_val = min(guide_top_k, probs.numel())
        top_vals, top_idxs = torch.topk(probs, k=k_val)
        
        # 3. Formulate candidate sequences and translate to AA
        aa_seqs = []
        candidate_ids = []
        for c_id in top_idxs:
            c_id = c_id.item()
            cand_ids = ids + [c_id]
            # Convert candidate codon tokens to amino acids
            cand_codons = [itos[i] for i in cand_ids if len(itos[i]) == 3 and not (itos[i].startswith("<") or itos[i].endswith(">"))]
            aa_seq = translate_codons_to_aa(cand_codons)
            aa_seqs.append(aa_seq)
            candidate_ids.append(c_id)

        # 4. Score all candidates using the batch helper
        critic_scores = batch_score_critic(
            critic_model=critic_model,
            tokenizer=c_tokenizer,
            aa_seqs=aa_seqs,
            target_task=target_task,
            target_class_idx=target_class_idx,
            device=device,
            ebm_model=ebm_model
        ) # (k_val,)

        # 5. Blend probabilities
        # log_blended(c) = log P_gen(c) + alpha * log P_critic(c)
        gen_log_probs = torch.log(top_vals + 1e-10)
        blended_log_probs = gen_log_probs + alpha * critic_scores
        blended_probs = torch.softmax(blended_log_probs, dim=-1)

        # 6. Sample next codon from blended distribution
        pick = torch.multinomial(blended_probs, 1).item()
        next_id = candidate_ids[pick]
        ids.append(next_id)

        # decode this token to a codon string to check stops
        tok = itos[next_id] if 0 <= next_id < len(itos) else ""
        is_codon = len(tok) == 3 and set(tok) <= set("ACGT")
        if is_codon:
            new_codons += 1
            if tok in STOP_CODONS:
                if new_codons < int(target_codons):
                    early_stop = True
                    if not require_terminal_stop:
                        had_terminal_stop = True
                        break
                else:
                    had_terminal_stop = True
                    break

        # EOS handling
        if eos_idx is not None and next_id == eos_idx:
            if new_codons >= int(target_codons) or not require_terminal_stop:
                break

        # Length target stop
        if new_codons >= int(target_codons) and not require_terminal_stop:
            break

    if new_codons >= int(hard_cap):
        hit_hard_cap = True

    guidance_components = ["ebm" if ebm_model is not None else "critic"]
    if require_terminal_stop:
        guidance_components.append("forced_terminal_stop")
    if not cds_only:
        guidance_components.append("non_cds_tokens")
    info = {
        "protocol": "guided",
        "guidance_components": guidance_components,
        "had_terminal_stop": bool(had_terminal_stop),
        "early_stop": bool(early_stop),
        "hit_hard_cap": bool(hit_hard_cap),
        "target_codons": int(target_codons),
        "generated_codons": int(new_codons),
        "cds_only": bool(cds_only),
        "require_terminal_stop": bool(require_terminal_stop),
        "generated_tokens": int(total_new_tokens),
    }
    return ids, info


# Standard genetic code dictionary
CODON_TABLE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAG':'K', 'AAA':'K',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAG':'E', 'GAA':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAG':'Q', 'CAA':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
    'AGA':'R', 'AGG':'R', 'AGC':'S', 'AGT':'S',
}

AA_TO_CODONS = defaultdict(list)
for codon, aa in CODON_TABLE.items():
    AA_TO_CODONS[aa].append(codon)


@torch.no_grad()
def generate_cds_synonymous(
    model,
    critic_model,
    c_tokenizer,
    device: torch.device,
    ctx_ids: List[int],
    stoi: Dict[str, int],
    itos: List[str],
    target_protein: str,
    alpha: float = 0.5,
    guide_top_k: int = 5,
    target_task: str = "stability",
    target_class_idx: int | None = None,
    ebm_model = None,
    temperature: float = 1.0,
) -> Tuple[List[int], dict]:
    """
    Generate codon tokens constrained to translate exactly to the target amino acid sequence.
    Supports EBM and Critic logit blending for thermodynamic/fold shape optimization.
    """
    from src.eval.inference_playground import translate_codons_to_aa

    ids = list(ctx_ids)
    new_codons = 0
    eos_idx = stoi.get("<EOS_CDS>")
    
    # 1. Constrained codon generation for each amino acid residue
    for t, target_aa in enumerate(target_protein):
        logits = _next_token_logits(model, device, ids)
        
        allowed_codons = AA_TO_CODONS.get(target_aa.upper(), [])
        allowed_ids = [stoi[c] for c in allowed_codons if c in stoi]
        if not allowed_ids:
            allowed_ids = _cds_token_ids(itos)
            
        logits = _mask_to_allowed_tokens(logits, allowed_ids)
        if temperature != 1.0:
            logits = logits / max(1e-6, float(temperature))
            
        probs = torch.softmax(logits, dim=-1)
        
        if (critic_model is not None or ebm_model is not None) and alpha != 0.0:
            valid_idxs = torch.where(probs > 0.0)[0]
            k_val = min(guide_top_k, valid_idxs.numel())
            if k_val > 0:
                top_vals, top_sub_idxs = torch.topk(probs[valid_idxs], k=k_val)
                top_idxs = valid_idxs[top_sub_idxs]
                
                aa_seqs = []
                candidate_ids = []
                for c_id in top_idxs:
                    c_id = c_id.item()
                    cand_ids = ids + [c_id]
                    cand_codons = [itos[i] for i in cand_ids if len(itos[i]) == 3 and not (itos[i].startswith("<") or itos[i].endswith(">"))]
                    aa_seq = translate_codons_to_aa(cand_codons)
                    aa_seqs.append(aa_seq)
                    candidate_ids.append(c_id)
                    
                critic_scores = batch_score_critic(
                    critic_model=critic_model,
                    tokenizer=c_tokenizer,
                    aa_seqs=aa_seqs,
                    target_task=target_task,
                    target_class_idx=target_class_idx,
                    device=device,
                    ebm_model=ebm_model
                )
                
                gen_log_probs = torch.log(top_vals + 1e-10)
                blended_log_probs = gen_log_probs + alpha * critic_scores
                blended_probs = torch.softmax(blended_log_probs, dim=-1)
                
                pick = torch.multinomial(blended_probs, 1).item()
                next_id = candidate_ids[pick]
            else:
                next_id = torch.multinomial(probs, 1).item()
        else:
            next_id = torch.multinomial(probs, 1).item()
            
        ids.append(next_id)
        new_codons += 1

    # 2. Append stop codon at the end of the open reading frame
    logits = _next_token_logits(model, device, ids)
    stop_codons = AA_TO_CODONS.get('_', ['TAA', 'TAG', 'TGA'])
    stop_ids = [stoi[c] for c in stop_codons if c in stoi]
    logits = _mask_to_allowed_tokens(logits, stop_ids)
    probs = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, 1).item()
    ids.append(next_id)
    new_codons += 1
    # 3. Append UTR boundary token (EOS)
    if eos_idx is not None:
        ids.append(eos_idx)
        
    info = {
        "protocol": "guided",
        "guidance_components": [
            "synonymous_template",
            *(["ebm" if ebm_model is not None else "critic"] if critic_model is not None or ebm_model is not None else []),
        ],
        "had_terminal_stop": True,
        "early_stop": False,
        "hit_hard_cap": False,
        "target_codons": len(target_protein) + 1,
        "generated_codons": new_codons,
        "cds_only": True,
        "require_terminal_stop": True,
        "generated_tokens": len(ids) - len(ctx_ids),
    }
    return ids, info
