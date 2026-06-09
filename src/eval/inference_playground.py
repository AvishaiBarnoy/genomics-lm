from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch

from src.codonlm.model_tiny_gpt import TinyGPT
from src.codonlm.codon_tokenize import STOP_CODONS, stoi as CODON_STOI, itos as CODON_ITOS

# Check if protein imports are available
try:
    from src.protein_lm.tokenizer import ProteinTokenizer
    from src.protein_lm.models_multi import MultiTaskProteinClassifier
    from src.protein_lm.models import ProteinConditionalTransformer
    from src.protein_lm.config import ProteinClassifierConfig, ProteinLMConfig, load_config
    PROTEIN_AVAILABLE = True
except ImportError:
    PROTEIN_AVAILABLE = False


def dev() -> torch.device:
    """Returns the optimal device for training/inference, prioritizing MPS and CUDA."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_codon_model(run_id_or_dir: str) -> Tuple[TinyGPT, List[str], Dict[str, int], torch.device]:
    """Loads a trained CodonLM model and its vocabulary from runs/ or outputs/ checkpoints.

    Args:
        run_id_or_dir: Run ID (e.g. '2026-06-04_stage2_6L4H_d256_e10') or folder path.

    Returns:
        A tuple of (model, itos, stoi, device).
    """
    device = dev()
    run_dir = Path(run_id_or_dir)
    if not run_dir.exists():
        # check runs/ or outputs/checkpoints/
        run_dir = Path("runs") / run_id_or_dir
        if not run_dir.exists():
            run_dir = Path("outputs/checkpoints") / run_id_or_dir

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_id_or_dir}")

    # load vocab
    itos_path = run_dir / "itos.txt"
    if not itos_path.exists():
        # fallback to standard codon itos
        itos = CODON_ITOS
        stoi = CODON_STOI
    else:
        itos = [line.strip() for line in itos_path.read_text().splitlines() if line.strip()]
        stoi = {tok: i for i, tok in enumerate(itos)}

    # load weights
    ckpt_path = run_dir / "weights.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = run_dir / "last.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"No weights found in run directory {run_dir}")

    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    cfg = state.get("cfg", {}) if isinstance(state, dict) else {}

    # rebuild model
    vocab_size = int(cfg.get("vocab_size", len(itos)))
    block_size = int(cfg.get("block_size", 256))
    n_layer = int(cfg.get("n_layer", 2))
    n_head = int(cfg.get("n_head", 4))
    n_embd = int(cfg.get("n_embd", 128))

    model = TinyGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.0,
        use_checkpoint=False,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    return model, itos, stoi, device


def query_next_codon(
    model: TinyGPT,
    stoi: Dict[str, int],
    itos: List[str],
    device: torch.device,
    dna_prefix: str,
    top_k: int = 5
) -> List[Dict[str, float]]:
    """Predicts next-codon probabilities given a DNA prefix sequence.

    Args:
        model: Loaded TinyGPT model.
        stoi: Vocabulary lookup dict mapping tokens to IDs.
        itos: Vocabulary list mapping IDs to tokens.
        device: Torch device.
        dna_prefix: Space-separated codons (e.g. 'ATG GCT') or raw DNA string.
        top_k: Number of top predictions to return.

    Returns:
        List of dictionaries with 'token' and 'prob' keys.
    """
    # Clean input
    prefix = dna_prefix.strip().upper()
    if not prefix:
        # start with BOS
        ids = [stoi.get("<BOS_CDS>", 1)]
    elif " " in prefix:
        codons = [c.strip() for c in prefix.split() if c.strip()]
        ids = [stoi.get("<BOS_CDS>", 1)] + [stoi[c] for c in codons if c in stoi]
    else:
        # raw dna string
        L = (len(prefix) // 3) * 3
        ids = [stoi.get("<BOS_CDS>", 1)]
        for i in range(0, L, 3):
            codon = prefix[i:i+3]
            if codon in stoi:
                ids.append(stoi[codon])

    if not ids:
        return []

    # Get predictions
    max_T = getattr(model, "block_size", 256)
    x_ids = ids[-max_T:]
    x = torch.tensor(x_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    with torch.no_grad():
        logits, _ = model(x)
        logits = logits[0, -1]
        probs = torch.softmax(logits, dim=-1)

    top_v, top_i = torch.topk(probs, k=min(top_k, probs.numel()))
    results = []
    for p, i in zip(top_v.tolist(), top_i.tolist()):
        results.append({"token": itos[i], "prob": float(p)})
    return results


def generate_cds(
    model: TinyGPT,
    stoi: Dict[str, int],
    itos: List[str],
    device: torch.device,
    dna_prefix: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    stop_on_eos: bool = True,
    stop_on_bio_stop: bool = True
) -> Tuple[List[str], Dict[str, bool]]:
    """Generates coding sequences starting from a DNA prefix.

    Args:
        model: Loaded TinyGPT model.
        stoi: Vocabulary lookup.
        itos: Vocabulary list.
        device: Torch device.
        dna_prefix: DNA prefix.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-K filtering parameter (0 to disable).
        stop_on_eos: Stop when <EOS_CDS> is produced.
        stop_on_bio_stop: Stop when a biological stop codon is produced.

    Returns:
        A tuple of (generated_tokens, generation_info).
    """
    prefix = dna_prefix.strip().upper()
    if not prefix:
        ids = [stoi.get("<BOS_CDS>", 1)]
    elif " " in prefix:
        codons = [c.strip() for c in prefix.split() if c.strip()]
        ids = [stoi.get("<BOS_CDS>", 1)] + [stoi[c] for c in codons if c in stoi]
    else:
        L = (len(prefix) // 3) * 3
        ids = [stoi.get("<BOS_CDS>", 1)]
        for i in range(0, L, 3):
            codon = prefix[i:i+3]
            if codon in stoi:
                ids.append(stoi[codon])

    eos_idx = stoi.get("<EOS_CDS>")
    max_T = getattr(model, "block_size", 256)
    
    generated_ids = list(ids)
    info = {"hit_hard_cap": False, "hit_eos": False, "hit_bio_stop": False}

    for _ in range(max_new_tokens):
        x_ids = generated_ids[-max_T:] if max_T is not None else generated_ids
        x = torch.tensor(x_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = model(x)
            logits = logits[0, -1]
            
        if temperature != 1.0:
            logits = logits / max(1e-6, float(temperature))
            
        probs = torch.softmax(logits, dim=-1)
        
        if top_k > 0:
            vals, idxs = torch.topk(probs, k=min(top_k, probs.numel()))
            idx = torch.multinomial(vals, 1).item()
            next_id = idxs[idx].item()
        else:
            next_id = torch.multinomial(probs, 1).item()
            
        generated_ids.append(next_id)
        
        tok = itos[next_id] if 0 <= next_id < len(itos) else ""
        if stop_on_eos and eos_idx is not None and next_id == eos_idx:
            info["hit_eos"] = True
            break
        if stop_on_bio_stop and tok in STOP_CODONS:
            info["hit_bio_stop"] = True
            break

    # Extract only the generated portion or return whole sequence
    generated_tokens = [itos[i] for i in generated_ids]
    return generated_tokens, info


def load_protein_classifier(
    config_path: str,
    ckpt_path: str,
    task_vocabs_path: str
) -> Tuple[MultiTaskProteinClassifier, ProteinTokenizer, Dict[str, Dict[int, str]], torch.device]:
    """Loads the multi-task protein classifier model and tokenizers.

    Args:
        config_path: Config file path.
        ckpt_path: Checkpoint weights file path.
        task_vocabs_path: JSON path for task vocabs mapping.

    Returns:
        A tuple of (model, tokenizer, itos_dict, device).
    """
    if not PROTEIN_AVAILABLE:
        raise ImportError("protein_lm package dependencies not available.")

    device = dev()
    
    # load vocabs
    with open(task_vocabs_path, "r") as f:
        vocabs = json.load(f)
        
    task_dims = {
        "family": len(vocabs["pfam"]),
        "function": len(vocabs["ec"]),
        "stability": len(vocabs["stability"])
    }
    
    # load config
    with open(config_path, "r") as f:
        import yaml
        cfg = yaml.safe_load(f) or {}

    tokenizer = ProteinTokenizer()
    model_cfg = ProteinClassifierConfig(
        vocab_size=len(tokenizer.vocab),
        block_size=cfg.get("block_size", 512),
        n_layer=cfg.get("n_layer", 4),
        n_head=cfg.get("n_head", 4),
        n_embd=cfg.get("n_embd", 128),
        dropout=0.0,
        num_classes=0
    )
    
    model = MultiTaskProteinClassifier(model_cfg, task_dims)
    if not os.path.exists(ckpt_path):
        # Fallback to consolidated runs layout
        p = Path(ckpt_path)
        alt_path = Path("runs") / p.parent.name / "checkpoints" / p.name
        if alt_path.exists():
            ckpt_path = str(alt_path)

    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
    else:
        print(f"[warn] checkpoint path {ckpt_path} not found; returning initialized model")
        
    model.to(device).eval()
    
    # Invert vocabs to map index -> label
    itos_dict = {
        "family": {idx: name for name, idx in vocabs["pfam"].items()},
        "function": {idx: name for name, idx in vocabs["ec"].items()},
        "stability": {idx: name for name, idx in vocabs["stability"].items()}
    }
    
    return model, tokenizer, itos_dict, device


def classify_protein(
    model: MultiTaskProteinClassifier,
    tokenizer: ProteinTokenizer,
    itos_dict: Dict[str, Dict[int, str]],
    device: torch.device,
    aa_sequence: str
) -> Dict[str, Dict[str, object]]:
    """Predicts Pfam family, EC function, and stability class for a given protein sequence.

    Args:
        model: Loaded MultiTaskProteinClassifier.
        tokenizer: ProteinTokenizer instance.
        itos_dict: Dictionary mapping task name to index-label dictionaries.
        device: Torch device.
        aa_sequence: Raw amino acid sequence string (e.g. 'MQA...').

    Returns:
        Dictionary containing predictions and scores for 'family', 'function', and 'stability'.
    """
    # Clean input
    seq = aa_sequence.strip().upper().replace(" ", "")
    max_length = getattr(model.config, "block_size", 512)
    
    tokens = [tokenizer.bos_token_id] + tokenizer.encode_sequence(seq)[:max_length-2] + [tokenizer.eos_token_id]
    pad_len = max_length - len(tokens)
    input_ids = tokens + [tokenizer.pad_token_id] * pad_len
    
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        logits_dict = model(x)
        
    results = {}
    for task in ["family", "function", "stability"]:
        logits = logits_dict[task][0]  # (V,)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(logits).item()
        
        # Get label and probability
        label = itos_dict[task].get(pred_idx, f"unknown_{pred_idx}")
        prob = float(probs[pred_idx].item())
        
        # Get top-3 choices
        top_v, top_i = torch.topk(probs, k=min(3, probs.numel()))
        choices = []
        for p, idx in zip(top_v.tolist(), top_i.tolist()):
            choices.append({"label": itos_dict[task].get(idx, f"unknown_{idx}"), "prob": float(p)})
            
        results[task] = {
            "prediction": label,
            "probability": prob,
            "choices": choices
        }
        
    return results


# Standard DNA translation table
CODON_TABLE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}


def translate_codons_to_aa(codon_list: List[str]) -> str:
    """Translates a list of codon strings into a protein amino acid sequence, stopping at stop codons."""
    aa_seq = ""
    for codon in codon_list:
        c = codon.upper()
        if len(c) != 3 or c.startswith("<") or c.endswith(">"):
            continue
        aa = CODON_TABLE.get(c, 'X')
        if aa == '_':
            break
        aa_seq += aa
    return aa_seq


def load_generative_protein_lm(
    config_path: str,
    ckpt_path: str
) -> Tuple[ProteinConditionalTransformer, ProteinTokenizer, torch.device]:
    """Loads a generative ProteinConditionalTransformer model for sequence likelihood scoring.

    Args:
        config_path: Config file path.
        ckpt_path: Checkpoint weights file path.

    Returns:
        A tuple of (model, tokenizer, device).
    """
    if not PROTEIN_AVAILABLE:
        raise ImportError("protein_lm package dependencies not available.")

    device = dev()
    tokenizer = ProteinTokenizer()
    config = load_config(config_path, ProteinLMConfig)
    config.vocab_size = len(tokenizer.vocab)

    model = ProteinConditionalTransformer(config)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd)
    else:
        print(f"[warn] checkpoint path {ckpt_path} not found")

    model.to(device).eval()
    return model, tokenizer, device


def score_protein_sequence(
    model: ProteinConditionalTransformer,
    tokenizer: ProteinTokenizer,
    device: torch.device,
    aa_sequence: str,
    conditions: Optional[List[str]] = None
) -> Dict[str, float]:
    """Computes the log-likelihood of the AA sequence under the ProteinLM.

    Args:
        model: Loaded ProteinConditionalTransformer.
        tokenizer: ProteinTokenizer.
        device: Torch device.
        aa_sequence: Amino acid sequence.
        conditions: Optional condition tokens.

    Returns:
        Dict with total_log_prob, avg_log_prob, and perplexity.
    """
    import torch.nn.functional as F
    
    tokens = [tokenizer.bos_token_id]
    if conditions:
        tokens += tokenizer.encode_conditions(conditions)
    tokens += tokenizer.encode_sequence(aa_sequence)
    tokens.append(tokenizer.eos_token_id)
    
    ids = torch.tensor([tokens], device=device).long()
    
    with torch.no_grad():
        # targets are shifted tokens
        targets = ids[:, 1:].contiguous()
        logits = model(ids[:, :-1]).contiguous()
        
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
        
        total_log_prob = target_log_probs.sum().item()
        avg_log_prob = target_log_probs.mean().item()
        
    return {
        "total_log_prob": total_log_prob,
        "avg_log_prob": avg_log_prob,
        "perplexity": torch.exp(torch.tensor(-avg_log_prob)).item()
    }
