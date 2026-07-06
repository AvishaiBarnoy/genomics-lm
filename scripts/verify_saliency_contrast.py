import torch
import numpy as np
import yaml
import os
from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.models_multi import MultiTaskProteinClassifier
from src.protein_lm.config import ProteinClassifierConfig

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protein_lm/critic_bidirectional_attention.yaml")
    ap.add_argument("--ckpt", default="runs/2026-07-05_critic_bidirectional_attention/checkpoints/best_critic.pt")
    args = ap.parse_args()
    
    config_path = args.config
    ckpt_path = args.ckpt
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    tokenizer = ProteinTokenizer()
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state.get("model_state_dict", state)
    state_dict = state_dict.get("model", state_dict)
    
    # Infer task dims
    task_dims = {}
    if "heads.family.weight" in state_dict:
        task_dims["family"] = state_dict["heads.family.weight"].shape[0]
    if "heads.function.weight" in state_dict:
        task_dims["function"] = state_dict["heads.function.weight"].shape[0]
    if "heads.stability.weight" in state_dict:
        task_dims["stability"] = state_dict["heads.stability.weight"].shape[0]
        
    model_cfg = ProteinClassifierConfig(
        vocab_size=len(tokenizer.vocab),
        block_size=cfg.get("block_size", 512),
        n_layer=cfg.get("n_layer", 4),
        n_head=cfg.get("n_head", 4),
        n_embd=cfg.get("n_embd", 128),
        dropout=0.0,
        num_classes=0,
        pooling=cfg.get("pooling", "mean"),
        bidirectional=cfg.get("bidirectional", True),
    )
    
    model = MultiTaskProteinClassifier(model_cfg, task_dims)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # tRNA Synthetase sequence with conserved YIHIG (idx 34) and KMSKS (idx 374) active site motifs
    seq = "MITLYNTLTRQKEVFKPIEPGKVKMYVCGPTVYNYIHIGNARPAINYDVVRRYFEYQGYNVEYVSNFTDVDDKLIKRSQELNQSVPEIAEKYIAAFHEDVGALNVRKATSNPRVMDHMDDIIQFIKDLVDQGYAYESGGDVYFRTRKFEGYGKLSHQSIDDLKVGARIDAGEHKEDALDFTLWKKAKPGEISWDSPFGEGRPGWHIECSVMAFHELGPTIDIHAGGSDLQFPHHENEIAQSEAHNHAPFANYWMHNGFINIDNEKMSKSLGNFILVHDIIKEVDPDVLRFFMISVHYRSPINYNLELVESARSGLERIRNSYQLIEERAQIATNIENQQTYIDQIDAILNRFETVMNDDFNTANAITAWYDLAKLANKYVLENTTSTEVIDKFKAVYQIFSDVLGVPLKSKNADELLDEDVEKLIEERNEARKNKDFARADEIRDMLKSQNIILEDTPQGVRFKRG"
    
    # Locate indices
    idx_yihig = seq.find("YIHIG")
    idx_kmsks = seq.find("KMSKS")
    
    print(f"[*] Found YIHIG at index: {idx_yihig}")
    print(f"[*] Found KMSKS at index: {idx_kmsks}")
    
    # Encode
    input_ids = [tokenizer.bos_token_id] + tokenizer.encode_sequence(seq) + [tokenizer.eos_token_id]
    input_ids = torch.tensor([input_ids]).long()
    
    with torch.no_grad():
        out = model(input_ids)
        
    attn = out["attention_weights"].squeeze(0).numpy()
    
    # Offset indices by +1 due to BOS token
    yihig_indices = list(range(idx_yihig + 1, idx_yihig + 1 + len("YIHIG")))
    kmsks_indices = list(range(idx_kmsks + 1, idx_kmsks + 1 + len("KMSKS")))
    active_indices = yihig_indices + kmsks_indices
    
    non_active_indices = [i for i in range(1, len(seq) + 1) if i not in active_indices]
    
    mean_active = attn[active_indices].mean()
    mean_non_active = attn[non_active_indices].mean()
    
    ratio = mean_active / mean_non_active
    
    print("\n=== Attention Saliency Contrast Verification (tRNA Synthetase) ===")
    print(f"Mean attention weight on Active Sites (YIHIG + KMSKS): {mean_active:.6f}")
    print(f"Mean attention weight on other residues: {mean_non_active:.6f}")
    print(f"Attention Saliency Contrast Ratio: {ratio:.4f}x")
    
    # Check if ratio satisfies validation threshold
    status = "SUCCESS" if ratio >= 2.0 else "WEAK"
    print(f"Status: {status} (Required >= 2.0x)")
    
    # Let's print the top 5 residues with highest attention weights to see if they are biochemically relevant
    sorted_indices = np.argsort(-attn)
    print("\nTop 10 highest attention residues:")
    for rank, idx in enumerate(sorted_indices[:10]):
        if idx == 0:
            token_name = "<BOS>"
        elif idx == len(attn) - 1:
            token_name = "<EOS>"
        else:
            token_name = f"{seq[idx-1]} (pos {idx})"
        print(f"  Rank {rank+1}: {token_name} -> score {attn[idx]:.6f}")

if __name__ == "__main__":
    main()
