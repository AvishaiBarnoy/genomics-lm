import torch
import copy
from pathlib import Path

def main():
    print("[*] Loading checkpoints...")
    helical_path = Path("runs/separate_heads_full_run/checkpoints/best.pt")
    strand_path = Path("runs/separate_heads_v2/checkpoints/best.pt")
    
    if not helical_path.exists():
        raise FileNotFoundError(f"Helical checkpoint missing at {helical_path}")
    if not strand_path.exists():
        raise FileNotFoundError(f"Strand checkpoint missing at {strand_path}")
        
    ckpt_helical = torch.load(helical_path, map_location="cpu")
    ckpt_strand = torch.load(strand_path, map_location="cpu")
    
    # Clone the strand checkpoint as our base
    print("[*] Merging state dicts...")
    ckpt_merged = copy.deepcopy(ckpt_strand)
    
    # Extract the helical projection parameters
    weight_4 = ckpt_helical["model"]["offset_projs.4.weight"]
    bias_4 = ckpt_helical["model"]["offset_projs.4.bias"]
    
    # Insert them into the merged model
    ckpt_merged["model"]["offset_projs.4.weight"] = weight_4
    ckpt_merged["model"]["offset_projs.4.bias"] = bias_4
    
    # Update configuration parameters
    print("[*] Updating config metadata...")
    cfg = ckpt_merged.get("cfg", {})
    cfg["multi_offset_targets"] = [2, 4, 8, 16, 32]
    cfg["multi_offset_weights"] = {2: 0.10, 4: 0.10, 8: 0.05, 16: 0.03, 32: 0.02}
    ckpt_merged["cfg"] = cfg
    
    # Save merged checkpoint
    out_dir = Path("runs/separate_heads_merged/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "best.pt"
    
    torch.save(ckpt_merged, out_path)
    print(f"[+] Merged checkpoint successfully saved to {out_path}")

if __name__ == "__main__":
    main()
