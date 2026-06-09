import numpy as np
import torch
import argparse
from scripts._shared import load_model, resolve_run, load_token_list


# Heuristic DNAshape parameters (Pentamer-based approximations)
def get_theoretical_shape(dna_seq):
    """
    Returns MGW (Angstroms), Roll (Degrees), and EP (kT/e).
    Heuristic:
    - A-tracts: narrow MGW (~3.3), low Roll (~0), low EP (negative/attractive).
    - G/C-rich: wide MGW (~5.8), high Roll (~5), neutral EP.
    """
    mgw, roll, ep, prot, helt = [], [], [], [], []
    for i in range(len(dna_seq)):
        window = dna_seq[max(0, i - 2) : min(len(dna_seq), i + 3)]

        # MGW (Minor Groove Width)
        if "AAAA" in window:
            m_val = 3.5
        elif "GGGG" in window or "CCCC" in window:
            m_val = 5.8
        else:
            m_val = 4.5

        # Roll (Bendability/Step Rotation)
        if "GC" in window or "CG" in window:
            r_val = 5.0
        elif "AA" in window or "TT" in window:
            r_val = 0.0
        else:
            r_val = 2.5

        # EP (Electrostatic Potential)
        if "AAAA" in window:
            e_val = -10.0
        elif "GGCC" in window:
            e_val = -2.0
        else:
            e_val = -5.0

        # ProT (Propeller Twist) - Stiffer base pairing
        if "GC" in window:
            pr_val = -11.0
        elif "AT" in window:
            pr_val = -18.0
        else:
            pr_val = -14.0

        # HelT (Helix Twist) - Degrees per bp
        if "CG" in window:
            h_val = 36.0
        elif "TA" in window:
            h_val = 32.0
        else:
            h_val = 34.0

        mgw.append(m_val)
        roll.append(r_val)
        ep.append(e_val)
        prot.append(pr_val)
        helt.append(h_val)

    return {
        "MGW": np.array(mgw),
        "Roll": np.array(roll),
        "EP": np.array(ep),
        "ProT": np.array(prot),
        "HelT": np.array(helt),
    }


def probe_structural_awareness(run_id, ckpt="best.pt"):
    run_id, run_dir = resolve_run(run_id=run_id)
    model, spec = load_model(run_dir, ckpt_name=ckpt)
    tokens = load_token_list(run_dir)
    stoi = {t: i for i, t in enumerate(tokens)}

    # 1. Prepare Test Data
    test_dna = "ATG" + "GGCC" * 5 + "AAAA" * 5 + "CGTA" * 5 + "TGA"
    codons = [test_dna[i : i + 3] for i in range(0, len(test_dna) - 2, 3)]
    input_ids = torch.tensor([[stoi[c] for c in codons if c in stoi]]).long()

    # 2. Get Hidden States
    with torch.no_grad():
        x = model.tok_emb(input_ids) + model.pos_emb(
            torch.arange(input_ids.size(1)).unsqueeze(0)
        )
        for block in model.blocks:
            x = block(x)
        hidden_states = x.squeeze(0).cpu().numpy()  # (T, D)

    # 3. Calculate Targets
    shape_targets = get_theoretical_shape(test_dna)

    results = {}
    print(f"=== Multi-Dimensional Structural Probe: {run_id} ===")

    for prop_name, values in shape_targets.items():
        # Pool target per codon
        codon_values = []
        for i in range(0, len(values) - 2, 3):
            codon_values.append(values[i : i + 3].mean())

        target = np.array(codon_values[: len(hidden_states)])

        # Linear Correlation Check
        from sklearn.decomposition import PCA

        pca = PCA(n_components=1)
        h_projected = pca.fit_transform(hidden_states).flatten()

        correlation = abs(np.corrcoef(h_projected, target)[0, 1])
        results[prop_name] = correlation

        status = "PASS" if correlation > 0.5 else "WEAK"
        print(f"[{status}] {prop_name}: {correlation:.4f}")

    avg_score = np.mean(list(results.values()))
    print("------------------------------------------")
    print(f"Overall Structural Awareness Score: {avg_score:.4f}")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--ckpt", default="best.pt")
    args = ap.parse_args()
    probe_structural_awareness(args.run_id, args.ckpt)
