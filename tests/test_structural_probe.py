import numpy as np
from scripts.probe_structural_awareness import get_theoretical_shape

def test_structural_heuristic_values():
    # Test A-tract (Narrow MGW, Low Roll)
    dna_a = "AAAAAAAAAA"
    shapes = get_theoretical_shape(dna_a)

    assert shapes["MGW"][5] == 3.5
    assert shapes["Roll"][5] == 0.0
    assert shapes["EP"][5] == -10.0

    # Test GC-rich (Wide MGW, High Roll)
    dna_gc = "GGGGGGGGGG"
    shapes_gc = get_theoretical_shape(dna_gc)

    assert shapes_gc["MGW"][5] == 5.8
    assert shapes_gc["Roll"][5] == 2.5 # baseline for non-step

    # Test CG steps (High Roll)
    dna_cg = "CGCGCGCGCG"
    shapes_cg = get_theoretical_shape(dna_cg)
    assert shapes_cg["Roll"][5] == 5.0

def test_structural_output_shapes():
    dna = "ATGC" * 10 # 40 bp
    shapes = get_theoretical_shape(dna)

    expected_keys = [
        "MGW", "Roll", "EP", "ProT", "HelT",
        "Slide", "Rise", "Shift", "Tilt",
        "Buckle", "Opening", "Shear", "Stagger", "Stretch"
    ]
    for key in expected_keys:
        assert len(shapes[key]) == 40
        assert isinstance(shapes[key], np.ndarray)

def test_regression_probe_alignment():
    # Verify alignment between hidden states and pooled shape targets
    # 1. Create a dummy sequence
    dna_seq = "ATG" + "AAAA" * 5 + "GGCC" * 5 + "CGTA" * 5 + "TGA"
    # Pool shape targets per codon
    targets = get_theoretical_shape(dna_seq)

    # 2. Mock model hidden states
    T = len(dna_seq) // 3 # number of codons
    D = 16
    hidden_states = np.random.randn(T, D)

    pooled_targets = {}
    for name, values in targets.items():
        codon_values = []
        for i in range(0, len(values) - 2, 3):
            codon_values.append(values[i : i + 3].mean())
        pooled_targets[name] = np.array(codon_values[:T])

        # Verify length alignment
        assert len(pooled_targets[name]) == len(hidden_states)

    # 3. Fit Ridge Regression
    for name in pooled_targets:
        from sklearn.linear_model import Ridge
        clf = Ridge(alpha=1.0)
        clf.fit(hidden_states, pooled_targets[name])
        preds = clf.predict(hidden_states)
        assert len(preds) == T


def test_shape_baselines_end_to_end(tmp_path):
    import json
    import torch
    import subprocess
    
    # Create a mock run directory with meta.json, itos.txt and weights.pt
    run_dir = tmp_path / "runs" / "test_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    itos_path = run_dir / "itos.txt"
    CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
    SPECIALS = ["<PAD>", "<BOS_CDS>", "<EOS_CDS>", "<SEP>"]
    VOCAB = SPECIALS + CODONS
    itos_path.write_text("\n".join(VOCAB) + "\n")
    
    meta = {
        "model_spec": {
            "model_type": "tiny_gpt",
            "vocab_size": len(VOCAB),
            "block_size": 32,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 4
        }
    }
    (run_dir / "meta.json").write_text(json.dumps(meta))
    
    from src.codonlm.model_tiny_gpt import TinyGPT
    model = TinyGPT(vocab_size=len(VOCAB), block_size=32, n_layer=1, n_head=1, n_embd=4)
    torch.save(model.state_dict(), run_dir / "weights.pt")
    
    test_npz = tmp_path / "test_set.npz"
    X_test = np.random.randint(4, len(VOCAB), size=(3, 32))
    X_test[:, 0] = 1
    X_test[:, -1] = 2
    np.savez(test_npz, X=X_test)
    packing_metadata = tmp_path / "test_packing.tsv"
    packing_metadata.write_text(
        "split\twindow_index\twindow_token_start\twindow_token_end\tsource_id\n"
        + "".join(
            f"test\t{index}\t0\t32\tgene-{index}\n" for index in range(3)
        )
    )
    output_prefix = tmp_path / "shape_results"
    
    cmd = [
        "python",
        "-m",
        "scripts.eval_shape_baselines",
        "--run_dir", str(run_dir),
        "--ckpt", "weights.pt",
        "--test_npz", str(test_npz),
        "--packing-metadata", str(packing_metadata),
        "--group-by", "gene",
        "--n-splits", "3",
        "--output-prefix", str(output_prefix),
        "--max_seqs", "3"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Script failed: {res.stderr}\nStdout: {res.stdout}"
    assert "Representation" in res.stdout
    assert "local_5mer" in res.stdout
    assert "pretrained" in res.stdout
    report = json.loads(output_prefix.with_suffix(".json").read_text())
    assert report["group_by"] == "gene"
    assert len(report["group_assignments"]) == 3
    assert output_prefix.with_suffix(".folds.tsv").exists()
