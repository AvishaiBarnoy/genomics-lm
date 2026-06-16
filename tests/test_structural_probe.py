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
