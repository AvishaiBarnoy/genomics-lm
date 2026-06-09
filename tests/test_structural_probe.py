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
    
    for key in ["MGW", "Roll", "EP", "ProT", "HelT"]:
        assert len(shapes[key]) == 40
        assert isinstance(shapes[key], np.ndarray)
