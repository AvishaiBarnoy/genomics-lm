import pytest
import numpy as np
from pathlib import Path
from src.codonlm.extract_genomic_tape import extract_tapes
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

def test_extract_tapes_logic(tmp_path):
    # Create a mock GenBank file with a single chromosome
    seq = "ATGCGT" * 100 # 600 bp
    record = SeqRecord(Seq(seq), id="test_genome", name="test", description="mock")
    record.annotations["molecule_type"] = "DNA"
    gb_path = tmp_path / "test.gb"
    SeqIO.write(record, gb_path, "genbank")
    
    out_txt = tmp_path / "tape.txt"
    out_meta = tmp_path / "meta.tsv"
    
    with open(out_txt, "w") as ft, open(out_meta, "w") as fm:
        count = extract_tapes(gb_path, window_bp=30, stride_bp=30, out_f=ft, out_m=fm, genome_id="test")
        
    # Total bp = 600. Window = 30. Expected = 20 segments.
    assert count == 20
    
    # Verify content of first segment
    lines = out_txt.read_text().splitlines()
    assert len(lines) == 20
    assert lines[0] == "ATGCGT" * 5

def test_extract_tapes_filtering(tmp_path):
    # Sequence with 'N' (should be skipped)
    seq = "ATGCGT" * 5 + "NNNNN" + "ATGCGT" * 5 # 30 + 5 + 30 = 65 bp
    record = SeqRecord(Seq(seq), id="test_n", name="test", description="mock")
    record.annotations["molecule_type"] = "DNA"
    gb_path = tmp_path / "test_n.gb"
    SeqIO.write(record, gb_path, "genbank")
    
    out_txt = tmp_path / "tape_n.txt"
    out_meta = tmp_path / "meta_n.tsv"
    
    with open(out_txt, "w") as ft, open(out_meta, "w") as fm:
        # Window size 30. First window [0:30] is clean. 
        # Second window [30:60] contains Ns.
        count = extract_tapes(gb_path, window_bp=30, stride_bp=30, out_f=ft, out_m=fm, genome_id="test")
        
    assert count == 1 # Only the first window should be saved
