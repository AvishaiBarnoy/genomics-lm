from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
import numpy as np
import pytest
import yaml
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

def create_mock_genome(path: Path, genome_id: str, organism: str):
    # Generates a mock genome with a coding sequence
    seq = "A" * 60 + "ATG" + "GCT" * 40 + "TAA" + "C" * 80
    record = SeqRecord(Seq(seq), id="test_rec", name="test", description="mock")
    record.annotations["molecule_type"] = "DNA"
    record.annotations["organism"] = organism
    cds_start = 60
    cds_end = 60 + 3 + (40 * 3) + 3
    record.features.append(
        SeqFeature(
            FeatureLocation(cds_start, cds_end, strand=1),
            type="CDS",
            qualifiers={"locus_tag": [f"mock_{genome_id}"]},
        )
    )
    SeqIO.write(record, path, "genbank")

def test_global_split_and_baselines_end_to_end(tmp_path):
    # Create 3 distinct genomes/gbff files
    gb1 = tmp_path / "GCF_000005845_ecoli.gbff"
    gb2 = tmp_path / "GCF_000240185_kleb.gbff"
    gb3 = tmp_path / "GCF_000006945_salm.gbff"
    
    create_mock_genome(gb1, "000005845", "Escherichia coli")
    create_mock_genome(gb2, "000240185", "Klebsiella pneumoniae")
    create_mock_genome(gb3, "000006945", "Salmonella enterica")

    # Set up config with these 3 datasets
    config_data = {
        "block_size": 128,
        "windows_per_seq": 1,
        "val_frac": 0.33,
        "test_frac": 0.33,
        "datasets": [
            {"name": "ecoli", "gbff": str(gb1), "min_len": 90},
            {"name": "kleb", "gbff": str(gb2), "min_len": 90},
            {"name": "salm", "gbff": str(gb3), "min_len": 90},
        ]
    }
    
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml.dump(config_data))
    
    run_dir = tmp_path / "runs" / "test_global_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Run build_global_manifest script
    cmd = [
        "python",
        "-m",
        "scripts.build_global_manifest",
        "--config",
        str(config_file),
        "--run-id",
        "test_global_run",
        "--run-dir",
        str(run_dir),
        "--group-by",
        "genome",
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Global split failed: {res.stderr}\nStdout: {res.stdout}"
    
    # 2. Verify splits and outputs
    pipeline_json_path = run_dir / "pipeline_prepare.json"
    assert pipeline_json_path.exists()
    
    pipeline_data = json.loads(pipeline_json_path.read_text())
    train_npz = Path(pipeline_data["train_npz"])
    val_npz = Path(pipeline_data["val_npz"])
    test_npz = Path(pipeline_data["test_npz"])
    
    assert train_npz.exists()
    assert val_npz.exists()
    assert test_npz.exists()
    
    # Verify metadata and ensure zero genomic leakage (each split must have mutually exclusive genomes)
    meta_tsv = Path("data/processed/global/test_global_run/cds_meta.tsv")
    assert meta_tsv.exists()
    
    # Read splits and genomes
    split_genomes = {"train": set(), "val": set(), "test": set()}
    with open(meta_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            split_genomes[row["split"]].add(row["genome"])
            
    # Check that splits do not overlap
    assert split_genomes["train"].isdisjoint(split_genomes["val"])
    assert split_genomes["train"].isdisjoint(split_genomes["test"])
    assert split_genomes["val"].isdisjoint(split_genomes["test"])
    
    print(f"Genomes per split: {split_genomes}")

    # 3. Run eval_ppl_baselines script
    cmd_baselines = [
        "python",
        "-m",
        "scripts.eval_ppl_baselines",
        "--train_npz",
        str(train_npz),
        "--test_npz",
        str(test_npz),
        "--vocab_size",
        "69",
    ]
    
    res_baselines = subprocess.run(cmd_baselines, capture_output=True, text=True)
    assert res_baselines.returncode == 0, f"Baselines evaluation failed: {res_baselines.stderr}"
    assert "Baseline Perplexity Comparison" in res_baselines.stdout
    assert "Uniform" in res_baselines.stdout
    assert "Unigram" in res_baselines.stdout
    
    print("Baseline PPL run completed successfully.")

    # 4. Run generate_synonymous_controls script
    cmd_controls = [
        "python",
        "-m",
        "scripts.generate_synonymous_controls",
        "--test_npz",
        str(test_npz),
    ]
    res_controls = subprocess.run(cmd_controls, capture_output=True, text=True)
    assert res_controls.returncode == 0, f"Controls generation failed: {res_controls.stderr}"
    
    # Check outputs exist and have correct shape
    out_dir = test_npz.parent
    control_syn = out_dir / "test_control_synonymous_bs128.npz"
    control_shuf = out_dir / "test_control_codon_shuffle_bs128.npz"
    control_prot = out_dir / "test_control_protein_shuffle_bs128.npz"
    
    assert control_syn.exists()
    assert control_shuf.exists()
    assert control_prot.exists()
    
    with np.load(test_npz) as test_data:
        expected_len = test_data["X"].shape[0]

    with np.load(control_syn) as data:
        assert data["X"].shape == (expected_len, 128)
        
    print("Synonymous controls test completed successfully.")

