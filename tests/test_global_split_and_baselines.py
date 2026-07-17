from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import yaml
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from scripts.build_global_manifest import resolve_genome_identity


def create_mock_genome(
    path: Path,
    genome_id: str,
    organism: str,
    *,
    record_id: str | None = None,
    cds_count: int = 1,
):
    # Generates a mock genome with a coding sequence
    seq = "A" * 60 + "ATG" + "GCT" * 40 + "TAA" + "C" * 80
    record = SeqRecord(
        Seq(seq), id=record_id or f"record_{genome_id}", name="test", description="mock"
    )
    record.annotations["molecule_type"] = "DNA"
    record.annotations["organism"] = organism
    cds_start = 60
    cds_end = 60 + 3 + (40 * 3) + 3
    for cds_idx in range(cds_count):
        record.features.append(
            SeqFeature(
                FeatureLocation(cds_start, cds_end, strand=1),
                type="CDS",
                qualifiers={"locus_tag": [f"mock_{genome_id}_{cds_idx}"]},
            )
        )
    SeqIO.write(record, path, "genbank")


def _run_global_builder(
    config_file: Path,
    run_dir: Path,
    output_dir: Path,
    *extra_args: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python",
            "-m",
            "scripts.build_global_manifest",
            "--config",
            str(config_file),
            "--run-id",
            run_dir.name,
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
            "--group-by",
            "genome",
            *extra_args,
        ],
        capture_output=True,
        text=True,
    )


def _write_config(path: Path, genomes: list[Path], **overrides) -> None:
    config = {
        "block_size": 128,
        "windows_per_seq": 1,
        "val_frac": 0.33,
        "test_frac": 0.33,
        "datasets": [
            {"name": f"genome_{idx}", "gbff": str(genome), "min_len": 90}
            for idx, genome in enumerate(genomes)
        ],
    }
    config.update(overrides)
    path.write_text(yaml.safe_dump(config))


def test_resolve_genome_identity_prefers_explicit_config(tmp_path):
    gbff = tmp_path / "genomic.gbff"
    record = SeqRecord(Seq("ATG"), id="NC_000001.1")

    genome_id, source = resolve_genome_identity(
        {"genome_id": "GCF_123456789.1"}, gbff, record
    )

    assert genome_id == "GCF_123456789.1"
    assert source == "config.genome_id"


def test_resolve_genome_identity_uses_parent_accession_for_generic_filename(tmp_path):
    gbff = tmp_path / "GCF_000005845.2_ASM584v2" / "genomic.gbff"
    record = SeqRecord(Seq("ATG"), id="NC_000001.1")

    genome_id, source = resolve_genome_identity({}, gbff, record)

    assert genome_id == "GCF_000005845.2"
    assert source == "path_accession"


def test_group_split_fails_closed_with_too_few_groups(tmp_path):
    genomes = [tmp_path / f"GCF_00000000{i}.1.gbff" for i in range(2)]
    for idx, genome in enumerate(genomes):
        create_mock_genome(genome, str(idx), f"Genus{idx} species", cds_count=2)
    config = tmp_path / "config.yaml"
    _write_config(config, genomes)

    result = _run_global_builder(
        config, tmp_path / "run", tmp_path / "processed"
    )

    assert result.returncode != 0
    assert "at least 3 distinct genome groups" in (result.stderr + result.stdout)


def test_sequence_fallback_requires_explicit_flag_and_is_marked_non_scientific(tmp_path):
    genomes = [tmp_path / f"GCF_00000000{i}.1.gbff" for i in range(2)]
    for idx, genome in enumerate(genomes):
        create_mock_genome(genome, str(idx), f"Genus{idx} species", cds_count=2)
    config = tmp_path / "config.yaml"
    _write_config(config, genomes)
    run_dir = tmp_path / "run"

    result = _run_global_builder(
        config,
        run_dir,
        tmp_path / "processed",
        "--allow-sequence-split",
    )

    assert result.returncode == 0, result.stderr
    manifest_path = Path(json.loads((run_dir / "pipeline_prepare.json").read_text())["combined_manifest"])
    manifest = json.loads(manifest_path.read_text())
    assert manifest["split_policy"]["effective_group_by"] == "sequence"
    assert manifest["split_policy"]["scientific_valid"] is False


def test_global_packing_is_deterministic_for_same_seed(tmp_path):
    genomes = [tmp_path / f"GCF_00000000{i}.1.gbff" for i in range(3)]
    for idx, genome in enumerate(genomes):
        create_mock_genome(genome, str(idx), f"Genus{idx} species")
    config = tmp_path / "config.yaml"
    _write_config(config, genomes)

    outputs = []
    for suffix in ("a", "b"):
        run_dir = tmp_path / f"run_{suffix}"
        output_dir = tmp_path / f"processed_{suffix}"
        result = _run_global_builder(config, run_dir, output_dir, "--seed", "2027")
        assert result.returncode == 0, result.stderr
        prep = json.loads((run_dir / "pipeline_prepare.json").read_text())
        outputs.append(prep)

    for split_key in ("train_npz", "val_npz", "test_npz"):
        with np.load(outputs[0][split_key]) as first, np.load(outputs[1][split_key]) as second:
            assert first.files == second.files
            for key in first.files:
                assert np.array_equal(first[key], second[key])

    first_manifest = json.loads(Path(outputs[0]["combined_manifest"]).read_text())
    second_manifest = json.loads(Path(outputs[1]["combined_manifest"]).read_text())
    for key in ("seed", "split_policy", "genome_sources", "packing"):
        assert first_manifest[key] == second_manifest[key]


def test_global_builder_rejects_identity_collisions_across_files(tmp_path):
    genomes = []
    for dirname in ("first", "second", "third"):
        directory = tmp_path / dirname
        directory.mkdir()
        genome = directory / "genomic.gbff"
        create_mock_genome(
            genome, dirname, f"Genus{dirname} species", record_id="NC_DUPLICATE.1"
        )
        genomes.append(genome)
    config = tmp_path / "config.yaml"
    _write_config(config, genomes)

    result = _run_global_builder(
        config, tmp_path / "run", tmp_path / "processed"
    )

    assert result.returncode != 0
    assert "Genome identity collision" in (result.stderr + result.stdout)

def test_global_split_and_baselines_end_to_end(tmp_path):
    # Create 3 distinct genomes/gbff files
    gb1 = tmp_path / "GCF_000005845_ecoli.gbff"
    gb2 = tmp_path / "GCF_000240185_kleb.gbff"
    gb3 = tmp_path / "GCF_000006945_salm.gbff"
    
    create_mock_genome(gb1, "000005845", "Escherichia coli")
    create_mock_genome(gb2, "000240185", "Klebsiella pneumoniae")
    create_mock_genome(gb3, "000006945", "Salmonella enterica")

    # Set up config with these 3 datasets
    config_file = tmp_path / "test_config.yaml"
    _write_config(config_file, [gb1, gb2, gb3])
    
    run_dir = tmp_path / "runs" / "test_global_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir = tmp_path / "processed" / "test_global_run"
    res = _run_global_builder(config_file, run_dir, output_dir)
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
    meta_tsv = output_dir / "cds_meta.tsv"
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


def test_main_dry_run_uses_global_builder(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "trainer": "codon_lm",
                "datasets": [
                    {"name": "placeholder", "gbff": "does-not-need-to-exist.gbff"}
                ],
            }
        )
    )

    result = subprocess.run(
        ["bash", str(repo_root / "main.sh"), "--config", str(config), "--dry-run"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(repo_root),
            "RUN_ID": "test-global-dry-run",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "python -m scripts.build_global_manifest" in result.stdout
    assert "python -m scripts.pipeline_prepare" not in result.stdout


def test_legacy_per_dataset_pipeline_requires_explicit_opt_in(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"datasets": []}))

    result = subprocess.run(
        [
            "python",
            "-m",
            "scripts.pipeline_prepare",
            "--config",
            str(config),
            "--run-id",
            "legacy",
            "--run-dir",
            str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--allow-legacy-per-dataset-split" in (result.stderr + result.stdout)
