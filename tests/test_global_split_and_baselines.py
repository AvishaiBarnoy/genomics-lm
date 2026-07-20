from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from scripts.build_global_manifest import resolve_genome_identity
from src.codonlm.codon_tokenize import itos, stoi, to_ids
from src.codonlm.extract_cds_from_genbank import reverse_complement


def create_mock_genome(
    path: Path,
    genome_id: str,
    organism: str,
    *,
    record_id: str | None = None,
    cds_count: int = 1,
    cds_sequence: str | None = None,
):
    # Generates a mock genome with a coding sequence
    if cds_sequence is None:
        digest = hashlib.sha256(genome_id.encode()).digest()
        alanine_codons = ("GCT", "GCC", "GCA", "GCG")
        cds_sequence = "ATG" + "".join(
            alanine_codons[digest[index % len(digest)] % len(alanine_codons)]
            for index in range(40)
        ) + "TAA"
    seq = "A" * 60 + cds_sequence + "C" * 80
    record = SeqRecord(
        Seq(seq), id=record_id or f"record_{genome_id}", name="test", description="mock"
    )
    record.annotations["molecule_type"] = "DNA"
    record.annotations["organism"] = organism
    cds_start = 60
    cds_end = 60 + len(cds_sequence)
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
    skip_homology: bool = True,
) -> subprocess.CompletedProcess[str]:
    audit_args = ["--skip-homology-audit"] if skip_homology else []
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
            *audit_args,
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


def test_reverse_complement_supports_iupac_ambiguity():
    assert reverse_complement("ARYN") == "NRYT"


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
        "--allow-cross-split-exact-duplicates",
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


def test_global_builder_fragments_ambiguity_after_source_split(tmp_path):
    genomes = [tmp_path / f"GCF_00000000{i}.1.gbff" for i in range(3)]
    create_mock_genome(
        genomes[0],
        "0",
        "Genus0 species",
        cds_sequence="ATG" + "GCT" * 20 + "NNN" + "GAA" * 20 + "TAA",
    )
    for idx, genome in enumerate(genomes[1:], start=1):
        create_mock_genome(genome, str(idx), f"Genus{idx} species")
    config = tmp_path / "config.yaml"
    _write_config(config, genomes, min_fragment_codons=2)
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "processed"

    result = _run_global_builder(config, run_dir, output_dir)

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "manifest.json").read_text())
    policy = manifest["tokenization"]["ambiguous_codon_policy"]
    assert policy["name"] == "split"
    assert policy["min_fragment_codons"] == 2
    assert policy["ambiguous_codons"] == 1
    assert policy["source_records_with_ambiguity"] == 1
    assert policy["retained_fragments"] == 4
    assert policy["discarded_fragments"] == 0

    source_splits = {}
    with open(output_dir / "cds_meta.tsv") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            source_splits[row["source_id"]] = row["split"]

    with open(output_dir / "cds_fragments.tsv") as handle:
        fragments = list(csv.DictReader(handle, delimiter="\t"))
    assert len(fragments) == 4
    assert all(row["split"] == source_splits[row["source_id"]] for row in fragments)
    fragment_counts = {
        source_id: sum(row["source_id"] == source_id for row in fragments)
        for source_id in source_splits
    }
    ambiguous_source = next(
        source_id for source_id, count in fragment_counts.items() if count == 2
    )
    ambiguous_fragments = [
        row for row in fragments if row["source_id"] == ambiguous_source
    ]
    assert [(int(row["codon_start"]), int(row["codon_end"])) for row in ambiguous_fragments] == [
        (0, 21),
        (22, 43),
    ]


def _dynamic_sequences(npz_path: Path) -> list[list[int]]:
    with np.load(npz_path) as data:
        flat = data["X"]
        lengths = data["lengths"]
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        return [flat[offsets[i] : offsets[i + 1]].tolist() for i in range(len(lengths))]


def _transition_counts(sequences: list[list[int]]) -> Counter:
    return Counter(
        (tokens[index], tokens[index + 1])
        for tokens in sequences
        for index in range(len(tokens) - 1)
    )


def test_global_dynamic_packing_keeps_starts_ends_and_all_transitions(tmp_path):
    genomes = [tmp_path / f"GCF_00000000{i}.1.gbff" for i in range(3)]
    for idx, genome in enumerate(genomes):
        create_mock_genome(genome, str(idx), f"Genus{idx} species")
    config = tmp_path / "config.yaml"
    _write_config(config, genomes, pack_mode="dynamic", block_size=8)
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "processed"

    result = _run_global_builder(config, run_dir, output_dir)

    assert result.returncode == 0, result.stderr
    sequences_by_split = {}
    with open(output_dir / "cds_meta.tsv") as metadata, open(
        output_dir / "cds_dna.txt"
    ) as dna:
        for row, sequence in zip(csv.DictReader(metadata, delimiter="\t"), dna):
            sequences_by_split[row["split"]] = to_ids(sequence.strip())
    for split in ("train", "val", "test"):
        chunks = _dynamic_sequences(output_dir / f"{split}_bs8.npz")
        assert chunks[0][0] == stoi["<BOS_CDS>"]
        assert chunks[-1][-1] == stoi["<EOS_CDS>"]
        assert _transition_counts(chunks) == _transition_counts(
            [sequences_by_split[split]]
        )
        with open(output_dir / f"{split}_packing.tsv") as handle:
            spans = list(csv.DictReader(handle, delimiter="\t"))
        assert spans[0]["continues_from_previous"] == "0"
        assert spans[-1]["continues_to_next"] == "0"
        assert all(row["split"] == split for row in spans)


def test_global_multi_packing_exposes_multiple_gene_spans(tmp_path):
    genomes = [tmp_path / f"GCF_00000000{i}.1.gbff" for i in range(3)]
    for idx, genome in enumerate(genomes):
        create_mock_genome(
            genome, str(idx), f"Genus{idx} species", cds_count=2
        )
    config = tmp_path / "config.yaml"
    _write_config(config, genomes, pack_mode="multi", block_size=100)
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "processed"

    result = _run_global_builder(config, run_dir, output_dir)

    assert result.returncode == 0, result.stderr
    for split in ("train", "val", "test"):
        with open(output_dir / f"{split}_packing.tsv") as handle:
            spans = list(csv.DictReader(handle, delimiter="\t"))
        assert len(spans) == 2
        assert spans[0]["window_index"] == spans[1]["window_index"] == "0"
        assert spans[0]["source_id"] != spans[1]["source_id"]
        with np.load(output_dir / f"{split}_bs100.npz") as data:
            segment_ids = data["segment_ids"][0]
        assert len(set(segment_ids[segment_ids >= 0])) == 2
        assert np.any(segment_ids == -1)


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


def test_global_builder_blocks_cross_split_exact_cds(tmp_path):
    genomes = [tmp_path / f"GCF_00000000{i}.1.gbff" for i in range(3)]
    duplicate = "ATG" + "GCT" * 40 + "TAA"
    create_mock_genome(genomes[0], "0", "Genus0 species", cds_sequence=duplicate)
    create_mock_genome(genomes[1], "1", "Genus1 species", cds_sequence=duplicate)
    create_mock_genome(
        genomes[2],
        "2",
        "Genus2 species",
        cds_sequence="ATG" + "AAA" * 40 + "TAA",
    )
    config = tmp_path / "config.yaml"
    _write_config(config, genomes)
    output_dir = tmp_path / "processed"

    result = _run_global_builder(config, tmp_path / "run", output_dir)

    assert result.returncode != 0
    report = json.loads((output_dir / "leakage_audit.json").read_text())
    assert report["status"] == "failed"
    assert report["exact_duplicates"]["count"] == 1
    assert report["blocking_reasons"] == ["cross_split_exact_duplicates"]


def test_global_builder_blocks_cross_split_protein_clusters(tmp_path):
    from tests.test_leakage_audit import _write_fake_mmseqs

    genomes = [tmp_path / f"GCF_00000000{i}.1.gbff" for i in range(3)]
    create_mock_genome(
        genomes[0], "0", "Genus0 species", cds_sequence="ATG" + "GCT" * 40 + "TAA"
    )
    create_mock_genome(
        genomes[1], "1", "Genus1 species", cds_sequence="ATG" + "GCC" * 40 + "TAA"
    )
    create_mock_genome(
        genomes[2], "2", "Genus2 species", cds_sequence="ATG" + "AAA" * 40 + "TAA"
    )
    config = tmp_path / "config.yaml"
    _write_config(config, genomes)
    executable = tmp_path / "mmseqs"
    _write_fake_mmseqs(executable)
    output_dir = tmp_path / "processed"

    result = _run_global_builder(
        config,
        tmp_path / "run",
        output_dir,
        "--mmseqs-executable",
        str(executable),
        skip_homology=False,
    )

    assert result.returncode != 0
    report = json.loads((output_dir / "leakage_audit.json").read_text())
    assert report["status"] == "failed"
    assert report["protein_homology"]["cross_split_cluster_count"] == 1
    assert report["protein_homology"]["tool"]["version"] == "fake-mmseqs-1.0"

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
    assert Path(pipeline_data["itos_path"]) == output_dir / "itos.txt"
    
    assert train_npz.exists()
    assert val_npz.exists()
    assert test_npz.exists()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["vocabulary"]["size"] == len(itos)
    assert manifest["vocabulary"]["token_ids_contiguous"] is True
    
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
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_conda = fake_bin / "conda"
    fake_conda.write_text(
        "#!/bin/sh\n"
        "if [ \"$1 $2\" = \"env list\" ]; then\n"
        "  printf '# conda environments:\\nbase /opt/conda\\n'\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1 $2\" = \"shell.bash hook\" ]; then\n"
        "  printf 'export PATH=/missing-python\\n'\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n"
    )
    fake_conda.chmod(0o755)
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
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
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
