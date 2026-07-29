import torch
from pathlib import Path

from scripts import query_model as Q
from scripts.eval_generation_prefix import (
    _extract_frozen_split_cds,
    _bootstrap_interval,
    _codon_to_aa,
    _load_vocab_for_run,
    _model_spec_from,
    _ngram_repeat_ratio,
    _score_stop_behavior,
    _select_device,
    _sample_seed,
    _training_match_coverage,
)


def test_extract_frozen_split_cds_balances_genomes(tmp_path):
    import json

    metadata = tmp_path / "cds_meta.tsv"
    dna = tmp_path / "cds_dna.txt"
    metadata.write_text(
        "line_idx\tsplit\tsource_id\tgenome\n"
        "0\ttest\ta1\tgenome-a\n"
        "1\ttest\ta2\tgenome-a\n"
        "2\ttest\tb1\tgenome-b\n"
        "3\ttest\tb2\tgenome-b\n"
        "4\ttrain\tt1\tgenome-train\n"
    )
    dna.write_text(
        "ATGAAATAA\nATGCCCTAA\nATGGGGTAA\nATGTTTTAA\nATGACGTAA\n"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": {
                    "source_metadata": {"path": "cds_meta.tsv"},
                    "source_dna": {"path": "cds_dna.txt"},
                }
            }
        )
    )
    run_dir = tmp_path / "run"

    selected_path, provenance = _extract_frozen_split_cds(
        run_dir, manifest, "test", max_genes=4, seed=17
    )

    assert len(selected_path.read_text().splitlines()) == 4
    assert provenance["split"] == "test"
    assert provenance["selected_group_counts"] == {
        "genome-a": 2,
        "genome-b": 2,
    }


def test_ngram_repeat_ratio_simple():
    seq = ["ATG", "AAA", "CCC", "ATG", "AAA", "CCC"]
    r = _ngram_repeat_ratio(seq, n=3)
    # two 3-grams repeated once each over 4 total grams: uniq=2, total=4 => repeat ratio=1-2/4=0.5
    assert abs(r - 0.5) < 1e-6


def test_training_match_coverage_counts_covered_positions():
    tokens = [1, 2, 3, 4, 5, 6]
    training = {(1, 2, 3), (3, 4, 5)}

    assert _training_match_coverage(tokens, 3, training) == 5 / 6


def test_protocol_sample_seed_is_stable_and_prompt_specific():
    assert _sample_seed(1337, 4, 8, 2) == _sample_seed(1337, 4, 8, 2)
    assert _sample_seed(1337, 4, 8, 2) != _sample_seed(1337, 4, 8, 3)


def test_bootstrap_interval_is_seeded():
    first = _bootstrap_interval([0.0, 1.0, 1.0], statistic="mean", seed=9, n_resamples=50)
    second = _bootstrap_interval([0.0, 1.0, 1.0], statistic="mean", seed=9, n_resamples=50)
    assert first == second


def test_codon_to_aa_mapping():
    assert _codon_to_aa("ATG") == "M"
    assert _codon_to_aa("TAA") == "Stop"


def test_stop_behavior_scoring():
    # valid end stop, no early stops
    codons = ["ATG", "AAA", "CCC", "TAG"]
    score, valid, early = _score_stop_behavior(codons, truth_len_codons=4)
    assert valid and not early and score == 1.0

    # no end stop; length error 50% => score decays to >=0 but < 1
    codons2 = ["ATG", "AAA"]
    score2, valid2, _ = _score_stop_behavior(codons2, truth_len_codons=4)
    assert not valid2 and 0.0 <= score2 < 1.0


def test_select_device_cpu():
    assert _select_device("cpu") == torch.device("cpu")


def test_model_spec_falls_back_to_checkpoint_cfg():
    ckpt = {
        "cfg": {
            "vocab_size": 69,
            "block_size": 512,
            "n_layer": 10,
            "n_head": 8,
            "n_embd": 384,
        }
    }
    assert _model_spec_from({"model_spec": {}}, ckpt) == {
        "vocab_size": 69,
        "block_size": 512,
        "n_layer": 10,
        "n_head": 8,
        "n_embd": 384,
    }


def test_load_vocab_falls_back_to_cfg_itos_path(tmp_path: Path):
    repo = tmp_path
    run_dir = tmp_path / "runs" / "run_without_root_vocab"
    run_dir.mkdir(parents=True)
    vocab_path = tmp_path / "data" / "processed" / "itos_codon.txt"
    vocab_path.parent.mkdir(parents=True)
    vocab_path.write_text("<PAD>\n<BOS_CDS>\nATG\n")

    itos, stoi = _load_vocab_for_run(
        run_dir,
        repo,
        {"itos_path": "data/processed/itos_codon.txt"},
    )

    assert itos == ["<PAD>", "<BOS_CDS>", "ATG"]
    assert stoi["ATG"] == 2


def test_dna_prefix_to_ids_omits_eos_marker():
    stoi = {"<BOS_CDS>": 1, "<EOS_CDS>": 2, "ATG": 3, "AAA": 4}

    assert Q.dna_to_ids("ATGAAA", stoi) == [1, 3, 4, 2]
    assert Q.dna_prefix_to_ids("ATGAAA", stoi) == [1, 3, 4]


def test_eval_generation_prefix_end_to_end(tmp_path):
    import json
    import torch
    import subprocess
    import csv
    import shutil
    from pathlib import Path

    repo_dir = Path(__file__).resolve().parents[1]
    test_run_dir = repo_dir / "runs" / "test_run_tmp"
    
    if test_run_dir.exists():
        shutil.rmtree(test_run_dir)
        
    try:
        test_run_dir.mkdir(parents=True, exist_ok=True)
        
        # Write mock itos.txt
        itos_path = test_run_dir / "itos.txt"
        CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
        SPECIALS = ["<PAD>", "<BOS_CDS>", "<EOS_CDS>", "<SEP>"]
        VOCAB = SPECIALS + CODONS
        itos_path.write_text("\n".join(VOCAB) + "\n")
        
        # Write mock meta.json
        meta = {
            "model_spec": {
                "model_type": "tiny_gpt",
                "vocab_size": len(VOCAB),
                "block_size": 64,
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 4
            },
            "cfg": {
                "vocab_size": len(VOCAB),
                "block_size": 64,
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 4,
                "dataset_name": "combined_hybrid",
                "train_npz": "runs/test_run_tmp/train_mock.npz"
            }
        }
        (test_run_dir / "meta.json").write_text(json.dumps(meta))
        
        # Write mock best.pt
        from src.codonlm.model_tiny_gpt import TinyGPT
        model = TinyGPT(vocab_size=len(VOCAB), block_size=64, n_layer=1, n_head=1, n_embd=4)
        torch.save(model.state_dict(), test_run_dir / "best.pt")
        
        # Write mock train_mock.npz
        import numpy as np
        X_train = np.random.randint(4, len(VOCAB), size=(5, 16), dtype=np.int64)
        Y_train = np.random.randint(4, len(VOCAB), size=(5, 16), dtype=np.int64)
        np.savez(test_run_dir / "train_mock.npz", X=X_train, Y=Y_train)
        
        # Write mock combined_manifest.json under runs/test_run_tmp/
        # to mock DNA database resolve
        manifest_data = {
            "datasets": [
                {"dna": "runs/test_run_tmp/test_dna.txt"}
            ]
        }
        (test_run_dir / "combined_manifest.json").write_text(json.dumps(manifest_data))
        
        # Write mock DNA file in tmp_path
        dna_path = test_run_dir / "test_dna.txt"
        dna_path.write_text("ATGAACGCGTAG\nATGGGGCCCTAA\n")
        
        # Run prefix generation script
        cmd = [
            "python",
            "-m",
            "scripts.eval_generation_prefix",
            "--run_id", "test_run_tmp",
            "--device", "cpu",
            "--preset", "quick",
            "--samples", "2",
            "--k_list", "1,2",
            "--max_genes", "2",
            "--max_new", "8",
            "--min_aa_len", "2",
            "--target_aa_len", "4",
            "--max_aa_len", "10",
            "--termination_bias",
            "--termination_stop_bias", "1.0",
            "--ci_resamples", "20",
        ]
        
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"Script failed: {res.stderr}\nStdout: {res.stdout}"
        
        # Check that samples.csv and summary.csv are written
        samples_csv = test_run_dir / "scores" / "gen_prefix" / "samples.csv"
        summary_csv = test_run_dir / "scores" / "gen_prefix" / "summary.csv"
        protocol_samples_csv = test_run_dir / "scores" / "gen_prefix" / "protocol_samples.csv"
        protocol_summary_csv = test_run_dir / "scores" / "gen_prefix" / "protocol_summary.csv"
        protocol_manifest = test_run_dir / "scores" / "gen_prefix" / "protocol_manifest.json"
        generated_fasta = test_run_dir / "scores" / "gen_prefix" / "generated_protocols.fasta"
        
        assert samples_csv.exists()
        assert summary_csv.exists()
        assert protocol_samples_csv.exists()
        assert protocol_summary_csv.exists()
        assert protocol_manifest.exists()
        assert generated_fasta.exists()

        with protocol_samples_csv.open() as f:
            protocol_rows = list(csv.DictReader(f))
        assert {row["protocol"] for row in protocol_rows} == {
            "raw_model",
            "cds_constrained",
            "guided",
        }
        grouped_seeds = {}
        for row in protocol_rows:
            key = (row["gene_idx"], row["k"], row["sample_id"])
            grouped_seeds.setdefault(key, set()).add(row["sample_seed"])
        assert all(len(seeds) == 1 for seeds in grouped_seeds.values())
        raw_rows = [row for row in protocol_rows if row["protocol"] == "raw_model"]
        assert all(row["cds_only"] == "False" for row in raw_rows)
        assert all(row["require_terminal_stop"] == "False" for row in raw_rows)
        guided_rows = [row for row in protocol_rows if row["protocol"] == "guided"]
        assert all("termination_bias" in row["guidance_components"] for row in guided_rows)

        with protocol_summary_csv.open() as f:
            protocol_summary_rows = list(csv.DictReader(f))
        assert {row["protocol"] for row in protocol_summary_rows} == {
            "raw_model",
            "cds_constrained",
            "guided",
        }
        assert "median_gqs_ci_low" in protocol_summary_rows[0]
        assert "terminal_stop_rate_ci_high" in protocol_summary_rows[0]

        manifest = json.loads(protocol_manifest.read_text())
        assert set(manifest["protocols"]) == {"raw_model", "cds_constrained", "guided"}
        assert manifest["protocols"]["raw_model"]["full_vocabulary"] is True
        assert manifest["protocols"]["raw_model"]["forced_terminal_stop"] is False
        
        # Verify columns in summary.csv
        with summary_csv.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) > 0
            # Check raw columns are present
            first_row = rows[0]
            assert "raw_median_gqs" in first_row
            assert "raw_mean_aa_len" in first_row
            assert "raw_terminal_stop_rate" in first_row
            assert "train_overlap_10" in first_row
            assert "train_overlap_20" in first_row
            
    finally:
        if test_run_dir.exists():
            shutil.rmtree(test_run_dir)
