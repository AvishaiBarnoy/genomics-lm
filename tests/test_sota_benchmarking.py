import json
from pathlib import Path
import pytest
from scripts.prepare_sota_benchmarks import main as prepare_benchmarks
from scripts.benchmark_zero_shot_mutations import main as run_zero_shot
from scripts.benchmark_gene_essentiality import main as run_essentiality
from scripts.generate_sota_report import main as run_report
import sys

def test_sota_benchmarking_pipeline(monkeypatch, tmp_path):
    # 1. Setup mock run directory with a meta.json and config
    run_id = "2026-06-06_stage2.5_6L4H_d256_e20"
    # We will point to the real run but override the output dir or run it on the existing run to verify end-to-end
    real_run_dir = Path("runs") / run_id

    if not real_run_dir.exists():
        pytest.skip(f"Real run {run_id} not available for testing")

    ckpt_dir = real_run_dir / "checkpoints"
    if not ckpt_dir.exists() or not any(ckpt_dir.glob("*.pt")):
        pytest.skip(f"Checkpoints for {run_id} not available for testing")
    # Verify that the benchmark CSV files exist
    bench_dir = Path("data/benchmarks")
    assert (bench_dir / "protein_dms.csv").exists()
    assert (bench_dir / "rrna_dms.csv").exists()
    assert (bench_dir / "kosuri_promoters.csv").exists()
    assert (bench_dir / "lambda_essentiality.csv").exists()
    assert (bench_dir / "pseudomonas_essentiality.csv").exists()

    # 2. Test Zero-Shot mutation scoring execution
    monkeypatch.setattr(sys, "argv", [
        "benchmark_zero_shot_mutations.py",
        "--run_id", run_id,
        "--device", "cpu"
    ])
    run_zero_shot()

    metrics_path = real_run_dir / "scores" / "metrics.json"
    assert metrics_path.exists()
    with open(metrics_path) as f:
        metrics = json.load(f)
    assert "sota_protein_dms_spearman" in metrics
    assert "sota_rrna_dms_spearman" in metrics

    # 3. Test Gene Essentiality execution
    monkeypatch.setattr(sys, "argv", [
        "benchmark_gene_essentiality.py",
        "--run_id", run_id,
        "--device", "cpu"
    ])
    run_essentiality()

    with open(metrics_path) as f:
        metrics = json.load(f)
    assert "sota_lambda_essentiality_f1" in metrics
    assert "sota_pseudomonas_essentiality_f1" in metrics

    # 4. Test SOTA Report generation
    monkeypatch.setattr(sys, "argv", [
        "generate_sota_report.py",
        "--run_id", run_id
    ])
    run_report()

    report_path = real_run_dir / "SOTA_BENCHMARK_REPORT.md"
    assert report_path.exists()
    report_content = report_path.read_text()
    assert "SOTA Prokaryotic Benchmarking & Efficiency Report" in report_content
    assert "Our Model (TinyGPT)" in report_content
