from pathlib import Path
import csv
from scripts.plot_eval_prefix import main as plot_main
import sys


def test_plot_prefix_summary(tmp_path: Path, monkeypatch):
    # Create a small summary.csv
    out_dir = tmp_path / "gen_prefix"
    out_dir.mkdir(parents=True, exist_ok=True)
    summ = out_dir / "summary.csv"
    with summ.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k","median_gqs","mean_aa_identity","median_gqs_norm"])
        writer.writeheader()
        writer.writerow({"k":1, "median_gqs": 10.0, "mean_aa_identity": 0.2, "median_gqs_norm": 0.05})
        writer.writerow({"k":3, "median_gqs": 12.0, "mean_aa_identity": 0.3, "median_gqs_norm": 0.04})
    # Run plotting
    argv = ["--summary", str(summ), "--out_dir", str(tmp_path / "figs")]
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(sys, "argv", ["plot_eval_prefix"] + argv)
    plot_main()
    assert (tmp_path / "figs" / "gqs_identity_vs_k.png").exists()

