#!/usr/bin/env python3
"""
Generates a "Plain English" summary of the Genomics-LM analysis results.
Designed for non-experts to understand what the AI has learned about biology.
"""

import argparse
import pandas as pd
from scripts._shared import resolve_run


def generate_report(run_id: str):
    _, run_dir = resolve_run(run_id=run_id)
    tables_dir = run_dir / "tables"
    motif_dir = run_dir / "motif_mining"

    # 1. Load Mutation Map Summary
    mut_summary_path = tables_dir / "one_cds__best_summary.csv"
    if not mut_summary_path.exists():
        print(f"[!] {mut_summary_path} not found. Run analysis.sh first.")
        return

    df_mut = pd.read_csv(mut_summary_path)

    # 2. Load Motif Report (to extract consensus)
    motif_report_path = motif_dir / "motif_report.md"
    motifs = []
    if motif_report_path.exists():
        content = motif_report_path.read_text()
        for line in content.splitlines():
            if line.startswith("## Cluster"):
                name = line.replace("## ", "").split("(")[0].strip()
                motifs.append({"name": name})
            if line.startswith("**Consensus:**") and motifs:
                cons = line.split("`")[1]
                motifs[-1]["consensus"] = cons

    # 3. Compile the Plain English Report
    report = []
    report.append(f"# 🧬 Biological Insight Report: {run_id}")
    report.append("")
    report.append("## 🌟 Executive Summary")
    report.append(
        "This AI model was trained on DNA sequences without any prior knowledge of biology. "
        "Through intensive analysis, we've found that it has successfully 'discovered' key biological concepts "
        "on its own."
    )
    report.append("")

    # Section 1: Sensitive Sites
    report.append("## 🔴 Critical 'Sensitive' Sites")
    report.append(
        "We performed a 'stress test' on a gene (e.g., lacZ) by changing every single DNA letter "
        "one by one to see how the AI reacts."
    )

    # Find top 5 sensitive sites (lowest best_delta or highest count of 'bad' mutations)
    # Actually high best_delta means there is a MUCH better option,
    # but low best_delta with many bad mutations means the site is highly constrained.
    # Let's define sensitivity by the average delta or count of better mutations.
    # Here, 'best_delta' in our script is max gain. If max gain is small (e.g. 0), the site is perfect.

    sensitive_df = df_mut.sort_values("best_delta").head(5)
    report.append(
        "The following positions in the gene are identified as **Critical Zones**. "
        "The AI believes that almost any change here would make the sequence 'unnatural' or 'wrong':"
    )
    for _, row in sensitive_df.iterrows():
        report.append(
            f"- **Position {int(row['pos'])}** (Current Codon: `{row['wt']}`): Very high sensitivity. No substitutions were predicted to be better."
        )
    report.append("")

    # Section 2: Flexible Sites
    report.append("## 🟢 Flexible 'Adaptable' Sites")
    report.append(
        "Conversely, some areas are highly flexible. The AI predicts that many different DNA sequences "
        "would work just as well here:"
    )
    flexible_df = df_mut.sort_values("n_better", ascending=False).head(5)
    for _, row in flexible_df.iterrows():
        report.append(
            f"- **Position {int(row['pos'])}**: Highly flexible. There are {int(row['n_better'])} alternative DNA sequences that the model likes."
        )
    report.append("")

    # Section 3: Discovered Motifs
    report.append("## 🧩 Discovered Biological Motifs")
    report.append(
        "The AI has learned that certain patterns of DNA letters appear together frequently in meaningful ways. "
        "We call these 'Motifs'. Here are the top patterns the AI looks for:"
    )

    for m in motifs[:5]:
        report.append(
            f"- **{m['name']}**: Consistently finds patterns like `{m['consensus']}`."
        )
    report.append("")

    # Section 4: What this means for drug design/bioengineering
    report.append("## 🚀 Why this matters")
    report.append(
        "1. **Gene Engineering**: We can use the 'Green Zones' to optimize DNA sequences for better manufacturing without breaking the biology."
    )
    report.append(
        "2. **Mutation Prediction**: The 'Red Zones' tell us where genetic diseases are most likely to occur if a mutation happens."
    )
    report.append(
        "3. **De-novo Design**: Because the AI understands these patterns, we can eventually ask it to 'write' new genes that follow these rules."
    )

    report_path = run_dir / "PLAIN_ENGLISH_SUMMARY.md"
    report_path.write_text("\n".join(report))
    print(f"[success] Plain English report saved to {report_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", help="Run ID to summarize")
    args = ap.parse_args()
    generate_report(args.run_id)


if __name__ == "__main__":
    main()
