import argparse
import json
from pathlib import Path
from difflib import SequenceMatcher


def string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def rank_model_motifs(run_id: str):
    run_dir = Path("runs") / run_id
    motif_report_path = run_dir / "motif_mining" / "motif_report.md"
    library_path = Path("src/eval/known_motifs.json")

    if not motif_report_path.exists():
        print(f"[!] Motif report not found for {run_id}. Run mine_motifs.py first.")
        return

    if not library_path.exists():
        # Auto-generate if missing
        from src.eval.known_motifs import save_library

        save_library()

    with open(library_path, "r") as f:
        known_motifs = json.load(f)

    # Extract discovered consensus sequences from the report
    discovered = []
    content = motif_report_path.read_text()
    for line in content.splitlines():
        if line.startswith("**Consensus:**"):
            # Format: **Consensus:** `SEQ`
            seq = line.split("`")[1].replace("<pad>", "").replace(" ", "")
            if seq:
                discovered.append(seq)

    # Compare each known motif against all discovered motifs
    results = []
    total_score = 0

    for name, data in known_motifs.items():
        known_seq = data["sequence"]
        best_match = 0
        best_seq = ""

        for d_seq in discovered:
            # Check for substrings or high similarity
            # Since motifs can be short, we check if known is in discovered or vice versa
            if known_seq in d_seq or d_seq in known_seq:
                score = 1.0
            else:
                score = string_similarity(known_seq, d_seq)

            if score > best_match:
                best_match = score
                best_seq = d_seq

        results.append(
            {
                "known_motif": name,
                "consensus": known_seq,
                "best_match_discovered": best_seq,
                "score": round(best_match, 3),
            }
        )
        total_score += best_match

    # Final Ranking Metric: Average similarity across known motifs
    avg_score = total_score / len(known_motifs)

    # Save Benchmark results
    benchmark = {
        "run_id": run_id,
        "biological_recall_score": round(avg_score, 4),
        "matches": results,
    }

    out_path = run_dir / "motif_mining" / "biological_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(benchmark, f, indent=4)

    print(f"\n=== Biological Benchmark: {run_id} ===")
    print(f"Overall Recall Score: {benchmark['biological_recall_score']}")
    for r in results:
        print(
            f"  - {r['known_motif']}: {r['score']} (Match: {r['best_match_discovered']})"
        )
    print("==========================================")
    print(f"[save] {out_path}")

    # Add to shared comparison summary if possible
    return benchmark


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", help="Run ID to benchmark")
    args = ap.parse_args()
    rank_model_motifs(args.run_id)
