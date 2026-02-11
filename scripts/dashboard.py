import argparse
import sys
from tabulate import tabulate
from src.eval.aggregator import ResultsAggregator

def format_metrics_table(metrics_dict):
    """Converts a dictionary of metrics into a formatted string table."""
    headers = ["Run ID", "Val Loss", "Perplexity"]
    rows = []
    for run_id, m in metrics_dict.items():
        rows.append([
            run_id,
            m.get("val_loss", m.get("best_val_loss", "N/A")),
            m.get("last_perplexity", "N/A")
        ])
    return tabulate(rows, headers=headers, tablefmt="grid")

def main():
    parser = argparse.ArgumentParser(description="Experiment Comparison Dashboard")
    parser.add_argument("--runs", type=str, help="Comma-separated list of run IDs to compare")
    parser.add_argument("--scores_dir", type=str, default="outputs/scores", help="Base directory for scores")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Base directory for run artifacts")
    parser.add_argument("--export", action="store_true", help="Export a full Markdown report with plots")
    parser.add_argument("--out_dir", type=str, help="Directory to save the exported report")
    
    args = parser.parse_args()
    
    if not args.runs:
        print("Error: No run IDs provided. Use --runs run_1,run_2")
        sys.exit(1)
        
    run_ids = [r.strip() for r in args.runs.split(",")]
    
    aggregator = ResultsAggregator(
        run_ids=run_ids,
        scores_base_dir=args.scores_dir,
        runs_base_dir=args.runs_dir
    )
    
    try:
        metrics = aggregator.load_metrics()
        print("\n--- Experiment Comparison Dashboard ---")
        print(format_metrics_table(metrics))
        
        if args.export:
            from src.eval.visualizer import Visualizer
            visualizer = Visualizer(aggregator)
            report_path = visualizer.export_report(output_dir=args.out_dir)
            print(f"\nFull report exported to: {report_path}")
            
    except Exception as e:
        print(f"Error loading metrics: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
