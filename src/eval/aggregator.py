import os
import json
import numpy as np

class ResultsAggregator:
    def __init__(self, run_ids, scores_base_dir="outputs/scores", runs_base_dir="runs"):
        self.run_ids = run_ids
        self.scores_base_dir = scores_base_dir
        self.runs_base_dir = runs_base_dir
        self.metrics = {}

    def load_metrics(self):
        self.metrics = {}
        for run_id in self.run_ids:
            # Try new consolidated layout first
            metrics_path = os.path.join(self.runs_base_dir, run_id, "scores", "metrics.json")
            if not os.path.exists(metrics_path):
                # Fallback to legacy layout
                metrics_path = os.path.join(self.scores_base_dir, run_id, "metrics.json")
                
            if not os.path.exists(metrics_path):
                print(f"Warning: Metrics not found for run {run_id} at {metrics_path}. Skipping.")
                continue
            with open(metrics_path, "r") as f:
                self.metrics[run_id] = json.load(f)
        return self.metrics


    def get_run_details(self, run_id):
        """Loads meta.json and log.txt for a specific run."""
        run_dir = os.path.join(self.runs_base_dir, run_id)
        if not os.path.exists(run_dir):
            raise FileNotFoundError(f"Run directory not found for {run_id}")
            
        details = {"meta": {}, "log": ""}
        
        meta_path = os.path.join(run_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                details["meta"] = json.load(f)
                
        log_path = os.path.join(run_dir, "log.txt")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                details["log"] = f.read()
                
        return details

    def get_artifacts(self, run_id):
        artifacts_path = os.path.join(self.runs_base_dir, run_id, "artifacts.npz")
        if not os.path.exists(artifacts_path):
            raise FileNotFoundError(f"Artifacts not found for run {run_id} at {artifacts_path}")
        return np.load(artifacts_path, allow_pickle=True)
