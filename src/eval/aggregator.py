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
            metrics_path = os.path.join(self.scores_base_dir, run_id, "metrics.json")
            if not os.path.exists(metrics_path):
                raise FileNotFoundError(f"Metrics not found for run {run_id} at {metrics_path}")
            with open(metrics_path, "r") as f:
                self.metrics[run_id] = json.load(f)
        return self.metrics

    def get_artifacts(self, run_id):
        artifacts_path = os.path.join(self.runs_base_dir, run_id, "artifacts.npz")
        if not os.path.exists(artifacts_path):
            raise FileNotFoundError(f"Artifacts not found for run {run_id} at {artifacts_path}")
        return np.load(artifacts_path, allow_pickle=True)
