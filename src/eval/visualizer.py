import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, aggregator):
        self.aggregator = aggregator

    def compute_pca(self, n_components=2):
        """
        Computes PCA for embeddings of all runs.
        Returns a dictionary mapping run_id to transformed embeddings.
        """
        pca_results = {}
        
        for run_id in self.aggregator.run_ids:
            try:
                artifacts = self.aggregator.get_artifacts(run_id)
                if 'embeddings' in artifacts:
                    emb = artifacts['embeddings']
                elif 'h_avg' in artifacts:
                    emb = artifacts['h_avg']
                else:
                    continue
                
                # If embeddings are 3D (N, T, D), average over T for visualization
                if len(emb.shape) == 3:
                    emb = np.mean(emb, axis=1)
                
                if len(emb.shape) != 2:
                    print(f"Skipping PCA for {run_id}: Embeddings shape {emb.shape} not supported (expected 2D or 3D).")
                    continue
                
                pca = PCA(n_components=n_components)
                transformed = pca.fit_transform(emb)
                pca_results[run_id] = transformed
                
            except Exception as e:
                print(f"Warning: Could not compute PCA for {run_id}: {e}")
                
        return pca_results

    def plot_pca_comparison(self, figsize=(12, 6)):
        """
        Plots PCA comparison for all runs side-by-side.
        Returns the matplotlib Figure object.
        """
        pca_data = self.compute_pca()
        n_runs = len(pca_data)
        
        if n_runs == 0:
            print("No data available for PCA plotting.")
            return plt.figure()

        fig, axes = plt.subplots(1, n_runs, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        for i, (run_id, data) in enumerate(pca_data.items()):
            ax = axes[i]
            ax.scatter(data[:, 0], data[:, 1], alpha=0.6, s=10)
            ax.set_title(f"PCA: {run_id}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        return fig