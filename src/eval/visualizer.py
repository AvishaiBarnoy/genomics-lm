import numpy as np
import os
import pandas as pd
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
                elif 'token_embeddings' in artifacts:
                    emb = artifacts['token_embeddings']
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

    def plot_attention_entropy(self, figsize=(10, 6)):
        """
        Plots the average attention entropy per layer for each run.
        Returns the matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        has_data = False
        for run_id in self.aggregator.run_ids:
            try:
                artifacts = self.aggregator.get_artifacts(run_id)
                if 'attn' not in artifacts:
                    continue
                
                # attn shape: (layers, batch, heads, T, T)
                attn = artifacts['attn']
                if len(attn.shape) != 5:
                    continue
                
                # Avoid log(0)
                attn = np.clip(attn, 1e-10, 1.0)
                
                # H = -sum(p * log(p)) over the last dim (attention weights)
                entropy = -np.sum(attn * np.log(attn), axis=-1)
                
                # Average over batch, heads, and tokens
                avg_entropy = np.mean(entropy, axis=(1, 2, 3))
                
                layers = np.arange(len(avg_entropy))
                ax.plot(layers, avg_entropy, marker='o', label=run_id)
                has_data = True
                
            except Exception as e:
                print(f"Warning: Could not compute attention entropy for {run_id}: {e}")
        
        if not has_data:
            print("No attention data available for entropy plotting.")
            return fig

        ax.set_title("Average Attention Entropy per Layer")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Entropy (nats)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_saliency_comparison(self, figsize=(12, 6)):
        """
        Plots saliency scores comparison for all runs.
        Returns the matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        has_data = False
        for run_id in self.aggregator.run_ids:
            try:
                saliency_path = os.path.join(self.aggregator.runs_base_dir, run_id, "tables", "saliency.csv")
                if not os.path.exists(saliency_path):
                    continue
                
                df = pd.read_csv(saliency_path)
                ax.plot(df['position'], df['saliency'], label=run_id, alpha=0.8)
                has_data = True
                
            except Exception as e:
                print(f"Warning: Could not load saliency for {run_id}: {e}")
        
        if not has_data:
            print("No saliency data available for plotting.")
            return fig

        ax.set_title("Saliency Scores Comparison")
        ax.set_xlabel("Position")
        ax.set_ylabel("Saliency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig