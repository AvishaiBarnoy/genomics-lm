import numpy as np
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.decomposition import PCA

class MotifClusterer:
    """
    Handles clustering of motif embeddings using various algorithms.
    """
    def __init__(self, method='kmeans', n_clusters=10, pca_components=None, random_state=42):
        """Initializes the MotifClusterer with a clustering method and hyperparameters."""
        self.method = method
        self.n_clusters = n_clusters
        self.pca_components = pca_components
        self.random_state = random_state
        self.model = None
        self.pca = None
        self.labels = None

    def fit_predict(self, embeddings):
        """
        Cluster the provided embeddings.
        
        Args:
            embeddings (np.ndarray): (N, D) array of window embeddings.
            
        Returns:
            np.ndarray: Cluster labels for each embedding.
        """
        X = embeddings
        if self.pca_components:
            # If n_components > dimensions, PCA handles it or we can cap it
            n_comp = min(self.pca_components, X.shape[1], X.shape[0])
            self.pca = PCA(n_components=n_comp, random_state=self.random_state)
            X = self.pca.fit_transform(X)
            
        if self.method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_clusters, 
                n_init='auto', 
                random_state=self.random_state
            )
        elif self.method == 'hdbscan':
            # Note: HDBSCAN doesn't use n_clusters, but min_cluster_size
            self.model = HDBSCAN(min_cluster_size=max(2, self.n_clusters))
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
            
        self.labels = self.model.fit_predict(X)
        return self.labels

    def get_centers(self, embeddings):
        """
        Get the centroid for each cluster.
        """
        if self.method == 'kmeans':
            return self.model.cluster_centers_
        else:
            # For non-centroid methods, compute mean of members in original space
            centers = []
            unique_labels = sorted(set(self.labels))
            for label in unique_labels:
                if label == -1:
                    continue # Skip noise in HDBSCAN
                mask = (self.labels == label)
                centers.append(embeddings[mask].mean(axis=0))
            return np.array(centers)

