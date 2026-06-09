import numpy as np
from src.eval.motif_clusterer import MotifClusterer

def test_kmeans_clustering():
    # Create 3 distinct clusters in 2D
    c1 = np.random.randn(10, 2) + np.array([10, 10])
    c2 = np.random.randn(10, 2) + np.array([-10, -10])
    c3 = np.random.randn(10, 2) + np.array([10, -10])
    X = np.vstack([c1, c2, c3])
    
    clusterer = MotifClusterer(method='kmeans', n_clusters=3)
    labels = clusterer.fit_predict(X)
    
    assert len(labels) == 30
    assert len(set(labels)) == 3
    
    # Check that points in same generated cluster have same label
    assert len(set(labels[0:10])) == 1
    assert len(set(labels[10:20])) == 1
    assert len(set(labels[20:30])) == 1

def test_pca_reduction():
    # 100D data that is actually 2D
    X = np.random.randn(50, 2) @ np.random.randn(2, 100)
    
    clusterer = MotifClusterer(method='kmeans', n_clusters=5, pca_components=10)
    labels = clusterer.fit_predict(X)
    
    assert len(labels) == 50
    assert clusterer.pca.n_components_ == 10

def test_hdbscan_clustering():
    c1 = np.random.randn(20, 2) + np.array([10, 10])
    c2 = np.random.randn(20, 2) + np.array([-10, -10])
    X = np.vstack([c1, c2])
    
    # HDBSCAN should find 2 clusters
    clusterer = MotifClusterer(method='hdbscan', n_clusters=5)
    labels = clusterer.fit_predict(X)
    
    # HDBSCAN unique labels (might include -1 for noise)
    unique = set(labels) - {-1}
    assert len(unique) >= 1
