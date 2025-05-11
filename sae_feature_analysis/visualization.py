import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from math import ceil, sqrt
from typing import Dict, List, Optional, Set


def plot_umap_scatter(embeddings: np.ndarray, labels: np.ndarray, title: str = None):
    """
    Plots a 2D UMAP scatter plot of feature embeddings colored by cluster labels.

    Parameters:
    - embeddings: Array of shape (n_features, dim)
    - labels: Array of cluster labels (same length as embeddings)
    - title: Optional title for the plot
    """
    reducer = umap.UMAP()
    embedding_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        embedding_2d[:, 0], embedding_2d[:, 1],
        c=labels, cmap="tab10", s=10
    )
    plt.colorbar(scatter)
    plt.title(title or "UMAP projection of feature clusters")
    plt.tight_layout()
    plt.show()


def plot_cluster_coactivation_heatmaps(
    C: np.ndarray,
    features_by_cluster: Dict[int, List[int]],
    inactive_features: Optional[Set[int]] = None,
    normalize: bool = True,
    max_clusters: Optional[int] = None,
    max_features: int = 30,
    cmap: str = "viridis"
):
    """
    Visualizes the coactivation heatmap for each cluster in a grid of subplots.

    Parameters:
    - C: Coactivation matrix (n_features x n_features)
    - features_by_cluster: Mapping cluster_id → list of feature indices
    - inactive_features: Optional set of features to exclude
    - normalize: Whether to normalize the submatrices
    - max_clusters: Maximum number of clusters to plot
    - max_features: Max number of features per cluster in heatmap
    - cmap: Colormap to use in heatmaps
    """
    inactive_features = inactive_features or set()
    cluster_ids = sorted(features_by_cluster.keys())

    if max_clusters is not None:
        cluster_ids = cluster_ids[:max_clusters]

    n = len(cluster_ids)
    cols = ceil(sqrt(n))
    rows = ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i, cluster_id in enumerate(cluster_ids):
        feature_ids = [
            fid for fid in features_by_cluster[cluster_id]
            if fid not in inactive_features
        ]
        if len(feature_ids) == 0:
            continue

        feature_ids = feature_ids[:max_features]
        submatrix = C[np.ix_(feature_ids, feature_ids)]

        if normalize:
            diag = np.diag(submatrix)
            D = np.sqrt(np.outer(diag, diag))
            D[D == 0] = 1e-8
            submatrix = submatrix / D
            np.fill_diagonal(submatrix, 1.0)

        ax = axes[i]
        sns.heatmap(
            submatrix,
            cmap=cmap,
            square=True,
            cbar=True,
            ax=ax
        )
        ax.set_title(f"Cluster {cluster_id} ({len(feature_ids)} features)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
