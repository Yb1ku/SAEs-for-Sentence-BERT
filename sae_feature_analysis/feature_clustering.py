import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
from typing import List, Tuple, Dict


def compute_weighted_feature_embeddings(
    keyword_data: List[Dict],
    model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
    top_k: int = 3
) -> np.ndarray:
    """
    Computes weighted SBERT embeddings for each feature based on its top keywords.

    Parameters:
    - keyword_data: List of dicts with 'keywords' as List[Tuple[str, float]]
    - model_name: HuggingFace model name
    - top_k: Number of keywords to use per feature

    Returns:
    - Array of shape (n_features, embedding_dim)
    """
    model = SentenceTransformer(model_name)
    feature_embeddings = []

    for entry in keyword_data:
        top_keywords = entry["keywords"][:top_k]
        words = [w for w, _ in top_keywords]
        scores = np.array([s for _, s in top_keywords])
        weights = scores / (scores.sum() + 1e-8)

        embeddings = model.encode(words)
        weighted_embedding = np.average(embeddings, axis=0, weights=weights)
        feature_embeddings.append(weighted_embedding)

    return np.array(feature_embeddings)


def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    metric: str = "euclidean"
) -> np.ndarray:
    """
    Applies HDBSCAN to group feature embeddings into semantic clusters.

    Parameters:
    - embeddings: Feature embeddings of shape (n_features, dim)
    - min_cluster_size: Minimum number of features per cluster
    - metric: Distance metric to use

    Returns:
    - Array of cluster labels (length = n_features)
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
    labels = clusterer.fit_predict(embeddings)
    return labels
