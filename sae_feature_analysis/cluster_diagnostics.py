from collections import defaultdict, Counter
import pandas as pd
from typing import Dict, List, Tuple, Union


def diagnose_cluster(
    cluster_id: int,
    df_clusters: pd.DataFrame,
    top_n_keywords: int = 10,
    sample_size: int = 5,
) -> Dict[str, Union[int, List[Tuple[str, int]], pd.DataFrame]]:
    """
    Prints a diagnostic summary of a cluster based on feature keywords.

    Parameters:
    - cluster_id: ID of the cluster to analyze.
    - df_clusters: DataFrame with columns 'cluster', 'feature_id', and 'top_keywords'.
    - top_n_keywords: Number of top frequent keywords to display.
    - sample_size: Number of feature samples to print.

    Returns:
    - Dictionary with diagnostic info (can be used programmatically).
    """
    subset = df_clusters[df_clusters["cluster"] == cluster_id]
    n_features = len(subset)

    all_keywords = []
    for kws_str in subset["top_keywords"]:
        all_keywords.extend(kws_str.split(", "))

    keyword_counter = Counter(all_keywords)
    most_common = keyword_counter.most_common(top_n_keywords)
    muestras = subset.sample(min(sample_size, n_features))[["feature_id", "top_keywords"]]

    print(f"\n📊 Cluster diagnosis {cluster_id}")
    print(f"- Nº of features: {n_features}")
    print(f"- Most frequent keywords:")
    for kw, count in most_common:
        print(f"  · {kw:<30} ({count} appearances)")
    print("\n🔍 Samples of features in this cluster:")
    print(muestras.to_string(index=False))

    return {
        "cluster_id": cluster_id,
        "num_features": n_features,
        "most_common_keywords": most_common,
        "sample_features": muestras,
    }


def representative_keyword_per_cluster(df_clusters: pd.DataFrame) -> Dict[int, str]:
    """
    Returns the most frequent keyword for each cluster.

    Parameters:
    - df_clusters: DataFrame with 'cluster' and 'top_keywords' columns.

    Returns:
    - Dictionary mapping cluster_id to its representative keyword.
    """
    cluster_keywords = defaultdict(list)

    for _, row in df_clusters.iterrows():
        cluster = row["cluster"]
        if cluster == -1:
            continue  # Ignore noise
        keywords = row["top_keywords"].split(", ")
        cluster_keywords[cluster].extend(keywords)

    return {
        cluster: Counter(kws).most_common(1)[0][0]
        for cluster, kws in cluster_keywords.items()
    }


def get_features_by_cluster(df_clusters: pd.DataFrame) -> Dict[int, List[int]]:
    """
    Groups feature IDs by their cluster.

    Parameters:
    - df_clusters: DataFrame with 'cluster' and 'feature_id' columns.

    Returns:
    - Dictionary mapping cluster_id to list of feature_ids.
    """
    clusters = defaultdict(list)
    for _, row in df_clusters.iterrows():
        cluster_id = row["cluster"]
        if cluster_id == -1:
            continue
        clusters[cluster_id].append(row["feature_id"])
    return dict(clusters)
