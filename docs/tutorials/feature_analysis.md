# Feature Analysis

This module provides tools for discovering and interpreting *feature families* and *semantic clusters* 
from the Sparse Autoencoder's latent space.

## Modules

- `coactivation_graph`: Build feature graphs based on coactivations.
- `family_evaluation`: Evaluate structural quality of extracted families.
- `feature_clustering`: Cluster features based on keyword embeddings.
- `cluster_diagnostics`: Interpret and diagnose feature clusters.
- `visualization`: Plot UMAP projections and coactivation heatmaps.
- `utils`: Auxiliary functions (e.g., matrix normalization, JSON I/O).

---

## Workflow

```python
# 1. Load the coactivation matrix
C = np.load("coactivation_matrix_csLG.npy")
```

```python
# 2. Build coactivation graph
from sae_feature_analysis import coactivation_graph

G = coactivation_graph.build_coactivation_graph(C)
DG = coactivation_graph.build_max_spanning_digraph(G, np.diag(C))
```

```python
# 3. Extract families
families = coactivation_graph.extract_feature_families(DG)
```

```python
# 4. Evaluate families
from sae_feature_analysis import family_evaluation

metrics = family_evaluation.evaluate_feature_families(C, families)
top_families = family_evaluation.sort_families_by_block_ratio(metrics)
```

```python
# 5. Load keywords and cluster features
from sae_feature_analysis import feature_clustering

keywords = load_json("feature_keywords.json")
embeddings = feature_clustering.compute_weighted_feature_embeddings(keywords)
labels = feature_clustering.cluster_embeddings(embeddings)
```
```python
# 6. Plot UMAP and heatmaps
from sae_feature_analysis import visualization

visualization.plot_umap_scatter(embeddings, labels)
```












