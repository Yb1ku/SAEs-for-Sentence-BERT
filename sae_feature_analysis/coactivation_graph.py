import numpy as np
import networkx as nx
from typing import List


def build_coactivation_graph(C: np.ndarray, threshold: float = 0.1, eps: float = 1e-6) -> nx.Graph:
    """
    Constructs an undirected coactivation graph based on the normalized coactivation matrix.

    Parameters:
    - C: Coactivation matrix of shape (n_features, n_features)
    - threshold: Minimum normalized coactivation to consider an edge
    - eps: Small value to avoid division by zero

    Returns:
    - G: Undirected NetworkX graph with weighted edges
    """
    activation_freq = np.diag(C)
    C_norm = C / (activation_freq[:, None] + eps)

    G = nx.Graph()
    n = C.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            weight = C_norm[i, j]
            if weight > threshold:
                G.add_edge(i, j, weight=weight)

    return G


def build_max_spanning_digraph(G: nx.Graph, activation_freq: np.ndarray) -> nx.DiGraph:
    """
    Builds a directed graph from the maximum spanning tree of an undirected graph.
    Parent-child direction is based on activation frequency (higher → lower).

    Parameters:
    - G: Undirected coactivation graph
    - activation_freq: Array with activation frequency per feature (e.g. np.diag(C))

    Returns:
    - DG: Directed graph where edges point from parent to child
    """
    mst = nx.maximum_spanning_tree(G, weight='weight')
    DG = nx.DiGraph()

    for u, v, d in mst.edges(data=True):
        if activation_freq[u] >= activation_freq[v]:
            DG.add_edge(u, v, weight=d['weight'])
        else:
            DG.add_edge(v, u, weight=d['weight'])

    return DG


def extract_feature_families(graph: nx.DiGraph, iterations: int = 3) -> List[List[int]]:
    """
    Extracts hierarchical feature families by performing DFS on disconnected roots.

    Parameters:
    - graph: Directed graph where root nodes have no incoming edges
    - iterations: Number of times to remove roots and repeat DFS extraction

    Returns:
    - all_families: List of families (each one is a list of feature indices)
    """
    all_families = []
    G_copy = graph.copy()

    for _ in range(iterations):
        roots = [n for n in G_copy.nodes if G_copy.in_degree(n) == 0]
        for root in roots:
            family = list(nx.dfs_preorder_nodes(G_copy, source=root))
            if len(family) > 1:
                all_families.append(family)
        G_copy.remove_nodes_from(roots)

    return all_families
