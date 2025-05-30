import numpy as np
import json
import networkx as nx

def load_coactivation_matrix(path: str, eps: float = 1e-6):
    """Carga la matriz de coactivación bruta y devuelve también la versión normalizada."""
    C = np.load(path)
    activation_freq = np.diag(C)
    C_norm = C / (activation_freq[:, None] + eps)
    return C, C_norm

def load_keyword_scores(path: str, top_k: int = 3):
    """Carga el archivo JSON de keywords y extrae el score del top-1 por feature."""
    with open(path, "r", encoding="utf-8") as f:
        keywords = json.load(f)

    for entry in keywords:
        entry["keywords"] = entry["keywords"][:top_k]

    scores = np.array([
        kw["keywords"][0][1] if kw["keywords"] else 0.0
        for kw in keywords
    ])
    return scores

def extract_feature_families(C_norm: np.ndarray,
                             activation_freq: np.ndarray,
                             keyword_scores: np.ndarray,
                             tau: float = 0.1,
                             percentile: float = 75.0):
    """Extrae familias jerárquicas de features a partir de coactivación y scores."""

    threshold = np.percentile(keyword_scores, percentile)
    interpretable_features = np.where(keyword_scores >= threshold)[0]

    C_sub = C_norm[np.ix_(interpretable_features, interpretable_features)]
    adj_matrix = np.where(C_sub > tau, C_sub, 0)

    G = nx.from_numpy_array(adj_matrix)
    G.remove_edges_from(nx.selfloop_edges(G))

    if G.number_of_edges() == 0:
        return []

    MST = nx.maximum_spanning_tree(G, weight="weight")

    DG = nx.DiGraph()
    act_counts = activation_freq[interpretable_features]
    for u, v, data in MST.edges(data=True):
        if act_counts[u] >= act_counts[v]:
            DG.add_edge(u, v, weight=data["weight"])
        else:
            DG.add_edge(v, u, weight=data["weight"])

    families = []
    for node in DG.nodes:
        if DG.in_degree(node) == 0:
            fam = list(nx.dfs_preorder_nodes(DG, source=node))
            if len(fam) > 1:
                families.append(fam)

    families_global = [[interpretable_features[i] for i in fam] for fam in families]
    return families_global

def clean_families(families):
    """Convierte índices np.int64 a int y elimina duplicados internos."""
    cleaned = []
    for fam in families:
        fam_int = sorted(set(int(i) for i in fam))
        cleaned.append(fam_int)
    return cleaned

def deduplicate_families(families, threshold=0.6):
    """Elimina familias con solapamiento alto."""
    final_families = []
    seen = []

    for fam in families:
        f_set = set(fam)
        discard = False
        for prev in seen:
            inter = len(f_set & prev)
            union = len(f_set | prev)
            if inter / union > threshold:
                discard = True
                break
        if not discard:
            final_families.append(fam)
            seen.append(f_set)

    return final_families

