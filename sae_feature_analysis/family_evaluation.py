import numpy as np
from typing import List, Dict


def evaluate_feature_families(
    C: np.ndarray,
    families: List[List[int]],
    eps: float = 1e-6
) -> List[Dict]:
    """
    Evaluates a list of feature families using coactivation metrics.

    Parameters:
    - C: Coactivation matrix of shape (n_features, n_features)
    - families: List of families (each a list of feature indices)
    - eps: Small constant to avoid division by zero

    Returns:
    - List of dicts, each containing metrics for one family
    """
    results = []

    for fam in families:
        if len(fam) < 2:
            continue

        parent = fam[0]
        children = fam[1:]

        # Mean parent-child coactivation
        R_pc = np.mean([C[parent, i] for i in children])

        # Mean child-child coactivation
        if len(children) > 1:
            coact_children = [
                C[i, j]
                for i in range(len(children))
                for j in range(i + 1, len(children))
            ]
            R_children = np.mean(coact_children)
        else:
            R_children = 0.0

        # Mean in-block coactivation (within family, excluding self-links)
        block_vals = [
            C[i, j]
            for i in fam
            for j in fam
            if i != j
        ]
        mean_in_block = np.mean(block_vals)

        # Mean off-block coactivation (between family and rest of features)
        all_indices = set(range(C.shape[0]))
        out_indices = list(all_indices - set(fam))
        off_vals = [C[i, j] for i in fam for j in out_indices]
        mean_off_block = np.mean(off_vals)

        # Intra/extra block ratio
        block_ratio = mean_in_block / (mean_off_block + eps)

        results.append({
            "parent": parent,
            "size": len(fam),
            "R_pc": R_pc,
            "R_children": R_children,
            "in_block": mean_in_block,
            "off_block": mean_off_block,
            "block_ratio": block_ratio,
            "members": fam,
        })

    return results


def sort_families_by_block_ratio(metrics: List[Dict], descending: bool = True) -> List[Dict]:
    """
    Sorts a list of evaluated families by their in/out block coactivation ratio.

    Parameters:
    - metrics: List of dicts containing family evaluation results
    - descending: Whether to sort in descending order

    Returns:
    - Sorted list of family metrics
    """
    return sorted(metrics, key=lambda x: x["block_ratio"], reverse=descending)
