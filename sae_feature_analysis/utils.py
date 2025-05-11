import numpy as np


def normalize_coactivation_matrix(C: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalizes a coactivation matrix using the geometric mean of diagonal values.

    Parameters:
    - C: Raw coactivation matrix (n_features x n_features)
    - eps: Small value to avoid division by zero

    Returns:
    - Normalized matrix (same shape as C)
    """
    diag = np.diag(C)
    D = np.sqrt(np.outer(diag, diag))
    D[D == 0] = eps
    C_norm = C / D
    np.fill_diagonal(C_norm, 1.0)
    return C_norm


def get_inactive_features(C: np.ndarray) -> np.ndarray:
    """
    Returns indices of features that never activate (row/col sums are zero).

    Parameters:
    - C: Coactivation matrix (n_features x n_features)

    Returns:
    - Array of feature indices with zero total coactivation
    """
    return np.where(C.sum(axis=0) == 0)[0]


def save_json(obj, path: str, indent: int = 2):
    """
    Saves a Python object to a JSON file.

    Parameters:
    - obj: Serializable Python object
    - path: Destination file path
    - indent: Indentation level for pretty printing
    """
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)


def load_json(path: str):
    """
    Loads a JSON file into a Python object.

    Parameters:
    - path: Path to JSON file

    Returns:
    - Loaded object (usually list or dict)
    """
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
