"""Cluster-quality metrics.

Internal metrics (no labels needed) are the headline numbers for the unlabeled
real-data comparison. External metrics (ARI/NMI) are only computable on the
synthetic self-test where ground truth exists, and are used there to verify the
framework actually recovers the known damage modes.
"""

import numpy as np


def internal_metrics(Z, labels):
    """Silhouette (max), Davies-Bouldin (min), Calinski-Harabasz (max).

    Noise points (label -1, from HDBSCAN) are excluded.
    """
    from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                                 calinski_harabasz_score)
    m = {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
    mask = labels >= 0
    valid = sorted(set(labels[mask]))
    if len(valid) >= 2 and np.sum(mask) > len(valid):
        Zv, lv = Z[mask], labels[mask]
        m["silhouette"] = float(silhouette_score(Zv, lv))
        m["davies_bouldin"] = float(davies_bouldin_score(Zv, lv))
        m["calinski_harabasz"] = float(calinski_harabasz_score(Zv, lv))
    return m


def external_metrics(labels_true, labels_pred):
    """Adjusted Rand Index and Normalized Mutual Info (synthetic only)."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    mask = labels_pred >= 0
    if np.sum(mask) < 2:
        return {"ari": None, "nmi": None}
    return {
        "ari": float(adjusted_rand_score(labels_true[mask], labels_pred[mask])),
        "nmi": float(normalized_mutual_info_score(labels_true[mask], labels_pred[mask])),
    }
