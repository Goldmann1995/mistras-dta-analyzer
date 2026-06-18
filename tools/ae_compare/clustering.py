"""Clusterers and internal-metric evaluation shared by every method.

Two clusterers are run on every latent space as an internal cross-check:

    KMeans     swept over k (default 2..5), best k chosen by silhouette
    HDBSCAN    density-based, picks its own number of clusters, -1 = noise

Each (method, clusterer) configuration is run ``n_runs`` times and the internal
metrics are reported as mean +/- std (KMeans is seeded per run; HDBSCAN is
deterministic so its std is 0).
"""

import numpy as np

from .metrics import internal_metrics, external_metrics


def _common_dim(latent, dim):
    """PCA-reduce a latent to ``dim`` so all methods are compared at equal width.

    M1 has only ~12 native dims; deep latents are wider. Projecting everyone to
    the same dimension removes dimensionality as a confound (dim<=0 disables)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    Z = StandardScaler().fit_transform(latent)
    if dim and dim > 0 and Z.shape[1] > dim:
        Z = PCA(dim, random_state=42).fit_transform(Z)
    return Z.astype(np.float32)


def run_kmeans(Z, k_min, k_max, seed):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    best = None
    for k in range(k_min, k_max + 1):
        if k >= Z.shape[0]:
            break
        lab = KMeans(k, random_state=seed, n_init=10).fit_predict(Z)
        if len(set(lab)) < 2:
            continue
        s = silhouette_score(Z, lab)
        if best is None or s > best[0]:
            best = (s, k, lab)
    if best is None:
        return None, None
    return best[2], best[1]


def run_hdbscan(Z, min_cluster_size, min_samples):
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:
        return None, None
    lab = HDBSCAN(min_cluster_size=max(min_cluster_size, 2),
                  min_samples=min_samples).fit_predict(Z)
    k = len(sorted(l for l in set(lab) if l >= 0))
    return lab, k


def evaluate_method(latent, cfg, labels_true=None):
    """Cluster one method's latent with both clusterers over n_runs.

    Returns a dict keyed by clusterer name, each with mean/std metrics, the
    representative (first-run) labels, and chosen k.
    """
    Z = _common_dim(latent, cfg.common_dim)
    out = {}

    # ---- KMeans (swept k, multi-seed) ----
    runs, rep_labels, rep_k = [], None, None
    for r in range(cfg.n_runs):
        lab, k = run_kmeans(Z, cfg.k_min, cfg.k_max, seed=42 + r)
        if lab is None:
            continue
        m = internal_metrics(Z, lab)
        if labels_true is not None:
            m.update(external_metrics(labels_true, lab))
        runs.append(m)
        if rep_labels is None:
            rep_labels, rep_k = lab, k
    out["kmeans"] = _aggregate(runs, rep_labels, rep_k, Z)

    # ---- HDBSCAN (deterministic) ----
    lab, k = run_hdbscan(Z, cfg.min_cluster_size, cfg.min_samples)
    if lab is not None and len(set(l for l in lab if l >= 0)) >= 2:
        m = internal_metrics(Z, lab)
        if labels_true is not None:
            m.update(external_metrics(labels_true, lab))
        out["hdbscan"] = _aggregate([m], lab, k, Z)
    else:
        out["hdbscan"] = _aggregate([], lab if lab is not None else np.full(len(Z), -1), k, Z)

    return out, Z


def _aggregate(runs, rep_labels, rep_k, Z):
    keys = ["silhouette", "davies_bouldin", "calinski_harabasz", "ari", "nmi"]
    agg = {"k": rep_k, "labels": rep_labels,
           "n_clusters": (len(set(l for l in rep_labels if l >= 0))
                          if rep_labels is not None else 0),
           "n_noise": int(np.sum(rep_labels == -1)) if rep_labels is not None else 0}
    for key in keys:
        vals = [r[key] for r in runs if key in r and r[key] is not None
                and not (isinstance(r[key], float) and np.isnan(r[key]))]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
        else:
            agg[f"{key}_mean"] = None
            agg[f"{key}_std"] = None
    return agg
