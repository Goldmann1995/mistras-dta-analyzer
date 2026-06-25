"""阶段 4 — 聚类。

主引擎: HDBSCAN (跑在 UMAP 嵌入 Z 上, 记录 noise 比例, 不预设簇数)。
基线对比: k-means / GMM(BIC 选 k), 同时在 Z 与 PCA 嵌入上跑。
有效性指标: silhouette、Davies-Bouldin (仅作次要证据)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .utils import get_logger

LOG = get_logger()


@dataclass
class ClusterResult:
    labels: np.ndarray              # HDBSCAN 每事件簇标签 (-1=noise)
    clusterer: object               # 已拟合 HDBSCAN (prediction_data=True)
    noise_ratio: float
    n_clusters: int
    probabilities: np.ndarray
    metrics: dict = field(default_factory=dict)
    baselines: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 有效性指标 (次要证据)
# ---------------------------------------------------------------------------
def validity_metrics(Z: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    mask = labels != -1
    uniq = np.unique(labels[mask])
    if len(uniq) < 2 or mask.sum() < 3:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"),
                "n_evaluated": int(mask.sum())}
    return {
        "silhouette": float(silhouette_score(Z[mask], labels[mask])),
        "davies_bouldin": float(davies_bouldin_score(Z[mask], labels[mask])),
        "n_evaluated": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# 主引擎 HDBSCAN
# ---------------------------------------------------------------------------
def cluster_hdbscan(Z: np.ndarray, cfg: dict) -> ClusterResult:
    import hdbscan

    hcfg = cfg.get("clustering", {}).get("hdbscan", {})
    n = len(Z)
    mcs = int(min(hcfg.get("min_cluster_size", 30), max(2, n // 2)))
    ms = hcfg.get("min_samples")
    ms = int(ms) if ms else None

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        cluster_selection_method=hcfg.get("cluster_selection_method", "eom"),
        prediction_data=True,    # 供 approximate_predict (跨试件)
    )
    labels = clusterer.fit_predict(Z)
    noise_ratio = float((labels == -1).mean())
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    metrics = validity_metrics(Z, labels)

    LOG.info("HDBSCAN: %d 簇, noise=%.1f%%, silhouette=%.3f (min_cluster_size=%d)",
             n_clusters, 100 * noise_ratio, metrics["silhouette"], mcs)
    return ClusterResult(
        labels=labels, clusterer=clusterer, noise_ratio=noise_ratio,
        n_clusters=n_clusters, probabilities=clusterer.probabilities_,
        metrics=metrics)


# ---------------------------------------------------------------------------
# 基线
# ---------------------------------------------------------------------------
def baseline_kmeans(Z: np.ndarray, k_range, seed: int) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best = None
    for k in k_range:
        if k >= len(Z):
            continue
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Z)
        sil = silhouette_score(Z, km.labels_) if len(set(km.labels_)) > 1 else -1
        if best is None or sil > best["silhouette"]:
            best = {"k": k, "labels": km.labels_, "silhouette": float(sil)}
    return best or {}


def baseline_gmm(Z: np.ndarray, k_range, seed: int) -> dict:
    """GMM, 用 BIC 选 k。"""
    from sklearn.mixture import GaussianMixture

    best = None
    for k in k_range:
        if k >= len(Z):
            continue
        gmm = GaussianMixture(n_components=k, random_state=seed,
                              covariance_type="full").fit(Z)
        bic = gmm.bic(Z)
        if best is None or bic < best["bic"]:
            best = {"k": k, "labels": gmm.predict(Z), "bic": float(bic)}
    return best or {}


def run_baselines(Z: np.ndarray, cfg: dict, Xz: np.ndarray | None = None) -> dict:
    """在 Z (UMAP) 与 PCA 嵌入上跑 k-means / GMM 基线。

    Z  : UMAP 嵌入 (主对比对象)。
    Xz : 标准化特征矩阵; 提供时在其 PCA 嵌入上也跑一遍基线 (UMAP vs PCA 对照)。
    """
    bcfg = cfg.get("clustering", {}).get("baselines", {})
    if not bcfg.get("enable", True):
        return {}
    seed = cfg.get("seed", 42)
    kr = bcfg.get("kmeans_k_range", [2, 3, 4, 5, 6])
    gr = bcfg.get("gmm_k_range", [2, 3, 4, 5, 6])

    out = {
        "umap_kmeans": _annotate(Z, baseline_kmeans(Z, kr, seed)),
        "umap_gmm": _annotate(Z, baseline_gmm(Z, gr, seed)),
    }
    if bcfg.get("run_on_pca", True) and Xz is not None:
        from sklearn.decomposition import PCA

        nc = int(min(bcfg.get("pca_components", 5), Xz.shape[0] - 1, Xz.shape[1]))
        Zp = PCA(n_components=nc, random_state=seed).fit_transform(Xz)
        out["pca_kmeans"] = _annotate(Zp, baseline_kmeans(Zp, kr, seed))
        out["pca_gmm"] = _annotate(Zp, baseline_gmm(Zp, gr, seed))
    for name, res in out.items():
        if res:
            LOG.info("基线 %-12s: k=%d, silhouette=%.3f",
                     name, res.get("k", -1), res.get("silhouette", float("nan")))
    return out


def _annotate(Z: np.ndarray, res: dict) -> dict:
    """给基线结果补充 silhouette/DBI 指标。"""
    if not res:
        return res
    res = dict(res)
    res.update(validity_metrics(Z, res["labels"]))
    return res
