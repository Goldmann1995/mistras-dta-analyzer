"""阶段 3 — UMAP 嵌入。

输入: 标准化特征 Xz。输出: 低维嵌入 Z + 已拟合 reducer (供 transform 新试件)。
关键: 固定 random_state; 跑多个种子, 用聚类标签一致性 / trustworthiness 评估稳定性
(把"UMAP 不可重复"反转成可量化的卖点)。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .utils import get_logger

LOG = get_logger()


def _umap_fit(reducer, Xz):
    """拟合 UMAP, 屏蔽固定 random_state 时的并行降级提示 (这是我们要的可复现行为)。"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*n_jobs value.*overridden.*")
        return reducer.fit_transform(Xz)


@dataclass
class EmbeddingResult:
    Z: np.ndarray               # 聚类用嵌入 [N x n_components]
    Z_viz: np.ndarray           # 2D 可视化嵌入
    reducer: object             # 已拟合 UMAP (聚类维度), 供 transform()
    reducer_viz: object
    stability: dict


def _make_umap(cfg: dict, n_components: int, n_samples: int, seed: int):
    import umap

    emb = cfg.get("embedding", {})
    # n_neighbors 必须 < n_samples
    n_neighbors = int(min(emb.get("n_neighbors", 15), max(2, n_samples - 1)))
    return umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=float(emb.get("min_dist", 0.0)),
        n_components=int(n_components),
        metric=emb.get("metric", "euclidean"),
        random_state=seed,
    )


def fit_umap(Xz: np.ndarray, cfg: dict, seed: int | None = None) -> EmbeddingResult:
    emb = cfg.get("embedding", {})
    seed = cfg.get("seed", 42) if seed is None else seed
    n = Xz.shape[0]
    n_comp = int(min(emb.get("n_components", 5), max(2, n - 2)))

    reducer = _make_umap(cfg, n_comp, n, seed)
    Z = _umap_fit(reducer, Xz)

    # 2D 可视化嵌入 (单独拟合, 不复用聚类嵌入)
    reducer_viz = _make_umap(cfg, 2, n, seed)
    Z_viz = _umap_fit(reducer_viz, Xz)

    LOG.info("UMAP 嵌入: %s -> %s (n_neighbors=%d, min_dist=%.2f, seed=%d)",
             Xz.shape, Z.shape, reducer.n_neighbors, reducer.min_dist, seed)

    stability = {}
    if emb.get("stability", {}).get("enable", True):
        if n >= 10:
            stability = embedding_stability(Xz, cfg)
        else:
            LOG.warning("样本过少 (%d), 跳过嵌入稳定性评估", n)
    return EmbeddingResult(Z, Z_viz, reducer, reducer_viz, stability)


def embedding_stability(Xz: np.ndarray, cfg: dict) -> dict:
    """多种子嵌入稳定性: trustworthiness 均值/方差 + 跨种子聚类标签一致性 (ARI)。"""
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import adjusted_rand_score

    emb = cfg.get("embedding", {})
    scfg = emb.get("stability", {})
    seeds = scfg.get("seeds", [0, 1, 2, 7, 42])
    n = Xz.shape[0]
    # trustworthiness 要求 n_neighbors < n_samples / 2
    k = int(min(scfg.get("trustworthiness_k", 10), max(1, n // 2 - 1)))
    n_comp = int(min(emb.get("n_components", 5), max(2, n - 2)))

    trusts, label_sets = [], []
    for s in seeds:
        reducer = _make_umap(cfg, n_comp, n, s)
        Z = _umap_fit(reducer, Xz)
        trusts.append(float(trustworthiness(Xz, Z, n_neighbors=k)))
        label_sets.append(_quick_labels(Z, cfg))

    # 跨种子两两 ARI (排除全噪声情形)
    aris = []
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            if len(set(label_sets[i])) > 1 and len(set(label_sets[j])) > 1:
                aris.append(adjusted_rand_score(label_sets[i], label_sets[j]))

    result = {
        "seeds": list(seeds),
        "trustworthiness_mean": float(np.mean(trusts)),
        "trustworthiness_std": float(np.std(trusts)),
        "trustworthiness_per_seed": trusts,
        "label_ari_mean": float(np.mean(aris)) if aris else float("nan"),
        "label_ari_std": float(np.std(aris)) if aris else float("nan"),
    }
    LOG.info("嵌入稳定性: trustworthiness=%.3f±%.3f, 跨种子标签 ARI=%.3f",
             result["trustworthiness_mean"], result["trustworthiness_std"],
             result["label_ari_mean"])
    return result


def _quick_labels(Z: np.ndarray, cfg: dict) -> np.ndarray:
    """在嵌入上快速跑一次 HDBSCAN, 仅用于稳定性评估。"""
    import hdbscan

    hcfg = cfg.get("clustering", {}).get("hdbscan", {})
    mcs = int(min(hcfg.get("min_cluster_size", 30), max(2, len(Z) // 2)))
    cl = hdbscan.HDBSCAN(min_cluster_size=mcs)
    return cl.fit_predict(Z)
