"""阶段 7 — 验证与稳健性。

跨试件泛化 (当前空白, 务必做): 在部分试件上 fit, 用 transform()/approximate_predict()
投影其余试件, 检查簇标签一致性。
消融: 超参敏感性、有/无去噪、特征版 vs 时频版。汇总"本方法 vs 基线"对比表。
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from . import clustering as clu
from . import embedding as emb
from . import features as feat
from .dataset import AEDataset
from .utils import get_logger

LOG = get_logger()


def _core_run(ds: AEDataset, cfg: dict) -> dict:
    """轻量核心: 特征 -> UMAP -> HDBSCAN, 返回关键指标 (用于消融, 关掉稳定性/基线)。"""
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("embedding", {}).setdefault("stability", {})["enable"] = False
    fs = feat.build_features(ds, cfg)
    er = emb.fit_umap(fs.Xz, cfg)
    cr = clu.cluster_hdbscan(er.Z, cfg)
    return {
        "n_features": len(fs.feature_names),
        "n_clusters": cr.n_clusters,
        "noise_ratio": cr.noise_ratio,
        "silhouette": cr.metrics.get("silhouette", float("nan")),
        "davies_bouldin": cr.metrics.get("davies_bouldin", float("nan")),
    }


# ---------------------------------------------------------------------------
# 跨试件泛化
# ---------------------------------------------------------------------------
def cross_specimen(ds: AEDataset, cfg: dict) -> pd.DataFrame:
    """留一法 (或按 n_train 划分): 训练集 fit, 投影测试试件, 测一致性。"""
    import hdbscan

    specimens = list(pd.unique(ds.events["specimen_id"]))
    if len(specimens) < 2:
        LOG.warning("仅 %d 个试件, 跳过跨试件验证 (需 >=2)", len(specimens))
        return pd.DataFrame()

    vcfg = cfg.get("validation", {}).get("cross_specimen", {})
    n_train = vcfg.get("n_train_specimens")
    rows = []
    # 留一法: 每个试件轮流作测试集
    for test_sp in specimens:
        train_sp = [s for s in specimens if s != test_sp]
        if n_train:
            train_sp = train_sp[:n_train]
        train_ds = ds.select(ds.events["specimen_id"].isin(train_sp).to_numpy())
        test_ds = ds.select((ds.events["specimen_id"] == test_sp).to_numpy())
        if len(train_ds) < 10 or len(test_ds) < 5:
            continue

        run_cfg = copy.deepcopy(cfg)
        run_cfg.setdefault("embedding", {}).setdefault("stability", {})["enable"] = False
        fs = feat.build_features(train_ds, run_cfg)
        er = emb.fit_umap(fs.Xz, run_cfg)
        cr = clu.cluster_hdbscan(er.Z, run_cfg)

        # 投影测试试件: 同一特征流程 -> reducer.transform -> approximate_predict
        Xz_test = feat.transform_features(test_ds, run_cfg, fs)
        Z_test = er.reducer.transform(Xz_test)
        test_labels, strengths = hdbscan.approximate_predict(cr.clusterer, Z_test)

        assigned = test_labels != -1
        rows.append({
            "test_specimen": test_sp,
            "n_train": len(train_ds),
            "n_test": len(test_ds),
            "train_clusters": cr.n_clusters,
            "test_assigned_ratio": float(assigned.mean()),
            "test_mean_strength": float(np.mean(strengths)) if len(strengths) else float("nan"),
            "test_clusters_seen": int(len(set(test_labels[assigned]))),
        })
        LOG.info("跨试件[test=%s]: 指派率=%.1f%%, 命中簇=%d/%d",
                 test_sp, 100 * rows[-1]["test_assigned_ratio"],
                 rows[-1]["test_clusters_seen"], cr.n_clusters)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 消融
# ---------------------------------------------------------------------------
def ablation(ds: AEDataset, cfg: dict) -> pd.DataFrame:
    """超参/分支/去噪消融对照表。"""
    variants: list[tuple[str, dict]] = []

    # 特征分支
    has_wfm = ds.has_waveforms
    branches = ["param", "waveform", "both"] if has_wfm else ["param"]
    for b in branches:
        variants.append((f"branch={b}", {"features.branch": b}))
    # 去噪开关
    variants.append(("bandpass=off", {"preprocess.bandpass.enable": False}))
    # scaler
    variants.append(("scaler=standard", {"features.scaler": "standard"}))
    # n_neighbors 敏感性
    for nn in (5, 30, 50):
        variants.append((f"n_neighbors={nn}", {"embedding.n_neighbors": nn}))

    rows = []
    for name, overrides in variants:
        run_cfg = copy.deepcopy(cfg)
        for dotted, val in overrides.items():
            _set_dotted(run_cfg, dotted, val)
        # 去噪消融需要在预处理阶段重跑 (这里 ds 已是预处理后的, 故仅对未滤波分支近似)
        try:
            res = _core_run(ds, run_cfg)
            res = {"variant": name, **res}
        except Exception as e:  # 某些极端配置在小数据上可能失败
            res = {"variant": name, "error": str(e)[:80]}
        rows.append(res)
        LOG.info("消融 %-20s -> %s", name,
                 {k: v for k, v in res.items() if k != "variant"})
    return pd.DataFrame(rows)


def _set_dotted(cfg: dict, dotted: str, value) -> None:
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


# ---------------------------------------------------------------------------
# 方法 vs 基线 汇总
# ---------------------------------------------------------------------------
def method_comparison(cr: clu.ClusterResult) -> pd.DataFrame:
    """把 HDBSCAN 与各基线整理成一张定量对比表。"""
    rows = [{
        "method": "HDBSCAN (UMAP)",
        "n_clusters": cr.n_clusters,
        "noise_ratio": cr.noise_ratio,
        "silhouette": cr.metrics.get("silhouette", float("nan")),
        "davies_bouldin": cr.metrics.get("davies_bouldin", float("nan")),
    }]
    for name, res in (cr.baselines or {}).items():
        if not res:
            continue
        rows.append({
            "method": name,
            "n_clusters": res.get("k", float("nan")),
            "noise_ratio": 0.0,
            "silhouette": res.get("silhouette", float("nan")),
            "davies_bouldin": res.get("davies_bouldin", float("nan")),
        })
    return pd.DataFrame(rows)
