"""阶段 5 — 簇 → 损伤机制标定 (论文成败关键)。

算每簇的特征分布 (尤其频率分段), 对照文献频带做初判。
真值交叉印证 (参考试验 / DIC 裂纹 / SEM 断口) 取决于数据是否具备:
本模块给出基于频带 + 力学量的初判, 真值标定留作可插拔接口。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import get_logger

LOG = get_logger()


def cluster_profiles(events: pd.DataFrame, features, labels: np.ndarray) -> pd.DataFrame:
    """每簇的关键特征分布 (中位数/均值/占比)。"""
    df = events.copy()
    df["cluster"] = labels
    # 把若干波形特征并入 (若存在)
    X = features.X.reset_index(drop=True)
    for col in ("w_peak_freq", "w_freq_centroid"):
        if col in X.columns:
            df[col] = X[col].to_numpy()

    rows = []
    for c in sorted(set(labels)):
        sub = df[df["cluster"] == c]
        row = {
            "cluster": int(c),
            "label": "noise" if c == -1 else f"C{c}",
            "n": int(len(sub)),
            "fraction": float(len(sub) / len(df)),
        }
        for col in ("peak_freq", "freq_centroid", "w_peak_freq", "w_freq_centroid",
                    "amp", "abs_energy", "rise_time", "duration", "counts"):
            if col in sub.columns:
                vals = sub[col].astype(float)
                row[f"{col}_median"] = float(vals.median())
                row[f"{col}_mean"] = float(vals.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _pick_freq(profile_row: pd.Series, freq_feature: str) -> tuple[float, str | None]:
    """从 profile 行里取用于分机制的频率 (优先配置项, 退回波形特征)。"""
    for key in (f"{freq_feature}_median", "peak_freq_median",
                "w_peak_freq_median", "freq_centroid_median",
                "w_freq_centroid_median"):
        if key in profile_row and np.isfinite(profile_row[key]):
            return float(profile_row[key]), key
    return float("nan"), None


def map_to_mechanism(profiles: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """按文献频带把每簇映射到候选损伤机制。

    重要: 频带是材料相关的, config 默认给的是 CFRP 示例, 须按实际体系调整;
    无真值时这是初判, 应在论文中明确为局限。
    """
    mcfg = cfg.get("mapping", {})
    bands = mcfg.get("mechanism_bands", [])
    freq_feature = mcfg.get("freq_feature", "peak_freq")

    out = profiles.copy()
    mechanisms, used_freq, used_key = [], [], []
    for _, r in profiles.iterrows():
        if r["cluster"] == -1:
            mechanisms.append("noise"); used_freq.append(float("nan")); used_key.append(None)
            continue
        f, key = _pick_freq(r, freq_feature)
        used_freq.append(f); used_key.append(key)
        name = "未标定 (unassigned)"
        if np.isfinite(f):
            for b in bands:
                if b["low_khz"] <= f < b["high_khz"]:
                    name = b["name"]
                    break
        mechanisms.append(name)
    out["mech_freq_khz"] = used_freq
    out["mech_freq_source"] = used_key
    out["mechanism"] = mechanisms

    if used_key and used_key[0] is None and all(k is None for k in used_key):
        LOG.warning("未找到可用频率特征, 机制标定不可靠 (无频率列)")
    LOG.info("机制初判完成 (基于频带 %s):", freq_feature)
    for _, r in out.iterrows():
        if r["cluster"] != -1:
            LOG.info("  %s: %.0f kHz -> %s (n=%d)",
                     r["label"], r.get("mech_freq_khz", float("nan")),
                     r["mechanism"], r["n"])
    return out


def attach_ground_truth(mapping_df: pd.DataFrame, truth: dict | None) -> pd.DataFrame:
    """可插拔: 若有真值 (簇->机制 人工/参考标定), 并入对照列。

    truth: {cluster_id: mechanism_name}
    """
    if not truth:
        return mapping_df
    out = mapping_df.copy()
    out["mechanism_truth"] = out["cluster"].map(truth)
    out["mech_match"] = out["mechanism_truth"].notna() & (
        out["mechanism_truth"] == out["mechanism"])
    return out
