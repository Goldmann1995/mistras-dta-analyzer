"""阶段 2 — 特征表征 (分支点)。

分支 A (param):   用 AEWin 已算的 AE 参数 + 派生量 (可解释基线)。
分支 B (waveform): 由原始波形算时频特征 (峰值频率/质心/分频段 partial power 等),
                   可选 CWT 能量分布。
两分支可合并 (both); 之后做相关性冗余筛查, 再标准化并保存 scaler (供新试件复用)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dataset import AEDataset
from .utils import get_logger

LOG = get_logger()


@dataclass
class FeatureSet:
    """特征产物容器。"""
    X: pd.DataFrame              # 原始 (未标准化) 特征表, 行序与 events 对齐
    Xz: np.ndarray              # 标准化后的特征矩阵 [N x F]
    feature_names: list[str]
    scaler: object              # 已拟合的 sklearn scaler (保存供 transform 新试件)
    dropped: list[str]          # 冗余筛查中被删除的列


# ---------------------------------------------------------------------------
# 分支 A — AE 参数特征
# ---------------------------------------------------------------------------
def param_features(ds: AEDataset, cfg: dict) -> pd.DataFrame:
    feat_cfg = cfg.get("features", {})
    cols = feat_cfg.get("param_columns") or ds.param_columns()
    cols = [c for c in cols if c in ds.events.columns
            and pd.api.types.is_numeric_dtype(ds.events[c])]
    X = ds.events[cols].astype(float).copy()

    if feat_cfg.get("derive_param", True):
        ev = ds.events
        eps = 1e-12
        if {"rise_time", "amp"} <= set(ev.columns):
            X["ra_value"] = ev["rise_time"].to_numpy() / (ev["amp"].to_numpy() + eps)
        if {"counts", "duration"} <= set(ev.columns):
            # 平均频率 (kHz): counts / duration(us) * 1e3
            X["avg_freq_derived"] = ev["counts"].to_numpy() / (
                ev["duration"].to_numpy() + eps) * 1e3
        if {"abs_energy", "duration"} <= set(ev.columns):
            X["energy_rate"] = ev["abs_energy"].to_numpy() / (
                ev["duration"].to_numpy() + eps)

    X.columns = [f"p_{c}" for c in X.columns]
    return X


# ---------------------------------------------------------------------------
# 分支 B — 波形时频特征
# ---------------------------------------------------------------------------
def _spectrum_features(wave: np.ndarray, srate: float,
                       bands_khz: list[list[float]]) -> dict:
    n = len(wave)
    wave = wave - np.mean(wave)
    mag = np.abs(np.fft.rfft(wave))
    freqs = np.fft.rfftfreq(n, d=1.0 / srate) / 1e3  # kHz
    power = mag ** 2
    total = power.sum() + 1e-20

    # 跳过 DC 求峰值频率
    if len(power) > 1:
        peak_idx = 1 + int(np.argmax(power[1:]))
    else:
        peak_idx = 0
    centroid = float((freqs * power).sum() / total)
    spread = float(np.sqrt(((freqs - centroid) ** 2 * power).sum() / total))

    out = {
        "w_peak_freq": float(freqs[peak_idx]),
        "w_freq_centroid": centroid,
        "w_freq_spread": spread,
        "w_log_energy": float(np.log10(np.sum(wave ** 2) + 1e-20)),
        "w_rms": float(np.sqrt(np.mean(wave ** 2))),
        "w_peak_amp": float(np.max(np.abs(wave))),
    }
    # 分频段 partial power (占总功率比例) —— 关键可解释特征
    for lo, hi in bands_khz:
        sel = (freqs >= lo) & (freqs < hi)
        out[f"w_pp_{int(lo)}_{int(hi)}"] = float(power[sel].sum() / total)
    return out


def _cwt_features(wave: np.ndarray, srate: float, wavelet: str,
                  n_scales: int, bands_khz: list[list[float]]) -> dict:
    import pywt

    wave = wave - np.mean(wave)
    # 用中心频率把目标 kHz 频带映射到尺度
    fc = pywt.central_frequency(wavelet)
    target_khz = np.linspace(20, srate / 2e3, n_scales)
    scales = fc * (srate / 1e3) / np.maximum(target_khz, 1e-6)
    coef, _ = pywt.cwt(wave, scales, wavelet, sampling_period=1.0 / srate)
    energy = (np.abs(coef) ** 2).sum(axis=1)
    total = energy.sum() + 1e-20
    out = {}
    for lo, hi in bands_khz:
        sel = (target_khz >= lo) & (target_khz < hi)
        out[f"cwt_{int(lo)}_{int(hi)}"] = float(energy[sel].sum() / total)
    return out


def waveform_features(ds: AEDataset, cfg: dict) -> pd.DataFrame:
    if not ds.has_waveforms:
        LOG.warning("数据无波形, 跳过分支 B (waveform 特征)")
        return pd.DataFrame(index=range(len(ds)))

    wcfg = cfg.get("features", {}).get("waveform", {})
    bands = wcfg.get("bands_khz", [[0, 150], [150, 300], [300, 450], [450, 1000]])
    cwt_cfg = wcfg.get("cwt", {})
    rows: list[dict] = []
    for i, w in enumerate(ds.waveforms):
        srate = ds.srate[i] if ds.srate is not None else np.nan
        if w is None or w.size < 8 or not np.isfinite(srate):
            rows.append({})
            continue
        feat = _spectrum_features(w, srate, bands)
        if cwt_cfg.get("enable", False):
            feat.update(_cwt_features(
                w, srate, cwt_cfg.get("wavelet", "morl"),
                cwt_cfg.get("n_scales", 32), bands))
        rows.append(feat)

    X = pd.DataFrame(rows)
    # 缺波形的行用列中位数填补 (保持矩阵稠密)
    X = X.fillna(X.median(numeric_only=True))
    LOG.info("分支 B: 计算 %d 维波形时频特征", X.shape[1])
    return X


# ---------------------------------------------------------------------------
# 冗余筛查 + 标准化
# ---------------------------------------------------------------------------
def screen_redundancy(X: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    """删去高度相关 (|corr| > threshold) 列对中的后者。"""
    if X.shape[1] < 2:
        return X, []
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if (upper[c] > threshold).any()]
    if drop:
        LOG.info("冗余筛查: 删除 %d 列高度相关特征 %s", len(drop), drop)
    return X.drop(columns=drop), drop


def standardize(X: pd.DataFrame, method: str = "robust"):
    from sklearn.preprocessing import RobustScaler, StandardScaler

    scaler = RobustScaler() if method == "robust" else StandardScaler()
    Xz = scaler.fit_transform(X.to_numpy())
    # RobustScaler 对零方差列可能产生 nan/inf, 兜底清理
    Xz = np.nan_to_num(Xz, nan=0.0, posinf=0.0, neginf=0.0)
    return Xz, scaler


def build_features(ds: AEDataset, cfg: dict) -> FeatureSet:
    """阶段 2 编排: 组装 (param/waveform/both) -> 冗余筛查 -> 标准化。"""
    feat_cfg = cfg.get("features", {})
    branch = feat_cfg.get("branch", "both")

    parts = []
    if branch in ("param", "both"):
        parts.append(param_features(ds, cfg))
    if branch in ("waveform", "both"):
        wf = waveform_features(ds, cfg)
        if wf.shape[1] > 0:
            parts.append(wf)
    if not parts:
        raise ValueError(f"特征分支 '{branch}' 未产生任何特征")

    X = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    # 丢弃常数列 / 全 NaN 列
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().any()]
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    nunique = X.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)
        LOG.info("删除 %d 个常数列: %s", len(const_cols), const_cols)

    dropped = []
    if feat_cfg.get("redundancy", {}).get("enable", True):
        X, dropped = screen_redundancy(
            X, feat_cfg["redundancy"].get("corr_threshold", 0.95))

    Xz, scaler = standardize(X, feat_cfg.get("scaler", "robust"))
    LOG.info("特征矩阵: %d 样本 x %d 特征 (分支=%s)", *Xz.shape, branch)
    return FeatureSet(X, Xz, list(X.columns), scaler, dropped)


def transform_features(ds: AEDataset, cfg: dict, fs: FeatureSet) -> np.ndarray:
    """用已拟合的特征流程把新试件投影到同一特征空间 (阶段 7 跨试件用)。"""
    feat_cfg = cfg.get("features", {})
    branch = feat_cfg.get("branch", "both")
    parts = []
    if branch in ("param", "both"):
        parts.append(param_features(ds, cfg))
    if branch in ("waveform", "both"):
        wf = waveform_features(ds, cfg)
        if wf.shape[1] > 0:
            parts.append(wf)
    X = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    # 对齐训练时的列 (缺失补 0, 多余丢弃)
    X = X.reindex(columns=fs.feature_names)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    Xz = fs.scaler.transform(X.to_numpy())
    return np.nan_to_num(Xz, nan=0.0, posinf=0.0, neginf=0.0)
