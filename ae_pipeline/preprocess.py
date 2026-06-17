"""阶段 1 — 预处理: 带通滤波、去预触发、幅值/时长阈值剔噪、多传感器首达波关联。

输入: 原始 AEDataset。输出: 清洗后的 AEDataset (波形已滤波, 噪声 hit 已剔除)。
"""
from __future__ import annotations

import numpy as np

from .dataset import AEDataset
from .utils import get_logger

LOG = get_logger()


# ---------------------------------------------------------------------------
# 波形级处理
# ---------------------------------------------------------------------------
def bandpass_filter(wave: np.ndarray, srate: float, low_khz: float,
                    high_khz: float, order: int = 4) -> np.ndarray:
    """对单条波形做零相位带通 (Butterworth, SOS + filtfilt)。"""
    from scipy.signal import butter, sosfiltfilt

    nyq = 0.5 * srate
    low = max(low_khz * 1e3 / nyq, 1e-6)
    high = min(high_khz * 1e3 / nyq, 0.999999)
    # 频带非法或波形过短 (filtfilt 需长度 > 3*(2*order+1)) 时原样返回
    if low >= high or len(wave) <= 3 * (2 * order + 1):
        return wave
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, wave)


def remove_pretrigger(wave: np.ndarray, tdly_samples: float) -> np.ndarray:
    """去除预触发段 (TDLY 为负表示触发前采样点数)。"""
    n_pre = int(round(abs(tdly_samples)))
    if 0 < n_pre < len(wave):
        return wave[n_pre:]
    return wave


def filter_waveforms(ds: AEDataset, cfg: dict) -> AEDataset:
    """按配置对全部波形带通滤波 (就地返回新 dataset)。"""
    bp = cfg.get("preprocess", {}).get("bandpass", {})
    if not bp.get("enable", False) or not ds.has_waveforms:
        return ds
    low, high, order = bp["low_khz"], bp["high_khz"], bp.get("order", 4)
    new_waves = []
    for i, w in enumerate(ds.waveforms):
        if w is None or ds.srate is None or not np.isfinite(ds.srate[i]):
            new_waves.append(w)
            continue
        new_waves.append(bandpass_filter(w, ds.srate[i], low, high, order))
    LOG.info("带通滤波完成: %d–%d kHz, order=%d", low, high, order)
    return AEDataset(ds.events.copy(), new_waves, ds.srate, ds.tdly, dict(ds.meta))


# ---------------------------------------------------------------------------
# 事件级剔噪
# ---------------------------------------------------------------------------
def threshold_mask(ds: AEDataset, cfg: dict) -> np.ndarray:
    """按幅值 / 持续时间阈值生成保留掩码。"""
    pp = cfg.get("preprocess", {})
    n = len(ds)
    mask = np.ones(n, dtype=bool)
    amp_thr = pp.get("amplitude_threshold_db", 0)
    if amp_thr and "amp" in ds.events.columns:
        mask &= ds.events["amp"].to_numpy() >= amp_thr
    dur_thr = pp.get("duration_min_us", 0)
    if dur_thr and "duration" in ds.events.columns:
        mask &= ds.events["duration"].to_numpy() >= dur_thr
    return mask


# ---------------------------------------------------------------------------
# 多传感器首达波关联 (区分直达信号与反射波)
# ---------------------------------------------------------------------------
def first_arrival_mask(ds: AEDataset, window_us: float) -> np.ndarray:
    """同一物理事件在多通道上的多次记录, 仅保留首达 (时间最早) 的那次。

    简化策略: 按时间排序, 用 < window 的滑动时间窗聚合, 每个窗内只保留首个 hit。
    """
    t = ds.events["time"].to_numpy()
    order = np.argsort(t, kind="stable")
    keep = np.zeros(len(t), dtype=bool)
    win = window_us * 1e-6
    last_kept_t = -np.inf
    for idx in order:
        if t[idx] - last_kept_t > win:
            keep[idx] = True
            last_kept_t = t[idx]
    return keep


def run(ds: AEDataset, cfg: dict) -> AEDataset:
    """阶段 1 编排: 滤波 -> 阈值剔噪 -> (可选) 首达关联。"""
    ds = filter_waveforms(ds, cfg)

    mask = threshold_mask(ds, cfg)
    n_drop = int((~mask).sum())
    if n_drop:
        LOG.info("阈值剔噪: 剔除 %d / %d hits", n_drop, len(ds))
    ds = ds.select(mask)

    fa = cfg.get("preprocess", {}).get("first_arrival", {})
    if fa.get("enable", False):
        mask = first_arrival_mask(ds, fa.get("window_us", 20))
        LOG.info("首达波关联: 保留 %d / %d hits", int(mask.sum()), len(ds))
        ds = ds.select(mask)

    LOG.info("预处理后剩余 %d hits", len(ds))
    return ds
