"""合成 AE 数据生成器 (测试 / 演示用)。

真实 DTA 文件常被 .gitignore 排除, 且仓库内测试文件 hit 数过少, 不足以演示
UMAP/HDBSCAN。本模块按"多机制 + 多试件"生成带波形的 AEDataset, 让整条流水线
可端到端跑通并产出有意义的结果。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import META_COLUMNS, AEDataset

# 几类典型损伤机制的特征中心 (peak_freq kHz, amp dB, rise us, 能量量级, 频带)
_MECHANISMS = [
    {"name": "matrix", "f_khz": 120, "amp": 50, "rise": 25, "energy": 1.0},
    {"name": "debond", "f_khz": 230, "amp": 60, "rise": 12, "energy": 4.0},
    {"name": "pullout", "f_khz": 380, "amp": 70, "rise": 6, "energy": 12.0},
    {"name": "fiber", "f_khz": 520, "amp": 80, "rise": 3, "energy": 30.0},
]


def _make_waveform(f_khz: float, amp_db: float, rise_us: float,
                   srate: float, n: int, rng) -> np.ndarray:
    """阻尼正弦 + 噪声, 模拟一次 AE 撞击波形。"""
    t = np.arange(n) / srate
    f = f_khz * 1e3
    amp = 10 ** ((amp_db - 60) / 20)            # dB -> 线性近似
    tau = max(rise_us, 1) * 1e-6 * 3
    env = np.exp(-t / tau) * (1 - np.exp(-t / (0.3 * tau)))
    sig = amp * env * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    sig += rng.normal(0, amp * 0.03, n)         # 本底噪声
    return sig.astype(np.float64)


def make_synthetic_dataset(n_per_mech: int = 120, n_specimens: int = 3,
                           seed: int = 42, with_waveforms: bool = True,
                           srate: float = 1e7, wfm_len: int = 1024) -> AEDataset:
    """生成多试件、多机制的合成 AEDataset。"""
    rng = np.random.default_rng(seed)
    rows, waveforms, srates, tdlys = [], [], [], []
    eid = 0
    for sp in range(n_specimens):
        sp_id = f"SYN-{sp:02d}"
        # 试件级随机扰动, 制造跨试件差异
        sp_shift = rng.normal(0, 0.04, len(_MECHANISMS))
        for mi, mech in enumerate(_MECHANISMS):
            n = n_per_mech + rng.integers(-15, 15)
            for _ in range(n):
                f = mech["f_khz"] * (1 + sp_shift[mi] + rng.normal(0, 0.06))
                amp = mech["amp"] + rng.normal(0, 3)
                rise = max(0.5, mech["rise"] * (1 + rng.normal(0, 0.2)))
                energy = max(0.01, mech["energy"] * (1 + rng.normal(0, 0.3)))
                duration = rise * rng.uniform(2, 6) + 20
                counts = max(1, int(duration / max(1.0, 1000 / f)))
                t_event = rng.uniform(0, 1000) + mi * 50  # 机制有时序先后
                rows.append({
                    "specimen_id": sp_id, "channel": int(rng.integers(1, 5)),
                    "time": t_event, "timestamp": 1.6e9 + t_event,
                    "load": t_event * 0.5, "crack_length": np.nan,
                    "rise_time": rise, "counts": counts, "energy": energy,
                    "duration": duration, "amp": amp, "abs_energy": energy * 10,
                    "freq_centroid": f * 1.1, "peak_freq": f,
                    "_mech": mech["name"],
                })
                if with_waveforms:
                    waveforms.append(_make_waveform(f, amp, rise, srate, wfm_len, rng))
                    srates.append(srate); tdlys.append(0.0)
                eid += 1

    df = pd.DataFrame(rows)
    df.insert(0, "event_id", np.arange(len(df)))
    df["has_waveform"] = with_waveforms
    ordered = META_COLUMNS + [c for c in df.columns if c not in META_COLUMNS]
    return AEDataset(
        df[ordered],
        waveforms=waveforms if with_waveforms else None,
        srate=np.array(srates) if with_waveforms else None,
        tdly=np.array(tdlys) if with_waveforms else None,
        meta={"synthetic": True},
    )
