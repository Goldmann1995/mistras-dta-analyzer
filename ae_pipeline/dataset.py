"""统一的事件级数据结构 (阶段 0)。

约定: 每个 AE 事件 (hit) 一行, 存为"长表" (pandas.DataFrame), 后续所有模块
读写这张表, 避免散乱。波形 (若有) 与事件按行序对齐, 单独存放以兼容变长。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# 长表的核心元数据列 (固定 schema, 与具体 AE 参数列分离)
META_COLUMNS = [
    "event_id",       # 全局事件序号 (0..N-1)
    "specimen_id",    # 试件标识 (由文件名推断)
    "channel",        # 通道号
    "time",           # 相对测试起点的时间 (秒)
    "timestamp",      # Unix 时间戳
    "load",           # 同步载荷 (无则 NaN)
    "crack_length",   # 裂纹长度 (无则 NaN)
    "has_waveform",   # 是否含原始波形
]

# MistrasDTA 原始字段 -> 标准化列名 (单位见注释)
FIELD_MAP = {
    "RISE": "rise_time",         # 上升时间
    "PCNTS": "pcounts",
    "COUN": "counts",            # 振铃计数
    "ENER": "energy",            # 能量 (AEWin 计)
    "DURATION": "duration",      # 持续时间 (us)
    "AMP": "amp",                # 幅值 (dB)
    "ASL": "asl",
    "THR": "threshold",
    "A-FRQ": "avg_freq",         # 平均频率 (kHz)
    "RMS": "rms",
    "R-FRQ": "reverb_freq",
    "I-FRQ": "init_freq",
    "SIG STRENGTH": "sig_strength",
    "ABS-ENERGY": "abs_energy",  # 绝对能量 (aJ)
    "FRQ-C": "freq_centroid",    # 频率质心 (kHz)
    "P-FRQ": "peak_freq",        # 峰值频率 (kHz)
}


@dataclass
class AEDataset:
    """事件长表 + 对齐的波形。

    Attributes
    ----------
    events : pandas.DataFrame
        每行一个 AE 事件, 含 META_COLUMNS 及可用的 AE 参数列。
    waveforms : list[np.ndarray | None] | None
        与 events 行序对齐的波形电压序列 (变长); 无波形时为 None。
    srate : np.ndarray | None
        每个事件的采样率 (Hz), 与行序对齐。
    tdly : np.ndarray | None
        每个事件的预触发延迟 (采样点数), 与行序对齐。
    """

    events: pd.DataFrame
    waveforms: Optional[list] = None
    srate: Optional[np.ndarray] = None
    tdly: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    # -- 基本属性 ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.events)

    @property
    def has_waveforms(self) -> bool:
        return self.waveforms is not None and any(w is not None for w in self.waveforms)

    def param_columns(self) -> list[str]:
        """返回长表中可用的数值型 AE 参数列 (排除元数据与下划线辅助列)。"""
        out = []
        for c in self.events.columns:
            if c in META_COLUMNS or c.startswith("_"):
                continue
            if pd.api.types.is_numeric_dtype(self.events[c]):
                out.append(c)
        return out

    # -- 子集 / 合并 -------------------------------------------------------
    def select(self, mask: np.ndarray, reindex: bool = True) -> "AEDataset":
        """按布尔掩码取子集, 同时对齐波形/采样率/延迟。"""
        mask = np.asarray(mask, dtype=bool)
        events = self.events.loc[mask].reset_index(drop=True).copy()
        waveforms = (
            [w for w, keep in zip(self.waveforms, mask) if keep]
            if self.waveforms is not None
            else None
        )
        srate = self.srate[mask] if self.srate is not None else None
        tdly = self.tdly[mask] if self.tdly is not None else None
        if reindex:
            events["event_id"] = np.arange(len(events))
        return AEDataset(events, waveforms, srate, tdly, dict(self.meta))

    @staticmethod
    def concat(datasets: list["AEDataset"]) -> "AEDataset":
        """跨试件拼接, 重新分配全局 event_id。"""
        datasets = [d for d in datasets if len(d) > 0]
        if not datasets:
            raise ValueError("没有可拼接的非空数据集")
        events = pd.concat([d.events for d in datasets], ignore_index=True)
        events["event_id"] = np.arange(len(events))

        any_wfm = any(d.waveforms is not None for d in datasets)
        waveforms = None
        if any_wfm:
            waveforms = []
            for d in datasets:
                if d.waveforms is not None:
                    waveforms.extend(d.waveforms)
                else:
                    waveforms.extend([None] * len(d))

        def _stack(attr):
            if all(getattr(d, attr) is not None for d in datasets):
                return np.concatenate([getattr(d, attr) for d in datasets])
            return None

        return AEDataset(events, waveforms, _stack("srate"), _stack("tdly"))

    # -- 持久化 ------------------------------------------------------------
    def save_table(self, path: str | Path) -> Path:
        """把长表存为 parquet (优先) 或 csv。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.events.to_parquet(p)
        except Exception:  # 无 pyarrow 时退回 csv
            p = p.with_suffix(".csv")
            self.events.to_csv(p, index=False)
        return p
