"""阶段 0/1 — 加载: 原始 DTA 导出 -> 统一事件长表 (AEDataset)。

依赖既有 ``MistrasDTA`` 读取器解析二进制, 本模块负责把 (rec, wfm) 两张
recarray 整理成标准化的长表, 并把波形按行序对齐。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import MistrasDTA

from .dataset import FIELD_MAP, META_COLUMNS, AEDataset
from .utils import get_logger

LOG = get_logger()


def _specimen_id(path: str | Path) -> str:
    return Path(path).stem


def _decode_waveforms(rec, wfm) -> tuple[list, np.ndarray, np.ndarray]:
    """把 wfm recarray 中的波形按 rec 行序对齐解码。

    通过 (时间键, 通道) 关联 rec 与 wfm; 缺失波形处置 None。
    """
    n = len(rec)
    waveforms: list = [None] * n
    srate = np.full(n, np.nan)
    tdly = np.full(n, np.nan)
    if wfm is None or len(wfm) == 0:
        return waveforms, srate, tdly

    # 建立 (time, ch) -> wfm 行索引 的查找表
    key_time = wfm["SSSSSSSS.mmmuuun"]
    key_ch = wfm["CH"]
    lookup: dict[tuple[float, int], int] = {}
    for j in range(len(wfm)):
        lookup[(float(key_time[j]), int(key_ch[j]))] = j

    rec_time = rec["SSSSSSSS.mmmuuun"]
    rec_ch = rec["CH"]
    for i in range(n):
        j = lookup.get((float(rec_time[i]), int(rec_ch[i])))
        if j is None:
            continue
        # WAVEFORM 以 float64 字节串存储 (电压, 单位 V)
        waveforms[i] = np.frombuffer(wfm["WAVEFORM"][j], dtype=np.float64).copy()
        srate[i] = float(wfm["SRATE"][j])
        tdly[i] = float(wfm["TDLY"][j])
    return waveforms, srate, tdly


def load_dta_file(
    path: str | Path,
    channels: list[int] | None = None,
    load_column: str | None = None,
    crack_length_column: str | None = None,
) -> AEDataset:
    """读取单个 DTA 文件为 AEDataset。"""
    path = Path(path)
    LOG.info("读取 DTA: %s", path)
    rec, wfm = MistrasDTA.read_bin(str(path))
    if rec is None or len(rec) == 0:
        raise ValueError(f"{path} 未解析到任何 AE hit")

    # --- 构建长表 ---
    df = pd.DataFrame()
    df["channel"] = np.asarray(rec["CH"], dtype=int)
    df["time"] = np.asarray(rec["SSSSSSSS.mmmuuun"], dtype=float)
    df["timestamp"] = (
        np.asarray(rec["TIMESTAMP"], dtype=float)
        if "TIMESTAMP" in rec.dtype.names
        else np.nan
    )

    # AE 参数列 -> 标准化命名
    for raw_name in rec.dtype.names:
        if raw_name in ("SSSSSSSS.mmmuuun", "CH", "TIMESTAMP"):
            continue
        std = FIELD_MAP.get(raw_name, raw_name.lower().replace("-", "_").replace(" ", "_"))
        df[std] = np.asarray(rec[raw_name], dtype=float)

    # 可选: 载荷 / 裂纹长度 (若 read_bin 暴露了相应列)
    df["load"] = (
        np.asarray(rec[load_column], dtype=float)
        if load_column and load_column in rec.dtype.names
        else np.nan
    )
    df["crack_length"] = (
        np.asarray(rec[crack_length_column], dtype=float)
        if crack_length_column and crack_length_column in rec.dtype.names
        else np.nan
    )

    # --- 对齐波形 ---
    waveforms, srate, tdly = _decode_waveforms(rec, wfm)
    df["has_waveform"] = [w is not None for w in waveforms]
    df["specimen_id"] = _specimen_id(path)
    df["event_id"] = np.arange(len(df))

    # 列顺序: 元数据在前, 参数列在后
    ordered = META_COLUMNS + [c for c in df.columns if c not in META_COLUMNS]
    ds = AEDataset(
        df[ordered],
        waveforms=waveforms if any(w is not None for w in waveforms) else None,
        srate=srate,
        tdly=tdly,
    )

    if channels is not None:
        ds = ds.select(ds.events["channel"].isin(channels).to_numpy())
    LOG.info(
        "  -> %d hits, %d 通道, 波形=%s",
        len(ds),
        ds.events["channel"].nunique(),
        ds.has_waveforms,
    )
    return ds


def _expand_inputs(inputs: list[str]) -> list[Path]:
    """把文件/目录列表展开成 .DTA 文件列表。"""
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.DTA")) + sorted(p.glob("*.dta")))
        elif p.exists():
            paths.append(p)
        else:
            LOG.warning("输入不存在, 跳过: %s", p)
    return paths


def load_dataset(cfg: dict) -> AEDataset:
    """按配置加载 (可多文件/多试件) 并拼接为单一 AEDataset。"""
    io_cfg = cfg.get("io", {})
    paths = _expand_inputs(io_cfg.get("inputs", []))
    if not paths:
        raise FileNotFoundError(
            "未找到任何 DTA 输入; 请在 config 的 io.inputs 中配置文件或目录"
        )
    datasets = [
        load_dta_file(
            p,
            channels=io_cfg.get("channels"),
            load_column=io_cfg.get("load_column"),
            crack_length_column=io_cfg.get("crack_length_column"),
        )
        for p in paths
    ]
    ds = AEDataset.concat(datasets) if len(datasets) > 1 else datasets[0]
    LOG.info("合计载入 %d hits, 来自 %d 个试件", len(ds), ds.events["specimen_id"].nunique())
    return ds
