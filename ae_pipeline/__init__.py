"""ae_pipeline —— 基于 UMAP 嵌入 + HDBSCAN 聚类的 AE 损伤机制识别流水线。

模块对应路线阶段:
  io_dta      阶段 0/1  加载 DTA -> 统一事件长表
  preprocess  阶段 1    带通滤波 / 剔噪 / 首达关联
  features    阶段 2    param + waveform 特征 (分支点) + 冗余筛查 + 标准化
  embedding   阶段 3    UMAP 嵌入 + 多种子稳定性
  clustering  阶段 4    HDBSCAN (主) + kmeans/GMM 基线 + 有效性指标
  mapping     阶段 5    簇 -> 损伤机制 (频带标定)
  evolution   阶段 6    断裂过程 / 时序刻画
  validation  阶段 7    跨试件泛化 + 消融
  pipeline    阶段 8    端到端编排 + 持久化 + 报告
"""
from __future__ import annotations

from .dataset import AEDataset
from .pipeline import run_pipeline

__all__ = ["AEDataset", "run_pipeline"]
__version__ = "0.1.0"
