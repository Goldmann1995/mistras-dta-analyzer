"""可视化助手: UMAP 嵌入散点、各簇频率分布 (支撑阶段 3/4/5)。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import get_logger

LOG = get_logger()


def plot_embedding(Z_viz: np.ndarray, labels: np.ndarray, out_path: str,
                   dpi: int = 150, title: str = "UMAP 嵌入 + HDBSCAN 聚类") -> str:
    """2D UMAP 散点, 按簇着色 (噪声灰色)。

    注意: UMAP 中不解读簇间距离/簇大小, 仅看邻接拓扑。
    """
    import matplotlib.pyplot as plt

    clusters = [c for c in sorted(set(labels)) if c != -1]
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(clusters), 1)))
    fig, ax = plt.subplots(figsize=(8, 7))
    noise = labels == -1
    if noise.any():
        ax.scatter(Z_viz[noise, 0], Z_viz[noise, 1], s=8, c="lightgray",
                   alpha=0.4, label="noise")
    for k, c in enumerate(clusters):
        m = labels == c
        ax.scatter(Z_viz[m, 0], Z_viz[m, 1], s=14, color=cmap[k],
                   alpha=0.7, label=f"C{c} (n={int(m.sum())})")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(title); ax.legend(fontsize=8, loc="best")
    ax.text(0.01, 0.01, "仅看邻接拓扑, 勿解读簇间距离/簇大小",
            transform=ax.transAxes, fontsize=7, color="gray")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOG.info("嵌入图已保存: %s", out_path)
    return out_path


def plot_cluster_frequency(events: pd.DataFrame, features, labels: np.ndarray,
                           out_path: str, freq_col: str = "peak_freq",
                           dpi: int = 150) -> str:
    """各簇的频率分布直方图 (机制标定的依据)。"""
    import matplotlib.pyplot as plt

    df = events.copy()
    df["cluster"] = labels
    if freq_col not in df.columns:
        # 退回波形峰值频率
        X = features.X.reset_index(drop=True)
        if "w_peak_freq" in X.columns:
            df[freq_col] = X["w_peak_freq"].to_numpy()
        else:
            LOG.warning("无频率列, 跳过频率分布图")
            return ""
    clusters = [c for c in sorted(set(labels)) if c != -1]
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(clusters), 1)))
    fig, ax = plt.subplots(figsize=(9, 6))
    for k, c in enumerate(clusters):
        vals = df.loc[df["cluster"] == c, freq_col].astype(float).dropna()
        if len(vals):
            ax.hist(vals, bins=30, alpha=0.5, color=cmap[k], label=f"C{c}")
    ax.set_xlabel(f"{freq_col} (kHz)"); ax.set_ylabel("频数")
    ax.set_title("各簇频率分布 (机制标定依据)"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOG.info("频率分布图已保存: %s", out_path)
    return out_path
