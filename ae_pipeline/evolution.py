"""阶段 6 — 断裂过程 / 时序刻画。

把"聚类"提升为"过程表征": 各簇活动随时间/载荷/裂纹长度演化、累积能量曲线、
簇起始先后 (损伤时序链), 并可与 sentry function 关联。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import get_logger

LOG = get_logger()


def _x_axis(events: pd.DataFrame, which: str) -> tuple[np.ndarray, str]:
    """选取演化横轴: time / load / crack_length, 缺失则回退到 time。"""
    if which in events.columns and events[which].notna().any():
        return events[which].to_numpy(dtype=float), which
    if which != "time":
        LOG.warning("演化横轴 '%s' 不可用, 回退到 time", which)
    return events["time"].to_numpy(dtype=float), "time"


def onset_sequence(events: pd.DataFrame, labels: np.ndarray,
                   x_axis: str = "time") -> pd.DataFrame:
    """各簇的起始时刻 (损伤时序链): 每簇第一个 hit 的横轴值。"""
    x, used = _x_axis(events, x_axis)
    cols = ["cluster", "label", f"onset_{used}", f"median_{used}", "n", "onset_rank"]
    rows = []
    for c in sorted(set(labels)):
        if c == -1:
            continue
        xc = x[labels == c]
        rows.append({"cluster": int(c), "label": f"C{c}", f"onset_{used}": float(np.min(xc)),
                     f"median_{used}": float(np.median(xc)), "n": int(len(xc))})
    if not rows:  # 无有效簇 (如样本过少全判为噪声)
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows).sort_values(f"onset_{used}").reset_index(drop=True)
    df["onset_rank"] = np.arange(1, len(df) + 1)
    return df


def cumulative_energy(events: pd.DataFrame, labels: np.ndarray,
                      energy_col: str, x_axis: str = "time") -> dict:
    """各簇累积能量曲线数据 {cluster: (x_sorted, cum_energy)}。"""
    x, _ = _x_axis(events, x_axis)
    if energy_col not in events.columns:
        energy_col = "abs_energy" if "abs_energy" in events.columns else None
    e = (events[energy_col].to_numpy(dtype=float)
         if energy_col else np.ones(len(events)))
    curves = {}
    for c in sorted(set(labels)):
        if c == -1:
            continue
        m = labels == c
        order = np.argsort(x[m])
        curves[int(c)] = (x[m][order], np.cumsum(e[m][order]))
    return curves


def sentry_function(events: pd.DataFrame, x_axis: str = "time", n_bins: int = 50,
                    energy_col: str = "abs_energy") -> tuple[np.ndarray, np.ndarray]:
    """Sentry function: 分箱内 ln(累积AE能量 / 该段内机械量) 的近似。

    无机械量(载荷/位移)时退化为分箱累积能量对数曲线, 仍能反映能量释放节律。
    """
    x, _ = _x_axis(events, x_axis)
    if energy_col not in events.columns:
        energy_col = "abs_energy" if "abs_energy" in events.columns else None
    e = events[energy_col].to_numpy(dtype=float) if energy_col else np.ones(len(events))
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    binned = np.array([e[idx == b].sum() for b in range(n_bins)])
    sentry = np.log(np.cumsum(binned) + 1.0)
    return centers, sentry


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def plot_evolution(events: pd.DataFrame, labels: np.ndarray, cfg: dict,
                   out_path: str) -> str:
    import matplotlib.pyplot as plt

    ev = cfg.get("evolution", {})
    x_axis = ev.get("x_axis", "time")
    energy_col = ev.get("energy_column", "abs_energy")
    x, used = _x_axis(events, x_axis)
    clusters = [c for c in sorted(set(labels)) if c != -1]
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(clusters), 1)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("AE 断裂过程 / 时序刻画 (阶段 6)", fontsize=15, fontweight="bold")

    # (1) 各簇活动散点 (横轴=时间/载荷, 纵轴=幅值)
    ax = axes[0, 0]
    yc = events["amp"].to_numpy() if "amp" in events.columns else np.zeros(len(events))
    for k, c in enumerate(clusters):
        m = labels == c
        ax.scatter(x[m], yc[m], s=12, alpha=0.6, color=cmap[k], label=f"C{c}")
    noise = labels == -1
    if noise.any():
        ax.scatter(x[noise], yc[noise], s=8, alpha=0.2, color="gray", label="noise")
    ax.set_xlabel(used); ax.set_ylabel("幅值 (dB)")
    ax.set_title("各簇活动演化"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (2) 各簇累积事件数
    ax = axes[0, 1]
    for k, c in enumerate(clusters):
        xs = np.sort(x[labels == c])
        ax.plot(xs, np.arange(1, len(xs) + 1), color=cmap[k], label=f"C{c}")
    ax.set_xlabel(used); ax.set_ylabel("累积撞击数")
    ax.set_title("各簇累积活动"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (3) 各簇累积能量
    ax = axes[1, 0]
    curves = cumulative_energy(events, labels, energy_col, x_axis)
    for k, c in enumerate(clusters):
        xs, ce = curves[c]
        ax.plot(xs, ce, color=cmap[k], label=f"C{c}")
    ax.set_xlabel(used); ax.set_ylabel(f"累积 {energy_col}")
    ax.set_title("各簇累积能量"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (4) sentry function
    ax = axes[1, 1]
    cx, sy = sentry_function(events, x_axis, energy_col=energy_col)
    ax.plot(cx, sy, "k-")
    ax.set_xlabel(used); ax.set_ylabel("ln(累积能量)")
    ax.set_title("Sentry function (能量释放节律)"); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.get("output", {}).get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    LOG.info("演化图已保存: %s", out_path)
    return out_path
