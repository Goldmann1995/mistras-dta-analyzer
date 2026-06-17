"""通用工具: 随机种子、日志、配置读取、产物持久化 (阶段 0 / 阶段 8)。"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

LOGGER_NAME = "ae_pipeline"


# ---------------------------------------------------------------------------
# 随机种子 (阶段 8: 可复现)
# ---------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """固定 python / numpy 全局随机种子。

    注意: UMAP 自身的可复现需在调用处传入 ``random_state``; HDBSCAN 为确定性算法。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
def get_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# 配置读取 (中央 config.yaml)
# ---------------------------------------------------------------------------
def load_config(path: str | os.PathLike) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def cfg_get(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    """按点号路径读取嵌套配置, 如 ``cfg_get(cfg, "embedding.n_neighbors")``。"""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


# ---------------------------------------------------------------------------
# 产物持久化 (joblib) (阶段 8)
# ---------------------------------------------------------------------------
def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_artifact(obj: Any, path: str | os.PathLike) -> Path:
    import joblib

    p = Path(path)
    ensure_dir(p.parent)
    joblib.dump(obj, p)
    return p


def load_artifact(path: str | os.PathLike) -> Any:
    import joblib

    return joblib.load(path)


def configure_fonts(fonts: list[str] | None) -> None:
    """配置 matplotlib 中文字体 (与项目既有脚本风格一致)。

    若运行环境缺少中文字体 (如 CI), 仅产生缺字提示而不影响出图, 这里一并屏蔽。
    """
    import warnings

    warnings.filterwarnings("ignore", message="Glyph .* missing from font.*")
    if not fonts:
        return
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = list(fonts)
    matplotlib.rcParams["axes.unicode_minus"] = False
