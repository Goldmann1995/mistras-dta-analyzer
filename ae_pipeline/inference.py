"""推断: 用持久化的模型束 (scaler + reducer + clusterer) 给新试件打标 (阶段 7/8)。

把训练阶段的全部要件打包成一个 ModelBundle, joblib 持久化后, 新数据进来即可
一键复跑: 特征 -> scaler.transform -> reducer.transform -> approximate_predict。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import features as feat
from . import preprocess as pre
from .dataset import AEDataset
from .utils import get_logger, load_artifact, save_artifact

LOG = get_logger()


@dataclass
class ModelBundle:
    """可持久化的端到端模型束。"""
    scaler: object                      # 已拟合 scaler
    feature_names: list                 # 训练时的特征列顺序 (供对齐新数据)
    reducer: object                     # 已拟合 UMAP (含 transform 能力)
    clusterer: object                   # 已拟合 HDBSCAN (prediction_data=True)
    cfg: dict                           # 训练所用配置 (特征/预处理分支一致性)
    mechanism_by_cluster: dict = field(default_factory=dict)  # 簇->机制名

    def save(self, path):
        return save_artifact(self, path)

    @staticmethod
    def load(path) -> "ModelBundle":
        return load_artifact(path)


def predict(ds: AEDataset, bundle: ModelBundle, do_preprocess: bool = True):
    """对新试件推断簇标签 / 强度 / 机制名。

    Returns: (labels, strengths, mechanisms)
    """
    import hdbscan

    if do_preprocess:
        ds = pre.run(ds, bundle.cfg)
    # bundle 同时具备 feature_names 与 scaler, 可直接复用特征变换
    Xz = feat.transform_features(ds, bundle.cfg, bundle)
    Z = bundle.reducer.transform(Xz)
    labels, strengths = hdbscan.approximate_predict(bundle.clusterer, Z)
    mechanisms = [
        "noise" if lbl == -1
        else bundle.mechanism_by_cluster.get(int(lbl), f"C{int(lbl)}")
        for lbl in labels
    ]
    LOG.info("推断 %d 事件: 指派率=%.1f%%", len(ds),
             100 * float((labels != -1).mean()))
    return np.asarray(labels), np.asarray(strengths), mechanisms
