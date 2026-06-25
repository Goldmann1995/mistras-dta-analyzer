"""端到端流水线编排 (阶段 8: 一键复跑 + 产物持久化 + 指标日志)。

阶段 0/1 加载预处理 -> 2 特征 -> 3 UMAP -> 4 HDBSCAN(+基线) -> 5 机制标定
-> 6 时序演化 -> 7 跨试件/消融验证 -> 8 持久化与报告。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from . import clustering as clu
from . import embedding as emb
from . import evolution as evo
from . import features as feat
from . import inference
from . import io_dta
from . import mapping as mp
from . import preprocess as pre
from . import validation as val
from . import viz
from .dataset import AEDataset
from .utils import (configure_fonts, ensure_dir, get_logger, load_config,
                    save_artifact, set_global_seed)

LOG = get_logger()


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run_pipeline(config_path: str = "configs/default.yaml",
                 dataset: AEDataset | None = None) -> dict:
    """运行完整流水线。

    config_path : 中央配置。
    dataset     : 可选直接传入 AEDataset (如合成数据); 否则按 config 的 io.inputs 加载。
    """
    cfg = load_config(config_path)
    seed = cfg.get("seed", 42)
    set_global_seed(seed)
    configure_fonts(cfg.get("output", {}).get("font"))
    out_dir = ensure_dir(cfg.get("output", {}).get("dir", "outputs"))
    dpi = cfg.get("output", {}).get("dpi", 150)
    save_fig = cfg.get("output", {}).get("save_figures", True)
    results: dict = {"config_path": str(config_path), "timestamp": datetime.now().isoformat()}

    # ---- 阶段 0/1: 加载 + 预处理 ----
    ds = dataset if dataset is not None else io_dta.load_dataset(cfg)
    ds = pre.run(ds, cfg)
    ds.save_table(out_dir / "events_clean.parquet")
    results["n_events"] = len(ds)
    results["n_specimens"] = int(ds.events["specimen_id"].nunique())

    # ---- 阶段 2: 特征 ----
    fs = feat.build_features(ds, cfg)
    save_artifact(fs.scaler, out_dir / "scaler.joblib")
    results["n_features"] = len(fs.feature_names)
    results["feature_names"] = fs.feature_names

    # ---- 阶段 3: UMAP ----
    er = emb.fit_umap(fs.Xz, cfg, seed=seed)
    save_artifact(er.reducer, out_dir / "umap_reducer.joblib")
    results["embedding_stability"] = er.stability

    # ---- 阶段 4: HDBSCAN + 基线 ----
    cr = clu.cluster_hdbscan(er.Z, cfg)
    cr.baselines = clu.run_baselines(er.Z, cfg, Xz=fs.Xz)
    save_artifact(cr.clusterer, out_dir / "hdbscan_clusterer.joblib")
    results["clustering"] = {
        "n_clusters": cr.n_clusters, "noise_ratio": cr.noise_ratio, **cr.metrics}

    # 标签写回长表
    labeled = ds.events.copy()
    labeled["cluster"] = cr.labels
    labeled["cluster_prob"] = cr.probabilities
    labeled.to_csv(out_dir / "events_labeled.csv", index=False)

    # ---- 阶段 5: 机制标定 ----
    profiles = mp.cluster_profiles(ds.events, fs, cr.labels)
    mapping_df = mp.map_to_mechanism(profiles, cfg)
    mapping_df.to_csv(out_dir / "cluster_mechanism_mapping.csv", index=False)
    mech_by_cluster = {int(r["cluster"]): r["mechanism"]
                       for _, r in mapping_df.iterrows() if r["cluster"] != -1}

    # 持久化端到端模型束 (阶段 8: 新试件一键推断)
    bundle = inference.ModelBundle(
        scaler=fs.scaler, feature_names=fs.feature_names, reducer=er.reducer,
        clusterer=cr.clusterer, cfg=cfg, mechanism_by_cluster=mech_by_cluster)
    bundle.save(out_dir / "model_bundle.joblib")
    results["mechanism_mapping"] = mapping_df[
        [c for c in ("label", "n", "mech_freq_khz", "mechanism") if c in mapping_df.columns]
    ].to_dict(orient="records")

    # ---- 阶段 6 + 可视化 ----
    if save_fig:
        viz.plot_embedding(er.Z_viz, cr.labels, str(out_dir / "umap_clusters.png"), dpi)
        viz.plot_cluster_frequency(ds.events, fs, cr.labels,
                                   str(out_dir / "cluster_frequency.png"),
                                   freq_col=cfg.get("mapping", {}).get("freq_feature", "peak_freq"),
                                   dpi=dpi)
        evo.plot_evolution(ds.events, cr.labels, cfg, str(out_dir / "evolution.png"))
    onset = evo.onset_sequence(ds.events, cr.labels, cfg.get("evolution", {}).get("x_axis", "time"))
    onset.to_csv(out_dir / "onset_sequence.csv", index=False)
    results["onset_sequence"] = onset.to_dict(orient="records")

    # ---- 阶段 7: 验证 ----
    comparison = val.method_comparison(cr)
    comparison.to_csv(out_dir / "method_comparison.csv", index=False)
    results["method_comparison"] = comparison.to_dict(orient="records")

    vcfg = cfg.get("validation", {})
    if vcfg.get("cross_specimen", {}).get("enable", True):
        cs = val.cross_specimen(ds, cfg)
        if len(cs):
            cs.to_csv(out_dir / "cross_specimen.csv", index=False)
            results["cross_specimen"] = cs.to_dict(orient="records")
    if vcfg.get("ablation", {}).get("enable", True):
        ab = val.ablation(ds, cfg)
        ab.to_csv(out_dir / "ablation.csv", index=False)
        results["ablation"] = ab.to_dict(orient="records")

    # ---- 阶段 8: 报告 ----
    with open(out_dir / "report.json", "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(results), fh, ensure_ascii=False, indent=2)
    _write_markdown_report(results, out_dir / "report.md")
    LOG.info("流水线完成, 产物目录: %s", out_dir)
    return results


def _write_markdown_report(results: dict, path: Path) -> None:
    lines = [
        "# AE 损伤机制识别 — 运行报告",
        f"- 时间: {results.get('timestamp')}",
        f"- 事件数: {results.get('n_events')}  |  试件数: {results.get('n_specimens')}",
        f"- 特征维度: {results.get('n_features')}",
        "",
        "## 聚类 (HDBSCAN on UMAP)",
    ]
    c = results.get("clustering", {})
    lines.append(f"- 簇数: {c.get('n_clusters')}  |  noise: {c.get('noise_ratio', float('nan')):.1%}"
                 f"  |  silhouette: {c.get('silhouette', float('nan')):.3f}")
    st = results.get("embedding_stability", {})
    if st:
        lines.append(f"- 嵌入稳定性: trustworthiness={st.get('trustworthiness_mean', float('nan')):.3f}"
                     f"±{st.get('trustworthiness_std', float('nan')):.3f}, "
                     f"跨种子 ARI={st.get('label_ari_mean', float('nan')):.3f}")
    lines += ["", "## 簇 → 机制初判 (频带, 材料相关, 仅初判)"]
    for m in results.get("mechanism_mapping", []):
        lines.append(f"- {m.get('label')}: {m.get('mech_freq_khz', float('nan')):.0f} kHz "
                     f"-> {m.get('mechanism')} (n={m.get('n')})")
    lines += ["", "## 方法 vs 基线"]
    for m in results.get("method_comparison", []):
        lines.append(f"- {m['method']}: k={m['n_clusters']}, "
                     f"silhouette={m.get('silhouette', float('nan')):.3f}")
    cs = results.get("cross_specimen")
    if cs:
        lines += ["", "## 跨试件泛化 (留一法)"]
        for r in cs:
            lines.append(f"- test={r['test_specimen']}: 指派率={r['test_assigned_ratio']:.1%}, "
                         f"命中簇={r['test_clusters_seen']}/{r['train_clusters']}")
    lines += ["", "> 提醒: UMAP 仅看邻接拓扑, 勿解读簇间距离/簇大小; 机制频带须按实际材料体系标定。"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
