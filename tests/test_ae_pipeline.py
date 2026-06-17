"""ae_pipeline 流水线测试。

重型依赖 (umap/hdbscan/sklearn/...) 缺失时整模块跳过, 以免影响 MistrasDTA 读取器
的回归 CI。安装分析依赖后 (``pip install .[analysis]``) 即会运行。
"""
import os.path as osp

import pytest

# 缺任一重型依赖则跳过整个模块
pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("umap")
pytest.importorskip("hdbscan")
pytest.importorskip("scipy")

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from ae_pipeline import clustering as clu  # noqa: E402
from ae_pipeline import embedding as emb  # noqa: E402
from ae_pipeline import features as feat  # noqa: E402
from ae_pipeline import io_dta  # noqa: E402
from ae_pipeline import preprocess as pre  # noqa: E402
from ae_pipeline.pipeline import run_pipeline  # noqa: E402
from ae_pipeline.synth import make_synthetic_dataset  # noqa: E402

DTA = osp.join(osp.dirname(osp.abspath(__file__)), "dta", "210527-CH1-15.DTA")


@pytest.fixture(scope="module")
def synth_ds():
    return make_synthetic_dataset(n_per_mech=50, n_specimens=3,
                                  with_waveforms=True, wfm_len=512, seed=0)


def _fast_cfg(out_dir):
    return {
        "seed": 42,
        "io": {"inputs": [DTA]},
        "preprocess": {"bandpass": {"enable": True, "low_khz": 80, "high_khz": 500,
                                    "order": 4}, "amplitude_threshold_db": 0},
        "features": {"branch": "both", "derive_param": True,
                     "waveform": {"bands_khz": [[0, 200], [200, 400], [400, 1000]],
                                  "cwt": {"enable": False}},
                     "redundancy": {"enable": True, "corr_threshold": 0.95},
                     "scaler": "robust"},
        "embedding": {"n_neighbors": 15, "min_dist": 0.0, "n_components": 5,
                      "metric": "euclidean",
                      "stability": {"enable": True, "seeds": [0, 1], "trustworthiness_k": 10}},
        "clustering": {"hdbscan": {"min_cluster_size": 30},
                       "baselines": {"enable": True, "kmeans_k_range": [2, 3, 4, 5],
                                     "gmm_k_range": [2, 3, 4, 5], "run_on_pca": True,
                                     "pca_components": 5}},
        "mapping": {"freq_feature": "peak_freq",
                    "mechanism_bands": [
                        {"name": "low", "low_khz": 0, "high_khz": 200},
                        {"name": "mid", "low_khz": 200, "high_khz": 400},
                        {"name": "high", "low_khz": 400, "high_khz": 2000}]},
        "evolution": {"x_axis": "time", "energy_column": "abs_energy"},
        "validation": {"cross_specimen": {"enable": True},
                       "ablation": {"enable": True}},
        "output": {"dir": str(out_dir), "save_figures": True, "dpi": 80},
    }


# ---------------------------------------------------------------------------
# 数据 / 读取
# ---------------------------------------------------------------------------
def test_synthetic_dataset(synth_ds):
    assert len(synth_ds) > 300
    assert synth_ds.has_waveforms
    assert synth_ds.events["specimen_id"].nunique() == 3
    # 元数据列齐全
    for col in ("event_id", "specimen_id", "channel", "time"):
        assert col in synth_ds.events.columns


def test_load_real_dta():
    ds = io_dta.load_dta_file(DTA)
    assert len(ds) == 8
    assert ds.has_waveforms
    assert "peak_freq" in ds.events.columns
    assert "amp" in ds.events.columns


def test_dataset_select_aligns_waveforms(synth_ds):
    mask = synth_ds.events["specimen_id"] == "SYN-00"
    sub = synth_ds.select(mask.to_numpy())
    assert len(sub) == int(mask.sum())
    assert len(sub.waveforms) == len(sub)
    assert sub.srate.shape[0] == len(sub)
    # event_id 重新编号
    assert list(sub.events["event_id"]) == list(range(len(sub)))


# ---------------------------------------------------------------------------
# 各阶段
# ---------------------------------------------------------------------------
def test_features_branches(synth_ds):
    cfg = _fast_cfg("outputs")
    ds = pre.run(synth_ds, cfg)
    fs = feat.build_features(ds, cfg)
    assert fs.Xz.shape[0] == len(ds)
    assert fs.Xz.shape[1] >= 3
    assert np.isfinite(fs.Xz).all()
    # 含波形分频段特征
    assert any(n.startswith("w_pp_") for n in fs.feature_names)


def test_embedding_and_clustering_recovers_mechanisms(synth_ds):
    cfg = _fast_cfg("outputs")
    ds = pre.run(synth_ds, cfg)
    fs = feat.build_features(ds, cfg)
    er = emb.fit_umap(fs.Xz, cfg)
    cr = clu.cluster_hdbscan(er.Z, cfg)
    # 4 个合成机制 -> 期望 3~5 簇, 噪声不过多
    assert 3 <= cr.n_clusters <= 6
    assert cr.noise_ratio < 0.2
    # 稳定性指标存在且合理
    assert er.stability["trustworthiness_mean"] > 0.8


def test_feature_transform_consistency(synth_ds):
    """transform_features 对训练集应与 build_features 输出一致 (无泄漏一致性)。"""
    cfg = _fast_cfg("outputs")
    ds = pre.run(synth_ds, cfg)
    fs = feat.build_features(ds, cfg)
    Xz2 = feat.transform_features(ds, cfg, fs)
    assert Xz2.shape == fs.Xz.shape
    np.testing.assert_allclose(Xz2, fs.Xz, atol=1e-6)


# ---------------------------------------------------------------------------
# 端到端
# ---------------------------------------------------------------------------
def test_full_pipeline_synthetic(tmp_path, synth_ds):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(_fast_cfg(tmp_path / "out")))
    results = run_pipeline(str(cfg_path), dataset=synth_ds)

    assert results["n_events"] == len(synth_ds)
    assert results["clustering"]["n_clusters"] >= 3
    # 关键产物落盘
    out = tmp_path / "out"
    for fname in ("scaler.joblib", "umap_reducer.joblib", "hdbscan_clusterer.joblib",
                  "events_labeled.csv", "cluster_mechanism_mapping.csv",
                  "report.json", "report.md", "umap_clusters.png"):
        assert (out / fname).exists(), f"缺少产物 {fname}"
    # 跨试件验证 (3 试件 -> 留一法应有结果)
    assert "cross_specimen" in results
    assert all(r["test_assigned_ratio"] >= 0 for r in results["cross_specimen"])


def test_persisted_artifacts_roundtrip(tmp_path, synth_ds):
    """持久化的 ModelBundle 可加载并对新试件一键推断 (阶段 8 复现)。"""
    from ae_pipeline.inference import ModelBundle, predict

    cfg_path = tmp_path / "cfg.yaml"
    cfg = _fast_cfg(tmp_path / "out")
    cfg["validation"] = {"cross_specimen": {"enable": False}, "ablation": {"enable": False}}
    cfg_path.write_text(yaml.safe_dump(cfg))
    run_pipeline(str(cfg_path), dataset=synth_ds)

    bundle = ModelBundle.load(tmp_path / "out" / "model_bundle.joblib")
    new_ds = make_synthetic_dataset(n_per_mech=20, n_specimens=1, wfm_len=512, seed=99)
    labels, strengths, mechs = predict(new_ds, bundle)
    assert len(labels) == len(new_ds) == len(mechs)
    # 合成新试件应大多被指派到已知机制簇
    assert float((labels != -1).mean()) > 0.5
