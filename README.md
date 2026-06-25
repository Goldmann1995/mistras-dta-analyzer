[![release](https://img.shields.io/github/v/release/d-cogswell/MistrasDTA)](https://github.com/d-cogswell/MistrasDTA/releases)
[![NewareNDA regression tests](https://github.com/d-cogswell/MistrasDTA/actions/workflows/tests.yml/badge.svg)](https://github.com/d-cogswell/MistrasDTA/actions/workflows/tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/d-cogswell/MistrasDTA/badge.svg?branch=development)](https://coveralls.io/github/d-cogswell/MistrasDTA?branch=development)

# MistrasDTA
Python module to read acoustic emissions hit data and waveforms from Mistras DTA files. The structure of these binary files is detailed in Appendix II of the Mistras user manual.

# Installation
MistrasDTA can be installed from PyPI with the following command:
```
python -m pip install MistrasDTA
```

# Usage
Read the hit summary table from a DTA file:
```
import MistrasDTA
rec, _ = MistrasDTA.read_bin('cluster.DTA', skip_wfm=True)

```

Read hit summary and waveform data from a DTA:
```
import MistrasDTA
from numpy.lib.recfunctions import join_by

# Read the binary file and join summary and waveform tables
rec, wfm = MistrasDTA.read_bin('cluster.DTA')
merged = join_by(['SSSSSSSS.mmmuuun', 'CH'], rec, wfm)

# Extract the first waveform in units of microseconds and volts
t, V = MistrasDTA.get_waveform_data(merged[0])
```

---

# AE 损伤机制识别流水线 (`ae_pipeline`)

在 `MistrasDTA` 读取器之上构建的模块化流水线: **UMAP 嵌入 + HDBSCAN 聚类**,
用于从声发射 (AE) 数据中无监督地识别损伤机制, 并刻画断裂过程时序。每个阶段对应
一个模块, 参数集中在 `configs/default.yaml`, 全流程可一键复跑。

## 安装分析依赖

核心读取器仅依赖 `numpy`; 流水线的科学计算栈放在可选 extra 里:

```
python -m pip install -e .[analysis]
```

## 快速开始

```bash
# 1) 无需真实数据: 用合成多机制/多试件数据端到端演示
python run_pipeline.py --synth --n-per-mech 150 --n-specimens 3

# 2) 跑真实 DTA: 在 configs/default.yaml 的 io.inputs 填入 .DTA 文件或目录
python run_pipeline.py --config configs/default.yaml
```

Python API:

```python
from ae_pipeline import run_pipeline
results = run_pipeline("configs/default.yaml")     # 返回指标 dict, 产物写入 outputs/

# 对新试件一键推断 (复用持久化的 scaler + UMAP + HDBSCAN)
from ae_pipeline.inference import ModelBundle, predict
from ae_pipeline import io_dta
bundle = ModelBundle.load("outputs/model_bundle.joblib")
ds = io_dta.load_dta_file("new_specimen.DTA")
labels, strengths, mechanisms = predict(ds, bundle)
```

## 阶段 → 模块 → 产物

| 阶段 | 模块 | 作用 | 主要产物 |
|----|------|------|---------|
| 0/1 | `io_dta` | DTA → 统一事件长表 | `events_clean.parquet` |
| 1 | `preprocess` | 带通滤波 / 阈值剔噪 / 首达关联 | — |
| 2 | `features` | param + waveform 特征、冗余筛查、标准化 | `scaler.joblib` |
| 3 | `embedding` | UMAP 嵌入 + 多种子稳定性 (trustworthiness/ARI) | `umap_reducer.joblib` |
| 4 | `clustering` | HDBSCAN (主) + kmeans/GMM 基线 + 有效性指标 | `hdbscan_clusterer.joblib`, `method_comparison.csv` |
| 5 | `mapping` | 簇 → 损伤机制 (频带标定) | `cluster_mechanism_mapping.csv` |
| 6 | `evolution` | 时序/载荷演化、累积能量、起始链、sentry | `evolution.png`, `onset_sequence.csv` |
| 7 | `validation` | 跨试件泛化 (留一法) + 消融 | `cross_specimen.csv`, `ablation.csv` |
| 8 | `pipeline` | 编排 + 持久化 + 报告 | `model_bundle.joblib`, `report.md/json` |

## 特征分支 (自适应)

`features.branch` 可取 `param` / `waveform` / `both`:
- **param**: 直接用 AEWin 的 AE 参数 (幅值/能量/上升时间/计数/峰值频率…) 及派生量 (RA 值、平均频率) —— 可解释基线。
- **waveform**: 由原始波形算时频特征 (峰值频率、频率质心、**分频段 partial power**), 可选 CWT 能量分布。
- **both**: 合并两者 (推荐), 经相关性冗余筛查后标准化。

## 方法学提醒

- UMAP 嵌入中**不要解读簇间距离 / 簇大小**的物理含义, 只用邻接拓扑;
- 损伤机制数目**不预设**, 交给 HDBSCAN 自适应;
- 机制频带是**材料相关的**, `configs/default.yaml` 给的是 CFRP 示例, 必须按你的体系
  和真值 (参考试验 / DIC 裂纹 / SEM 断口) 标定, 无真值时应在论文中明确为局限。

