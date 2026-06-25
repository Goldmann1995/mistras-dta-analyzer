#!/usr/bin/env python
"""AE 损伤机制识别流水线 —— 命令行入口。

用法:
    python run_pipeline.py                         # 用 configs/default.yaml 读取真实 DTA
    python run_pipeline.py --config my.yaml        # 指定配置
    python run_pipeline.py --synth                 # 用合成数据端到端演示 (无需 DTA)
    python run_pipeline.py --synth --n-per-mech 200 --n-specimens 4
"""
from __future__ import annotations

import argparse

from ae_pipeline.pipeline import run_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="AE UMAP+HDBSCAN 损伤机制识别流水线")
    ap.add_argument("--config", default="configs/default.yaml", help="中央配置文件路径")
    ap.add_argument("--synth", action="store_true", help="使用合成数据 (忽略 io.inputs)")
    ap.add_argument("--n-per-mech", type=int, default=120, help="合成: 每机制事件数")
    ap.add_argument("--n-specimens", type=int, default=3, help="合成: 试件数")
    ap.add_argument("--no-waveforms", action="store_true", help="合成: 不生成波形")
    args = ap.parse_args()

    dataset = None
    if args.synth:
        from ae_pipeline.synth import make_synthetic_dataset

        dataset = make_synthetic_dataset(
            n_per_mech=args.n_per_mech, n_specimens=args.n_specimens,
            with_waveforms=not args.no_waveforms)

    results = run_pipeline(args.config, dataset=dataset)
    c = results.get("clustering", {})
    print("\n=== 摘要 ===")
    print(f"事件数: {results['n_events']} | 试件数: {results['n_specimens']} "
          f"| 特征: {results['n_features']}")
    print(f"簇数: {c.get('n_clusters')} | noise: {c.get('noise_ratio', 0):.1%} "
          f"| silhouette: {c.get('silhouette', float('nan')):.3f}")
    print(f"产物已写入: {results.get('config_path') and 'outputs/'}")


if __name__ == "__main__":
    main()
