"""Orchestrates the full comparison: data -> 6 features -> 2 clusterers ->
metrics -> per-method plots -> comparison table + JSON.
"""

import os
import csv
import json
import time

import numpy as np

from . import encoders, viz
from .clustering import evaluate_method


def run_comparison(ds, cfg):
    """Run every requested method on the dataset ``ds`` and write outputs.

    Returns the in-memory results dict (also saved as ``results.json``).
    """
    os.makedirs(cfg.out, exist_ok=True)
    results = {}
    has_truth = ds.labels_true is not None

    print(f"\n=== Comparing methods {cfg.methods} on {len(ds)} hits "
          f"(common_dim={cfg.common_dim}, n_runs={cfg.n_runs}) ===")

    for mid in cfg.methods:
        ext = encoders.build(mid)
        ok, why = ext.available()
        if not ok:
            print(f"\n[{mid}] {ext.name}: SKIPPED — {why}")
            results[ext.name] = {"skipped": why}
            continue

        print(f"\n[{mid}] {ext.name}: extracting features ...")
        t0 = time.time()
        latent = ext.fit_transform(ds, cfg)
        if latent is None:
            print(f"[{mid}] {ext.name}: no features produced — SKIPPED")
            results[ext.name] = {"skipped": "no features produced"}
            continue
        dt = time.time() - t0

        clusters, Z = evaluate_method(latent, cfg, ds.labels_true)
        results[ext.name] = {
            "latent_dim": int(latent.shape[1]),
            "seconds": round(dt, 2),
            "clusters": clusters,
        }

        # report + plots for the headline clusterer (kmeans) and hdbscan
        for cname, agg in clusters.items():
            sil = agg.get("silhouette_mean")
            line = (f"    {cname:8s} k={agg['k']} clusters={agg['n_clusters']} "
                    f"noise={agg['n_noise']}  sil={_fmt(sil)} "
                    f"DBI={_fmt(agg.get('davies_bouldin_mean'))} "
                    f"CHI={_fmt(agg.get('calinski_harabasz_mean'), 1)}")
            if has_truth:
                line += f"  ARI={_fmt(agg.get('ari_mean'))} NMI={_fmt(agg.get('nmi_mean'))}"
            print(line)

        # visualizations use the kmeans labels (always present if >=2 clusters)
        labels = clusters["kmeans"]["labels"]
        if labels is not None:
            viz.embedding_plot(cfg.out, ext.name, Z, labels, proj=cfg.projection,
                               title_extra=f"({mid})")
            viz.ra_af_plot(cfg.out, ext.name, ds.ra, ds.af, labels)
        np.save(os.path.join(cfg.out, f"latent_{ext.name}.npy"), latent)

    # comparison figures + tables
    viz.summary_bar(cfg.out, _runnable(results), "kmeans")
    viz.summary_bar(cfg.out, _runnable(results), "hdbscan")
    _write_table(cfg.out, results, has_truth)
    with open(os.path.join(cfg.out, "results.json"), "w") as f:
        json.dump(_jsonable(results), f, indent=2)

    print(f"\n=== Done. Outputs in {cfg.out}/ ===")
    _print_ranking(results, has_truth)
    return results


def _runnable(results):
    return {k: v for k, v in results.items() if "clusters" in v}


def _fmt(x, nd=3):
    return "  n/a" if x is None else f"{x:.{nd}f}"


def _write_table(out_dir, results, has_truth):
    cols = ["method", "clusterer", "latent_dim", "seconds", "k", "n_clusters",
            "n_noise", "silhouette", "silhouette_std", "davies_bouldin",
            "calinski_harabasz"]
    if has_truth:
        cols += ["ari", "nmi"]
    with open(os.path.join(out_dir, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for name, res in results.items():
            if "clusters" not in res:
                continue
            for cname, agg in res["clusters"].items():
                row = [name, cname, res["latent_dim"], res["seconds"],
                       agg["k"], agg["n_clusters"], agg["n_noise"],
                       _r(agg.get("silhouette_mean")), _r(agg.get("silhouette_std")),
                       _r(agg.get("davies_bouldin_mean")),
                       _r(agg.get("calinski_harabasz_mean"), 1)]
                if has_truth:
                    row += [_r(agg.get("ari_mean")), _r(agg.get("nmi_mean"))]
                w.writerow(row)


def _print_ranking(results, has_truth):
    rows = []
    for name, res in results.items():
        if "clusters" not in res:
            continue
        agg = res["clusters"].get("kmeans", {})
        sil = agg.get("silhouette_mean")
        if sil is not None:
            rows.append((name, sil, agg.get("ari_mean")))
    rows.sort(key=lambda r: r[1], reverse=True)
    print("\nRanking by KMeans silhouette (higher = better):")
    for i, (name, sil, ari) in enumerate(rows, 1):
        extra = f"  ARI={ari:.3f}" if (has_truth and ari is not None) else ""
        print(f"  {i}. {name:14s} sil={sil:.3f}{extra}")


def _r(x, nd=4):
    return "" if x is None else round(float(x), nd)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k != "labels"}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
