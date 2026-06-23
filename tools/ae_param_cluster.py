#!/usr/bin/env python3
"""Standalone parametric clustering for Mistras AE hits.

Clusters AE events using their scalar features (amplitude, peak frequency,
waveform entropy, etc.) — no autoencoder or latent space involved.
Fast and interpretable: outputs scatter plots, decision-tree rules, and CSV.

Pipeline:
    .DTA file  →  parse hits + waveforms
               →  apply signal filter (filter_config.json)
               →  extract features (amplitude, peak freq, entropy…)
               →  standardize  →  cluster (KMeans / GMM / HDBSCAN / DBSCAN)
               →  decision-tree rules (explainable)
               →  scatter plots + CSV + summary

Examples:
    python tools/ae_param_cluster.py data.DTA
    python tools/ae_param_cluster.py data.DTA --features amplitude peak_frequency entropy
    python tools/ae_param_cluster.py data.DTA --algorithm hdbscan --features amplitude peak_frequency entropy duration
    python tools/ae_param_cluster.py data.DTA --clusters 5 --no-filter
"""

import os
import sys
import csv
import argparse
import json
import textwrap

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from MistrasDTA import read_bin, get_waveform_data  # noqa: E402

# ── feature registry ──────────────────────────────────────────────────────── #
# Maps friendly name → raw recarray field name
FIELD_MAP = {
    'amplitude':      'AMP',
    'energy':         'ENER',
    'abs_energy':     'ABS-ENERGY',
    'duration':       'DURATION',
    'rise':           'RISE',
    'counts':         'COUN',
    'peak_frequency': 'P-FRQ',
    'avg_frequency':  'A-FRQ',
    'freq_centroid':  'FRQ-C',
    'rev_frequency':  'R-FRQ',
    'init_frequency': 'I-FRQ',
    'rms':            'RMS',
    'signal_strength': 'SIG STRENGTH',
}

COMPUTED_FEATURES = {'entropy'}

ALL_FEATURES = sorted(FIELD_MAP.keys()) + sorted(COMPUTED_FEATURES)


# ── waveform entropy (same algorithm as backend) ─────────────────────────── #
def compute_waveform_entropy(V):
    from scipy.stats import skew, kurtosis
    n = len(V)
    if n < 2:
        return 0.0
    sigma = np.std(V)
    if sigma == 0:
        return 0.0
    b_n = 3.49 * sigma * n ** (-1.0 / 3.0)
    sk = skew(V)
    kurt_val = kurtosis(V, fisher=True)
    sk2 = sk ** 2
    c_sk = np.sqrt(1.0 + 2.0 * sk2) if sk2 > 0 else 1.0
    c_kur = (1.0 + (kurt_val / 4.0)) ** (-0.2) if kurt_val > -4.0 else 1.0
    b_opt = b_n * c_sk * c_kur
    if b_opt <= 0:
        b_opt = b_n if b_n > 0 else 1.0
    v_range = np.max(V) - np.min(V)
    if v_range == 0:
        return 0.0
    num_bins = max(1, int(np.ceil(v_range / b_opt)))
    hist, _ = np.histogram(V, bins=num_bins)
    hist = hist[hist > 0]
    P = hist / n
    return float(-np.sum(P * np.log(P)))


# ── filter config ─────────────────────────────────────────────────────────── #
def _load_filter_config(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    default = os.path.join(_REPO_ROOT, 'filter_config.json')
    if os.path.exists(default):
        with open(default) as f:
            return json.load(f)
    return {'filters': []}


def _build_filter_mask(rec, config):
    n = len(rec)
    keep = np.ones(n, dtype=bool)
    field_map = config.get('field_name_map', {})
    names = rec.dtype.names
    removed_counts = []

    for rule in config.get('filters', []):
        field_name = rule.get('field', '')
        raw_field = field_map.get(field_name, field_name)
        if raw_field not in names:
            continue
        vals = rec[raw_field].astype(float)
        rule_mask = np.ones(n, dtype=bool)
        if 'exclude_values' in rule:
            for ev in rule['exclude_values']:
                rule_mask &= vals != float(ev)
        if 'min' in rule:
            rule_mask &= vals >= float(rule['min'])
        if 'max' in rule:
            rule_mask &= vals <= float(rule['max'])
        removed = int(np.sum(keep & ~rule_mask))
        if removed > 0:
            removed_counts.append((field_name, removed))
        keep &= rule_mask

    total = n - int(np.sum(keep))
    if total > 0:
        details = "; ".join(f"{f}: -{c}" for f, c in removed_counts)
        print(f"  [filter] removed {total}/{n} hits ({details})")
    return keep


# ── data loading ──────────────────────────────────────────────────────────── #
def load_data(dta_path, features, channel, filter_config):
    print(f"[1/4] Reading {dta_path} ...")
    rec, wfm = read_bin(dta_path)
    print(f"  hits={len(rec)}  waveforms={len(wfm)}")

    need_entropy = 'entropy' in features
    rec_features = [f for f in features if f != 'entropy']

    # validate requested features exist
    names = rec.dtype.names
    for f in rec_features:
        rf = FIELD_MAP.get(f, f)
        if rf not in names:
            raise SystemExit(f"Feature '{f}' (field '{rf}') not found. Available: {list(names)}")

    # channel mask
    mask = np.ones(len(rec), dtype=bool)
    if channel is not None:
        mask &= rec['CH'] == channel
        print(f"  channel {channel}: {np.sum(mask)} hits")

    # signal filter
    fmask = _build_filter_mask(rec, filter_config)
    mask &= fmask

    indices = np.where(mask)[0]
    if len(indices) < 2:
        raise SystemExit(f"Only {len(indices)} hits after filtering — need at least 2.")

    filtered_rec = rec[mask]

    # build feature matrix
    columns = {}
    for f in rec_features:
        rf = FIELD_MAP.get(f, f)
        columns[f] = filtered_rec[rf].astype(float)

    # compute entropy from waveforms
    if need_entropy:
        print(f"  computing waveform entropy for {len(indices)} hits ...")
        has_wfm = isinstance(wfm, np.recarray) and len(wfm) > 0
        same_len = has_wfm and len(rec) == len(wfm)

        entropies = np.full(len(indices), np.nan)
        matched = 0

        if has_wfm and same_len:
            for j, idx in enumerate(indices):
                try:
                    _, V = get_waveform_data(wfm[idx])
                    if wfm[idx]['TDLY'] < 0:
                        trim = abs(int(wfm[idx]['TDLY']))
                        V = V[trim:]
                    entropies[j] = compute_waveform_entropy(V)
                    matched += 1
                except Exception:
                    pass
        elif has_wfm:
            wfm_times = wfm['SSSSSSSS.mmmuuun'] if 'SSSSSSSS.mmmuuun' in wfm.dtype.names else None
            rec_times = rec['SSSSSSSS.mmmuuun'] if 'SSSSSSSS.mmmuuun' in rec.dtype.names else None
            if wfm_times is not None and rec_times is not None:
                for j, idx in enumerate(indices):
                    t = rec_times[idx]
                    ch = rec[idx]['CH']
                    candidates = np.where((wfm['CH'] == ch) & (np.abs(wfm_times - t) < 1e-6))[0]
                    if len(candidates) > 0:
                        wi = candidates[0]
                        try:
                            _, V = get_waveform_data(wfm[wi])
                            if wfm[wi]['TDLY'] < 0:
                                trim = abs(int(wfm[wi]['TDLY']))
                                V = V[trim:]
                            entropies[j] = compute_waveform_entropy(V)
                            matched += 1
                        except Exception:
                            pass

        print(f"  entropy computed for {matched}/{len(indices)} hits")
        columns['entropy'] = entropies

    # stack into matrix, filter NaN rows
    feat_names = [f for f in features if f in columns]
    X_raw = np.column_stack([columns[f] for f in feat_names])
    valid = np.all(np.isfinite(X_raw), axis=1)

    if np.sum(valid) < 2:
        raise SystemExit("Not enough valid data points after removing NaN values.")

    X_raw = X_raw[valid]
    indices = indices[valid]

    # extract time and channel for CSV output
    time_field = 'SSSSSSSS.mmmuuun'
    times = rec[time_field][indices].astype(float) if time_field in rec.dtype.names else np.zeros(len(indices))
    channels = rec['CH'][indices].astype(int) if 'CH' in rec.dtype.names else np.zeros(len(indices), dtype=int)

    print(f"  {len(indices)} hits with {len(feat_names)} features ready for clustering")
    return X_raw, feat_names, indices, times, channels


# ── clustering ────────────────────────────────────────────────────────────── #
def do_cluster(X_raw, feat_names, algorithm, n_clusters, eps, min_samples, min_cluster_size, max_tree_depth):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.mixture import GaussianMixture
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    print(f"[2/4] Clustering ({algorithm}) ...")

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    if algorithm == 'kmeans':
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X)
    elif algorithm == 'gmm':
        labels = GaussianMixture(n_components=n_clusters, random_state=42, n_init=3).fit_predict(X)
    elif algorithm == 'dbscan':
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    elif algorithm == 'hdbscan':
        try:
            import hdbscan as _hdb
        except ImportError:
            raise SystemExit("pip install hdbscan")
        labels = _hdb.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples,
        ).fit_predict(X)
    else:
        raise SystemExit(f"Unknown algorithm: {algorithm}")

    unique = sorted(set(labels))
    valid_labels = [l for l in unique if l >= 0]
    noise = int(np.sum(labels == -1))
    print(f"  {len(valid_labels)} clusters, {noise} noise points")

    # metrics
    metrics = {}
    non_noise = labels >= 0
    if len(valid_labels) >= 2 and np.sum(non_noise) > len(valid_labels):
        metrics['silhouette'] = float(silhouette_score(X[non_noise], labels[non_noise]))
        metrics['calinski_harabasz'] = float(calinski_harabasz_score(X[non_noise], labels[non_noise]))
        metrics['davies_bouldin'] = float(davies_bouldin_score(X[non_noise], labels[non_noise]))
        print(f"  silhouette={metrics['silhouette']:.3f}  CH={metrics['calinski_harabasz']:.0f}  "
              f"DB={metrics['davies_bouldin']:.3f}")

    # decision tree
    tree_rules = []
    tree_importance = []
    tree_accuracy = 0.0
    if np.sum(non_noise) > 10 and len(valid_labels) >= 2:
        print("[3/4] Fitting decision tree for interpretable rules ...")
        dt = DecisionTreeClassifier(
            max_depth=max_tree_depth,
            min_samples_leaf=max(5, len(X) // 100),
            random_state=42,
        )
        dt.fit(X_raw[non_noise], labels[non_noise])
        tree_accuracy = float(dt.score(X_raw[non_noise], labels[non_noise]))
        print(f"  tree accuracy: {tree_accuracy:.1%}")

        importances = dt.feature_importances_
        tree_importance = [(feat_names[i], float(importances[i])) for i in range(len(feat_names))]
        tree_importance.sort(key=lambda x: x[1], reverse=True)

        tree_rules = _extract_tree_rules(dt, feat_names, valid_labels)
        for r in tree_rules[:8]:
            print(f"    {r['rule_text']}")
    else:
        print("[3/4] Skipping decision tree (too few points or clusters)")

    return labels, valid_labels, metrics, tree_rules, tree_importance, tree_accuracy


def _extract_tree_rules(tree_model, feature_names, class_labels):
    from sklearn.tree import _tree
    tree_ = tree_model.tree_
    rules = []

    def recurse(node, path):
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            values = tree_.value[node][0]
            total = values.sum()
            if total == 0:
                return
            predicted = int(tree_model.classes_[np.argmax(values)])
            confidence = float(np.max(values) / total)
            conditions = []
            for feat_idx, op, thresh in path:
                conditions.append({
                    'feature': feature_names[feat_idx],
                    'op': op,
                    'value': round(float(thresh), 4),
                })
            rule_text = " AND ".join(
                f"{c['feature']} {c['op']} {c['value']}" for c in conditions
            ) + f" → cluster {predicted} ({confidence:.0%}, n={int(total)})"
            rules.append({
                'conditions': conditions,
                'cluster': predicted,
                'samples': int(total),
                'confidence': confidence,
                'rule_text': rule_text,
            })
            return
        feat = tree_.feature[node]
        thresh = tree_.threshold[node]
        recurse(tree_.children_left[node], path + [(feat, '<=', thresh)])
        recurse(tree_.children_right[node], path + [(feat, '>', thresh)])

    recurse(0, [])
    rules.sort(key=lambda r: r['samples'], reverse=True)
    return rules


# ── output ────────────────────────────────────────────────────────────────── #
def save_outputs(args, X_raw, feat_names, indices, times, channels, labels,
                 valid_labels, metrics, tree_rules, tree_importance, tree_accuracy):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = args.out
    os.makedirs(out, exist_ok=True)
    print(f"[4/4] Saving to {out}/ ...")

    n_clusters = len(valid_labels)
    cmap = plt.cm.get_cmap('tab10', max(n_clusters, 1))

    def cluster_colors(labs):
        colors = []
        for l in labs:
            if l < 0:
                colors.append((0.7, 0.7, 0.7, 0.4))
            else:
                colors.append(cmap(l % 10))
        return colors

    cc = cluster_colors(labels)

    # ── pairwise scatter plots ──
    nf = len(feat_names)
    if nf >= 2:
        from itertools import combinations
        pairs = list(combinations(range(nf), 2))
        n_pairs = len(pairs)
        cols = min(3, n_pairs)
        rows = (n_pairs + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
        for idx, (i, j) in enumerate(pairs):
            ax = axes[idx // cols][idx % cols]
            ax.scatter(X_raw[:, i], X_raw[:, j], c=cc, s=8, alpha=0.6, edgecolors='none')
            ax.set_xlabel(feat_names[i])
            ax.set_ylabel(feat_names[j])
            ax.set_title(f'{feat_names[i]} vs {feat_names[j]}')
        for idx in range(n_pairs, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)
        fig.suptitle(f'Parametric Clustering ({args.algorithm}, k={n_clusters})', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'scatter_pairs.png'), dpi=150)
        plt.close(fig)
        print(f"  scatter_pairs.png ({n_pairs} pair(s))")

    # ── cluster timeline ──
    if np.any(times > 0):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.scatter(times, labels, c=cc, s=6, alpha=0.5, edgecolors='none')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Cluster')
        ax.set_title('Cluster assignment over time')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'cluster_timeline.png'), dpi=150)
        plt.close(fig)
        print("  cluster_timeline.png")

    # ── feature boxplots per cluster ──
    fig, axes = plt.subplots(1, nf, figsize=(5 * nf, 5), squeeze=False)
    for fi in range(nf):
        ax = axes[0][fi]
        data_per_cluster = []
        tick_labels = []
        for cl in valid_labels:
            data_per_cluster.append(X_raw[labels == cl, fi])
            tick_labels.append(f'C{cl}')
        if labels.min() < 0:
            data_per_cluster.append(X_raw[labels < 0, fi])
            tick_labels.append('noise')
        bp = ax.boxplot(data_per_cluster, labels=tick_labels, patch_artist=True)
        for k, box in enumerate(bp['boxes']):
            if k < n_clusters:
                box.set_facecolor(cmap(valid_labels[k] % 10))
            else:
                box.set_facecolor((0.7, 0.7, 0.7, 0.5))
        ax.set_title(feat_names[fi])
        ax.set_ylabel(feat_names[fi])
    fig.suptitle('Feature distributions per cluster', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'feature_boxplots.png'), dpi=150)
    plt.close(fig)
    print("  feature_boxplots.png")

    # ── feature importance bar chart ──
    if tree_importance:
        fig, ax = plt.subplots(figsize=(6, 4))
        names_sorted = [t[0] for t in tree_importance]
        vals_sorted = [t[1] for t in tree_importance]
        ax.barh(names_sorted[::-1], vals_sorted[::-1])
        ax.set_xlabel('Importance')
        ax.set_title(f'Decision Tree Feature Importance (acc={tree_accuracy:.1%})')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'feature_importance.png'), dpi=150)
        plt.close(fig)
        print("  feature_importance.png")

    # ── CSV ──
    csv_path = os.path.join(out, 'cluster_labels.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['hit_index', 'time', 'channel'] + feat_names + ['cluster']
        writer.writerow(header)
        for k in range(len(indices)):
            row = [int(indices[k]), f"{times[k]:.7f}", int(channels[k])]
            row += [f"{X_raw[k, fi]:.6f}" for fi in range(nf)]
            row.append(int(labels[k]))
            writer.writerow(row)
    print(f"  cluster_labels.csv ({len(indices)} rows)")

    # ── summary.txt ──
    lines = [
        f"Parametric AE Clustering Summary",
        f"{'=' * 40}",
        f"Input:       {args.input}",
        f"Algorithm:   {args.algorithm}",
        f"Features:    {', '.join(feat_names)}",
        f"Clusters:    {n_clusters}",
        f"Total hits:  {len(labels)}",
        f"Noise:       {int(np.sum(labels < 0))}",
        f"",
    ]
    if metrics:
        lines.append("Metrics:")
        for k, v in metrics.items():
            lines.append(f"  {k}: {v:.4f}")
        lines.append("")

    lines.append("Cluster sizes:")
    for cl in valid_labels:
        cnt = int(np.sum(labels == cl))
        pct = 100.0 * cnt / len(labels)
        lines.append(f"  Cluster {cl}: {cnt} ({pct:.1f}%)")

        # per-cluster feature stats
        cmask = labels == cl
        for fi, fn in enumerate(feat_names):
            vals = X_raw[cmask, fi]
            lines.append(f"    {fn}: mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  "
                         f"min={np.min(vals):.3f}  max={np.max(vals):.3f}  median={np.median(vals):.3f}")
    lines.append("")

    if tree_rules:
        lines.append(f"Decision Tree Rules (accuracy={tree_accuracy:.1%}):")
        for r in tree_rules:
            lines.append(f"  {r['rule_text']}")
        lines.append("")

    if tree_importance:
        lines.append("Feature Importance:")
        for name, imp in tree_importance:
            lines.append(f"  {name}: {imp:.4f}")

    summary = "\n".join(lines)
    with open(os.path.join(out, 'summary.txt'), 'w') as f:
        f.write(summary)
    print("  summary.txt")
    print()
    print(summary)


# ── main ──────────────────────────────────────────────────────────────────── #
def main():
    ap = argparse.ArgumentParser(
        description="Parametric clustering of Mistras AE hits on scalar features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('input', help='path to a .DTA file')
    ap.add_argument('--out', default='ae_param_cluster_out', help='output directory')

    g = ap.add_argument_group('features')
    g.add_argument('--features', nargs='+',
                   default=['amplitude', 'peak_frequency', 'entropy'],
                   help=f'features to cluster on. Available: {", ".join(ALL_FEATURES)}')

    g = ap.add_argument_group('clustering')
    g.add_argument('--algorithm', choices=['kmeans', 'gmm', 'hdbscan', 'dbscan'],
                   default='kmeans')
    g.add_argument('--clusters', type=int, default=4, help='for kmeans/gmm')
    g.add_argument('--eps', type=float, default=0.5, help='for dbscan')
    g.add_argument('--min-samples', type=int, default=5, dest='min_samples')
    g.add_argument('--min-cluster-size', type=int, default=30, dest='min_cluster_size',
                   help='for hdbscan')
    g.add_argument('--max-tree-depth', type=int, default=5, dest='max_tree_depth',
                   help='decision tree max depth')

    g = ap.add_argument_group('data selection')
    g.add_argument('--channel', type=int, default=None, help='restrict to one channel')

    g = ap.add_argument_group('signal filtering')
    g.add_argument('--filter-config', default=None, dest='filter_config',
                   help='path to filter_config.json; default: repo-root filter_config.json')
    g.add_argument('--no-filter', action='store_true', dest='no_filter',
                   help='disable signal pre-filtering entirely')

    args = ap.parse_args()

    # validate features
    for f in args.features:
        if f not in FIELD_MAP and f not in COMPUTED_FEATURES:
            raise SystemExit(f"Unknown feature '{f}'. Available: {', '.join(ALL_FEATURES)}")

    filter_cfg = {'filters': []} if args.no_filter else _load_filter_config(args.filter_config)
    if filter_cfg.get('filters'):
        print(f"[filter] {len(filter_cfg['filters'])} rule(s) active")

    X_raw, feat_names, indices, times, channels = load_data(
        args.input, args.features, args.channel, filter_cfg)

    labels, valid_labels, metrics, tree_rules, tree_importance, tree_accuracy = do_cluster(
        X_raw, feat_names, args.algorithm, args.clusters,
        args.eps, args.min_samples, args.min_cluster_size, args.max_tree_depth)

    save_outputs(args, X_raw, feat_names, indices, times, channels, labels,
                 valid_labels, metrics, tree_rules, tree_importance, tree_accuracy)


if __name__ == '__main__':
    main()
