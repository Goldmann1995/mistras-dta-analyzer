#!/usr/bin/env python3
"""Clustering validation plots for AE deep-cluster results.

Reads the cluster_labels.csv and latent_codes.npy produced by
ae_deep_cluster.py (or ae_hierarchical_cluster.py) and the original .DTA
file, then generates publication-quality validation figures.

Outputs (saved into --out):
    silhouette_samples.png     per-sample silhouette (knife plot)
    k_sweep.png                silhouette / CH / DB vs k
    feature_violin.png         feature distribution per cluster (violin)
    inter_cluster_dist.png     pairwise centroid distance heatmap
    cumulative_activity.png    cumulative AE hits per cluster over time
    cluster_dunn_table.txt     Dunn index + per-cluster stats

Examples:
    python tools/ae_cluster_validate.py yang.DTA --results ae_final/
    python tools/ae_cluster_validate.py yang.DTA --results ae_final/ --kmax 10
"""

import os
import sys
import argparse

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from MistrasDTA import read_bin, get_waveform_data  # noqa: E402


FEATURE_FIELDS = [
    ('AMP', 'amplitude_dB'), ('ENER', 'energy'), ('ABS-ENERGY', 'abs_energy'),
    ('RISE', 'rise_us'), ('DURATION', 'duration_us'), ('COUN', 'counts'),
    ('A-FRQ', 'avg_freq_kHz'), ('P-FRQ', 'peak_freq_kHz'),
    ('FRQ-C', 'centroid_freq_kHz'), ('R-FRQ', 'rev_freq_kHz'),
    ('I-FRQ', 'init_freq_kHz'),
]


def load_results(results_dir):
    import csv
    labels_path = os.path.join(results_dir, 'cluster_labels.csv')
    latent_path = os.path.join(results_dir, 'latent_codes.npy')
    if not os.path.isfile(labels_path):
        raise SystemExit(f"cluster_labels.csv not found in {results_dir}")
    if not os.path.isfile(latent_path):
        raise SystemExit(f"latent_codes.npy not found in {results_dir}")

    latent = np.load(latent_path)
    with open(labels_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    labels = []
    times = []
    entropies = []
    for r in rows:
        if 'cluster' in r:
            labels.append(int(r['cluster']))
        elif 'final_cluster' in r:
            labels.append(int(r['final_cluster']))
        else:
            labels.append(0)
        times.append(float(r.get('time_s', r.get('time', 0))))
        ent = r.get('entropy_nats', '')
        entropies.append(float(ent) if ent else np.nan)

    return latent, np.array(labels), np.array(times), np.array(entropies), rows


def load_hit_features(dta_path, csv_rows):
    rec, wfm = read_bin(dta_path)
    if not isinstance(rec, np.recarray) or len(rec) == 0:
        return {}

    same_len = isinstance(rec, np.recarray) and len(rec) == len(wfm)
    rec_times = None
    if not same_len and 'SSSSSSSS.mmmuuun' in rec.dtype.names:
        rec_times = rec['SSSSSSSS.mmmuuun']

    feat_dict = {lbl: [] for _, lbl in FEATURE_FIELDS}
    feat_dict['entropy_nats'] = []

    for r in csv_rows:
        idx = int(r.get('wfm_index', r.get('index', 0)))
        time_s = float(r.get('time_s', r.get('time', 0)))

        rec_row = None
        if same_len and idx < len(rec):
            rec_row = rec[idx]
        elif rec_times is not None:
            rec_row = rec[int(np.argmin(np.abs(rec_times - time_s)))]

        for field, lbl in FEATURE_FIELDS:
            if rec_row is not None and field in rec_row.dtype.names:
                feat_dict[lbl].append(float(rec_row[field]))
            else:
                feat_dict[lbl].append(np.nan)

        ent = r.get('entropy_nats', '')
        feat_dict['entropy_nats'].append(float(ent) if ent else np.nan)

    for k in feat_dict:
        feat_dict[k] = np.array(feat_dict[k])

    return feat_dict


# ------------------------------------------------------------------ #
# Plot 1: Per-sample silhouette (knife plot)
# ------------------------------------------------------------------ #
def plot_silhouette_samples(latent, labels, out_dir, plt):
    from sklearn.metrics import silhouette_samples, silhouette_score
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(latent)
    valid_mask = labels >= 0
    if np.sum(valid_mask) < 4:
        return
    Z_v, lab_v = Z[valid_mask], labels[valid_mask]
    sample_sil = silhouette_samples(Z_v, lab_v)
    avg_sil = silhouette_score(Z_v, lab_v)

    clusters = sorted(set(lab_v))
    cmap = plt.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(8, max(5, len(clusters) * 1.2)))
    y_lower = 0
    for i, c in enumerate(clusters):
        c_sil = sample_sil[lab_v == c]
        c_sil.sort()
        size = len(c_sil)
        y_upper = y_lower + size
        ax.barh(range(y_lower, y_upper), c_sil, height=1.0,
                color=cmap(i % 10), edgecolor='none', alpha=0.8)
        ax.text(-0.05, y_lower + size / 2, f'C{c}\nn={size}',
                fontsize=9, va='center', ha='right')
        y_lower = y_upper + 2

    ax.axvline(avg_sil, color='red', linestyle='--', lw=1.5,
               label=f'avg silhouette = {avg_sil:.4f}')
    ax.axvline(0, color='black', linestyle='-', lw=0.5)
    ax.set_xlabel('Silhouette coefficient')
    ax.set_ylabel('Samples (sorted within cluster)')
    ax.set_title('Per-sample silhouette plot')
    ax.set_yticks([])
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(-0.3, max(0.5, sample_sil.max() * 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'silhouette_samples.png'), dpi=150)
    plt.close()
    print(f"  silhouette_samples.png  (avg={avg_sil:.4f})")

    neg_frac = np.sum(sample_sil < 0) / len(sample_sil)
    return avg_sil, neg_frac


# ------------------------------------------------------------------ #
# Plot 2: k-sweep (silhouette, CH, DB vs k)
# ------------------------------------------------------------------ #
def plot_k_sweep(latent, out_dir, plt, kmax=10):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                                 davies_bouldin_score)

    Z = StandardScaler().fit_transform(latent)
    ks = list(range(2, kmax + 1))
    sils, chs, dbs = [], [], []

    for k in ks:
        lab = KMeans(k, random_state=42, n_init=10).fit_predict(Z)
        sils.append(silhouette_score(Z, lab))
        chs.append(calinski_harabasz_score(Z, lab))
        dbs.append(davies_bouldin_score(Z, lab))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(ks, sils, 'o-', color='#0891b2', lw=1.8, markersize=6)
    best_k_sil = ks[int(np.argmax(sils))]
    ax.axvline(best_k_sil, color='red', ls='--', alpha=0.6)
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Silhouette score')
    ax.set_title(f'Silhouette (best k={best_k_sil})')
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ks, chs, 's-', color='#059669', lw=1.8, markersize=6)
    best_k_ch = ks[int(np.argmax(chs))]
    ax.axvline(best_k_ch, color='red', ls='--', alpha=0.6)
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Calinski-Harabasz index')
    ax.set_title(f'Calinski-Harabasz (best k={best_k_ch})')
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(ks, dbs, 'D-', color='#dc2626', lw=1.8, markersize=6)
    best_k_db = ks[int(np.argmin(dbs))]
    ax.axvline(best_k_db, color='red', ls='--', alpha=0.6)
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Davies-Bouldin index')
    ax.set_title(f'Davies-Bouldin (best k={best_k_db})')
    ax.grid(alpha=0.3)

    plt.suptitle('Cluster number selection (KMeans on latent space)', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'k_sweep.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  k_sweep.png  (sil best k={best_k_sil}, CH best k={best_k_ch}, DB best k={best_k_db})")
    return best_k_sil, best_k_ch, best_k_db


# ------------------------------------------------------------------ #
# Plot 3: Feature violin plots
# ------------------------------------------------------------------ #
def plot_feature_violin(feat_dict, labels, out_dir, plt):
    display_keys = [
        ('peak_freq_kHz', 'Peak Freq (kHz)'),
        ('centroid_freq_kHz', 'Centroid Freq (kHz)'),
        ('amplitude_dB', 'Amplitude (dB)'),
        ('energy', 'Energy'),
        ('duration_us', 'Duration (μs)'),
        ('rise_us', 'Rise Time (μs)'),
        ('entropy_nats', 'Entropy (nats)'),
    ]
    available = [(k, name) for k, name in display_keys
                 if k in feat_dict and not np.all(np.isnan(feat_dict[k]))]
    if not available:
        print("  (skipping feature_violin: no features available)")
        return

    clusters = sorted(set(labels[labels >= 0]))
    cmap = plt.get_cmap('tab10')
    n_feat = len(available)
    ncol = min(4, n_feat)
    nrow = int(np.ceil(n_feat / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for idx, (key, name) in enumerate(available):
        ax = axes[idx // ncol][idx % ncol]
        ax.set_visible(True)
        vals = feat_dict[key]
        data = []
        positions = []
        for i, c in enumerate(clusters):
            mask = (labels == c) & ~np.isnan(vals)
            if np.sum(mask) > 1:
                data.append(vals[mask])
                positions.append(i)

        if not data:
            continue

        parts = ax.violinplot(data, positions=positions, showmedians=True, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(cmap(clusters[positions[i]] % 10))
            pc.set_alpha(0.6)
        parts['cmedians'].set_color('black')

        bp = ax.boxplot(data, positions=positions, widths=0.15,
                        showfliers=False, patch_artist=False,
                        medianprops=dict(color='black', lw=1.5),
                        whiskerprops=dict(color='gray'),
                        capprops=dict(color='gray'))

        ax.set_xticks(range(len(clusters)))
        ax.set_xticklabels([f'C{c}' for c in clusters])
        ax.set_title(name, fontsize=11)
        ax.grid(alpha=0.2, axis='y')

    plt.suptitle('Feature distributions per cluster', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_violin.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  feature_violin.png  ({len(available)} features)")


# ------------------------------------------------------------------ #
# Plot 4: Inter-cluster centroid distance heatmap
# ------------------------------------------------------------------ #
def plot_inter_cluster_dist(latent, labels, out_dir, plt):
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import pdist, squareform

    Z = StandardScaler().fit_transform(latent)
    clusters = sorted(set(labels[labels >= 0]))
    if len(clusters) < 2:
        return

    centroids = np.array([Z[labels == c].mean(axis=0) for c in clusters])
    D = squareform(pdist(centroids, metric='euclidean'))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(D, cmap='YlOrRd', interpolation='nearest')
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([f'C{c}' for c in clusters])
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f'C{c}' for c in clusters])

    for i in range(len(clusters)):
        for j in range(len(clusters)):
            ax.text(j, i, f'{D[i, j]:.2f}', ha='center', va='center',
                    fontsize=10, color='white' if D[i, j] > D.max() * 0.6 else 'black')

    plt.colorbar(im, ax=ax, label='Euclidean distance (latent)')
    ax.set_title('Inter-cluster centroid distance')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'inter_cluster_dist.png'), dpi=150)
    plt.close()
    print(f"  inter_cluster_dist.png  (min={D[D>0].min():.2f}, max={D.max():.2f})")


# ------------------------------------------------------------------ #
# Plot 5: Cumulative AE activity per cluster
# ------------------------------------------------------------------ #
def plot_cumulative_activity(times, labels, out_dir, plt):
    clusters = sorted(set(labels[labels >= 0]))
    cmap = plt.get_cmap('tab10')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for i, c in enumerate(clusters):
        mask = labels == c
        t_c = np.sort(times[mask])
        ax.plot(t_c, np.arange(1, len(t_c) + 1), color=cmap(i % 10),
                lw=1.5, label=f'C{c} (n={int(np.sum(mask))})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative hits')
    ax.set_title('Cumulative AE activity per cluster')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    total = len(labels[labels >= 0])
    for i, c in enumerate(clusters):
        mask = labels == c
        t_c = np.sort(times[mask])
        ax.plot(t_c, np.arange(1, len(t_c) + 1) / total * 100,
                color=cmap(i % 10), lw=1.5, label=f'C{c}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative fraction (%)')
    ax.set_title('Normalized cumulative activity')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cumulative_activity.png'), dpi=150)
    plt.close()
    print(f"  cumulative_activity.png")


# ------------------------------------------------------------------ #
# Plot 6: Cluster separation radar chart
# ------------------------------------------------------------------ #
def plot_cluster_radar(feat_dict, labels, out_dir, plt):
    radar_keys = [
        ('peak_freq_kHz', 'Peak Freq'),
        ('centroid_freq_kHz', 'Centroid Freq'),
        ('amplitude_dB', 'Amplitude'),
        ('energy', 'Energy'),
        ('duration_us', 'Duration'),
        ('rise_us', 'Rise Time'),
    ]
    available = [(k, n) for k, n in radar_keys
                 if k in feat_dict and not np.all(np.isnan(feat_dict[k]))]
    if len(available) < 3:
        return

    clusters = sorted(set(labels[labels >= 0]))
    cmap = plt.get_cmap('tab10')

    feat_names = [n for _, n in available]
    N_feat = len(feat_names)
    angles = np.linspace(0, 2 * np.pi, N_feat, endpoint=False).tolist()
    angles += angles[:1]

    cluster_means = []
    for c in clusters:
        means = []
        for k, _ in available:
            v = feat_dict[k][labels == c]
            v = v[~np.isnan(v)]
            means.append(np.mean(v) if len(v) > 0 else 0)
        cluster_means.append(means)
    cluster_means = np.array(cluster_means)

    col_min = cluster_means.min(axis=0)
    col_max = cluster_means.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    normed = (cluster_means - col_min) / col_range

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i, c in enumerate(clusters):
        vals = normed[i].tolist() + [normed[i][0]]
        ax.plot(angles, vals, 'o-', color=cmap(i % 10), lw=1.8,
                markersize=5, label=f'C{c}')
        ax.fill(angles, vals, color=cmap(i % 10), alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feat_names, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title('Cluster feature profiles (normalized)', fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cluster_radar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  cluster_radar.png")


# ------------------------------------------------------------------ #
# Stats: Dunn index + summary table
# ------------------------------------------------------------------ #
def compute_dunn_index(latent, labels):
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(latent)
    clusters = sorted(set(labels[labels >= 0]))
    if len(clusters) < 2:
        return np.nan

    max_intra = 0
    for c in clusters:
        pts = Z[labels == c]
        if len(pts) < 2:
            continue
        dists = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
        max_intra = max(max_intra, dists.max())

    if max_intra == 0:
        return np.nan

    min_inter = np.inf
    for i, c1 in enumerate(clusters):
        for c2 in clusters[i + 1:]:
            p1, p2 = Z[labels == c1], Z[labels == c2]
            d = np.linalg.norm(p1[:, None] - p2[None, :], axis=2).min()
            min_inter = min(min_inter, d)

    return min_inter / max_intra


def write_stats(latent, labels, feat_dict, out_dir, avg_sil, neg_frac):
    from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                                 davies_bouldin_score)
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(latent)
    valid = labels >= 0
    clusters = sorted(set(labels[valid]))

    lines = [
        "Clustering Validation Report",
        "=" * 50,
        "",
        "Global metrics:",
        f"  Silhouette score       : {avg_sil:.4f}",
        f"  Negative silhouette %  : {neg_frac * 100:.1f}%",
        f"  Calinski-Harabasz      : {calinski_harabasz_score(Z[valid], labels[valid]):.1f}",
        f"  Davies-Bouldin         : {davies_bouldin_score(Z[valid], labels[valid]):.4f}",
    ]

    dunn = compute_dunn_index(latent, labels)
    lines.append(f"  Dunn index             : {dunn:.4f}")
    lines += [
        "",
        "Interpretation guide:",
        "  Silhouette  : [-1, 1]  >0.25 reasonable, >0.5 strong",
        "  CH index    : higher = better separated",
        "  DB index    : lower = better (< 1 is good)",
        "  Dunn index  : higher = better (> 1 means inter > intra)",
        "",
        "Per-cluster statistics:",
    ]

    stat_keys = [k for k, _ in FEATURE_FIELDS
                 if _ in feat_dict and not np.all(np.isnan(feat_dict[_]))]
    stat_labels = [lbl for _, lbl in FEATURE_FIELDS if lbl in feat_dict
                   and not np.all(np.isnan(feat_dict[lbl]))]

    for c in clusters:
        mask = labels == c
        n = int(np.sum(mask))
        lines.append(f"\n  Cluster C{c} (n={n}, {100 * n / np.sum(valid):.1f}%):")
        for lbl in stat_labels:
            v = feat_dict[lbl][mask]
            v = v[~np.isnan(v)]
            if len(v) > 0:
                lines.append(f"    {lbl:>20s}: mean={np.mean(v):10.2f}  "
                             f"std={np.std(v):10.2f}  "
                             f"median={np.median(v):10.2f}")

    report = "\n".join(lines)
    path = os.path.join(out_dir, 'validation_report.txt')
    with open(path, 'w') as f:
        f.write(report + "\n")
    print(f"  validation_report.txt")
    print(f"\n{report}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(
        description="Generate clustering validation plots for AE analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('input', help='path to the original .DTA file')
    ap.add_argument('--results', required=True,
                    help='directory with cluster_labels.csv and latent_codes.npy')
    ap.add_argument('--out', default=None,
                    help='output dir for validation plots (default: results dir)')
    ap.add_argument('--kmax', type=int, default=10,
                    help='max k for k-sweep plot')

    args = ap.parse_args()
    out_dir = args.out or args.results
    os.makedirs(out_dir, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print(f"Loading results from {args.results}/ ...")
    latent, labels, times, entropies, csv_rows = load_results(args.results)
    print(f"  {len(labels)} samples, {len(set(labels) - {-1})} clusters, "
          f"latent dim={latent.shape[1]}")

    print(f"Loading hit features from {args.input} ...")
    feat_dict = load_hit_features(args.input, csv_rows)
    print(f"  {len([k for k, v in feat_dict.items() if not np.all(np.isnan(v))])} features loaded")

    print(f"\nGenerating validation plots in {out_dir}/ ...")

    sil_result = plot_silhouette_samples(latent, labels, out_dir, plt)
    avg_sil = sil_result[0] if sil_result else 0
    neg_frac = sil_result[1] if sil_result else 1

    plot_k_sweep(latent, out_dir, plt, kmax=args.kmax)
    plot_feature_violin(feat_dict, labels, out_dir, plt)
    plot_inter_cluster_dist(latent, labels, out_dir, plt)
    plot_cumulative_activity(times, labels, out_dir, plt)
    plot_cluster_radar(feat_dict, labels, out_dir, plt)
    write_stats(latent, labels, feat_dict, out_dir, avg_sil, neg_frac)

    print("\nDone.")


if __name__ == '__main__':
    main()
