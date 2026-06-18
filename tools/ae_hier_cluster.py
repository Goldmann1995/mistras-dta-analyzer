#!/usr/bin/env python3
"""Two-stage (hierarchical) latent-space clustering for Mistras AE waveforms.

Motivation: on this data every flat clustering collapses to 2 dominant
frequency modes (~90 kHz / ~270 kHz). The expected 4 damage mechanisms may be
*nested* inside those 2 modes. This script tests that hypothesis directly:

    stage 1:  split all events into K1 classes (default 2, KMeans)
    stage 2:  inside EACH stage-1 class, cluster again (default K2=2)
              -> K1 x K2 final groups (e.g. 0.0, 0.1, 1.0, 1.1)

It reuses the trained autoencoder latent from ae_deep_cluster.py (same denoise /
feature / model settings as the `yang_hdbscan_check` condition by default), so
the AE is trained once and both stages run on the same latent codes.

Each stage prints a silhouette k-scan so you can see whether a branch genuinely
supports sub-splitting or is being cut arbitrarily.

Example (the yang_hdbscan_check condition):
    python tools/ae_hier_cluster.py yang.DTA --out yang_hier \
        --feature fft --denoise wavelet --include-hit-features
"""
import os
import sys
import csv
import argparse
from types import SimpleNamespace

import numpy as np

# import the heavy lifting (load/denoise/train) from the sibling script
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import ae_deep_cluster as ae  # noqa: E402


# --------------------------------------------------------------------------- #
def _freq(feat):
    for k in ('peak_freq_kHz', 'centroid_freq_kHz', 'avg_freq_kHz'):
        if k in feat:
            return float(feat[k])
    return np.nan


# Parametric AE features used when clustering on physical parameters directly
# (mechanism-discriminating: frequency content, energy, shape).
PARAM_KEYS = ['peak_freq_kHz', 'amplitude_dB']


def _build_space(latent_sub, meta_sub, include_hit):
    """Standardized clustering space: latent (+ optional AE amp/peak-freq)."""
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(latent_sub)
    if include_hit:
        aux = ae._hit_feature_matrix(meta_sub)
        if aux is not None:
            aux = StandardScaler().fit_transform(aux)
            Z = np.column_stack([Z, aux])
    return Z


def _param_matrix(meta_sub):
    """Build a standardized physical-parameter matrix for clustering, using
    exactly the features in PARAM_KEYS (peak frequency + amplitude).
    Returns (Z_standardized, used_cols) or (None, [])."""
    from sklearn.preprocessing import StandardScaler
    avail = [k for k in PARAM_KEYS if any(k in md['feat'] for md in meta_sub)]
    if not avail:
        return None, []
    rows = [[md['feat'].get(k, np.nan) for k in avail] for md in meta_sub]
    M = np.array(rows, dtype=float)
    if M.size == 0:
        return None, []
    # impute any missing values with the column mean so KMeans doesn't choke
    colmean = np.nanmean(M, axis=0)
    inds = np.where(np.isnan(M))
    M[inds] = np.take(colmean, inds[1])
    return StandardScaler().fit_transform(M), avail


def _stage_space(args, latent_sub, meta_sub, space):
    """Pick the clustering space for a stage: 'latent' (CAE codes, optionally
    augmented with hit features) or 'params' (physical parameters directly)."""
    if space == 'params':
        Z, cols = _param_matrix(meta_sub)
        if Z is not None:
            return Z, f"params({len(cols)}: {','.join(cols)})"
        print("    [warning] no physical parameters available; using latent")
    return _build_space(latent_sub, meta_sub, args.include_hit_features), 'latent'


def _kscan(Z, kmax):
    """Return {k: silhouette} and the best k on the standardized space Z."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    res, best_k, best_s = {}, None, -1.0
    kmax = min(kmax, max(2, Z.shape[0] // 2))
    for k in range(2, kmax + 1):
        if Z.shape[0] <= k:
            break
        lab = KMeans(k, random_state=42, n_init=10).fit_predict(Z)
        s = float(silhouette_score(Z, lab))
        res[k] = s
        if s > best_s:
            best_s, best_k = s, k
    return res, best_k


def _kmeans(Z, k):
    from sklearn.cluster import KMeans
    return KMeans(k, random_state=42, n_init=10).fit_predict(Z)


def _describe(members, meta):
    """Mean physical features over a list of member indices."""
    keys = ['peak_freq_kHz', 'centroid_freq_kHz', 'avg_freq_kHz',
            'amplitude_dB', 'energy', 'duration_us', 'rise_us']
    row = {'count': len(members)}
    for k in keys:
        vals = [meta[i]['feat'][k] for i in members if k in meta[i]['feat']]
        row[k] = float(np.mean(vals)) if vals else np.nan
    # RA value = rise time / amplitude (classic AE mechanism discriminator)
    ra = [meta[i]['feat']['rise_us'] / meta[i]['feat']['amplitude_dB']
          for i in members
          if 'rise_us' in meta[i]['feat'] and meta[i]['feat'].get('amplitude_dB')]
    row['RA_us_per_dB'] = float(np.mean(ra)) if ra else np.nan
    return row


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Two-stage hierarchical AE waveform clustering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('input', help='path to a .DTA file')
    ap.add_argument('--out', default='ae_hier_out')
    # stage controls
    ap.add_argument('--stage1-k', type=int, default=0, dest='stage1_k',
                    help='first-stage k for KMeans; 0 = auto-pick by silhouette k-scan')
    ap.add_argument('--stage2-k', type=int, default=0, dest='stage2_k',
                    help='second-stage k for each sub-cluster KMeans; 0 = auto-pick by silhouette k-scan')
    ap.add_argument('--stage2-space', choices=['latent', 'params'], default='latent',
                    dest='stage2_space',
                    help="space the 2nd stage clusters on: 'latent' = CAE codes, "
                         "'params' = physical AE parameters (freq/energy/RA/...)")
    ap.add_argument('--scan-kmax', type=int, default=5, dest='scan_kmax')
    # representation (defaults mirror the yang_hdbscan_check condition)
    ap.add_argument('--feature', choices=['waveform', 'fft', 'both'], default='fft')
    ap.add_argument('--model', choices=['cae', 'vae'], default='cae')
    ap.add_argument('--latent-dim', type=int, default=16, dest='latent_dim')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--batch-size', type=int, default=64, dest='batch_size')
    ap.add_argument('--early-stop', type=int, default=20, dest='early_stop')
    ap.add_argument('--length', type=int, default=0, dest='fixed_length')
    ap.add_argument('--max-waveforms', type=int, default=5000, dest='max_waveforms')
    ap.add_argument('--keep-pretrigger', action='store_true', dest='keep_pretrigger')
    ap.add_argument('--channel', type=int, default=None)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--include-hit-features', action='store_true',
                    dest='include_hit_features',
                    help='augment each stage with AE amplitude + peak frequency')
    # denoising (forwarded to ae.make_denoiser)
    ap.add_argument('--denoise', choices=['none', 'wavelet', 'bandpass', 'wavelet+bandpass'],
                    default='wavelet')
    ap.add_argument('--denoise-wavelet', default='db4', dest='denoise_wavelet')
    ap.add_argument('--denoise-level', type=int, default=0, dest='denoise_level')
    ap.add_argument('--denoise-mode', choices=['soft', 'hard'], default='soft',
                    dest='denoise_mode')
    ap.add_argument('--denoise-band', type=float, nargs=2, default=[20.0, 400.0],
                    dest='denoise_band')
    ap.add_argument('--denoise-order', type=int, default=4, dest='denoise_order')
    # projection for the standard latent_scatter figure (matches ae_deep_cluster)
    ap.add_argument('--projection', choices=['umap', 'tsne', 'pca'], default='pca')
    ap.add_argument('--projection-dim', choices=[2, 3], type=int, default=3,
                    dest='projection_dim')
    ap.add_argument('--umap-neighbors', type=int, default=15, dest='umap_neighbors')
    ap.add_argument('--umap-mindist', type=float, default=0.1, dest='umap_mindist')
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit("PyTorch required.  pip install torch")

    os.makedirs(args.out, exist_ok=True)

    # ---- shared front-end: denoise + load + train (once) ----
    denoiser = ae.make_denoiser(args)
    if denoiser is not None:
        print(f"[0/3] Denoising: {args.denoise}")
    X, X_wave, meta, length, C = ae.load_waveforms(
        args.input, args.channel, args.max_waveforms,
        args.fixed_length, args.keep_pretrigger, args.feature, denoiser)
    latent, loss_curve = ae.train(args, X, length, C)
    N = latent.shape[0]

    # ---- STAGE 1 ----
    print(f"[STAGE 1] clustering (KMeans on latent"
          f"{' + hit features' if args.include_hit_features else ''}) ...")
    Z1 = _build_space(latent, meta, args.include_hit_features)
    scan1, best1 = _kscan(Z1, args.scan_kmax)
    print("  k-scan (silhouette): "
          + "  ".join(f"k={k}:{s:.3f}" for k, s in scan1.items())
          + f"   [best k={best1}]")
    k1 = best1 if args.stage1_k == 0 else args.stage1_k
    k1 = max(1, min(k1, N - 1))
    s1_labels = _kmeans(Z1, k1)

    lines = ["AE Hierarchical (two-stage) clustering — summary",
             "=" * 48,
             f"input        : {args.input}",
             f"channel      : {args.channel if args.channel is not None else 'all'}",
             f"denoise      : {args.denoise}    feature: {args.feature} ({C} ch)",
             f"model        : {args.model}  latent={args.latent_dim}  "
             f"waveforms={N}",
             f"include_hit_features: {args.include_hit_features}",
             f"stage2_space : {args.stage2_space}",
             "",
             f"STAGE 1: KMeans k={k1}",
             "  k-scan: " + ", ".join(f"k{k}={s:.3f}" for k, s in scan1.items())
             + f"  (best={best1})",
             ""]

    # ---- STAGE 2: recurse inside each stage-1 cluster ----
    final_labels = np.empty(N, dtype=object)
    final_rows = []
    for g in sorted(set(s1_labels)):
        members = np.where(s1_labels == g)[0]
        d = _describe(list(members), meta)
        print(f"\n[STAGE 2] stage-1 cluster {g}: n={len(members)}  "
              f"peak_freq~{d['peak_freq_kHz']:.0f}kHz  energy~{d['energy']:.1f}")
        lines.append(f"STAGE-1 cluster {g}: n={len(members)}  "
                     f"peak_freq={d['peak_freq_kHz']:.1f}kHz  "
                     f"centroid={d['centroid_freq_kHz']:.1f}kHz  "
                     f"energy={d['energy']:.2f}  dur={d['duration_us']:.0f}us")

        lat_g = latent[members]
        meta_g = [meta[i] for i in members]
        Zg, where = _stage_space(args, lat_g, meta_g, args.stage2_space)
        scan_g, best_g = _kscan(Zg, args.scan_kmax)
        print(f"    sub-clustering on {where}")
        print("    sub k-scan (silhouette): "
              + "  ".join(f"k={k}:{s:.3f}" for k, s in scan_g.items())
              + f"   [best k={best_g}]")
        lines.append(f"    sub-space: {where}")
        lines.append("    sub k-scan: "
                     + ", ".join(f"k{k}={s:.3f}" for k, s in scan_g.items())
                     + f"  (best={best_g})")

        k2 = best_g if args.stage2_k == 0 else args.stage2_k
        k2 = max(1, min(k2, len(members) - 1))
        sub = np.zeros(len(members), dtype=int) if k2 <= 1 else _kmeans(Zg, k2)

        for s in sorted(set(sub)):
            sub_members = members[sub == s]
            tag = f"{g}.{s}"
            final_labels[sub_members] = tag
            r = _describe(list(sub_members), meta)
            r['cluster'] = tag
            final_rows.append(r)
            print(f"      -> {tag}: n={len(sub_members):5d}  "
                  f"peak_freq={r['peak_freq_kHz']:6.1f}  "
                  f"centroid={r['centroid_freq_kHz']:6.1f}  "
                  f"energy={r['energy']:7.2f}  dur={r['duration_us']:6.0f}  "
                  f"RA={r['RA_us_per_dB']:.3f}")
            lines.append(f"      sub {tag}: n={len(sub_members)}  "
                         f"peak_freq={r['peak_freq_kHz']:.1f}  "
                         f"centroid={r['centroid_freq_kHz']:.1f}  "
                         f"energy={r['energy']:.2f}  dur={r['duration_us']:.0f}  "
                         f"RA={r['RA_us_per_dB']:.3f}")
        lines.append("")

    # ---- outputs ----
    keys = ['peak_freq_kHz', 'centroid_freq_kHz', 'avg_freq_kHz',
            'amplitude_dB', 'energy', 'duration_us', 'rise_us', 'RA_us_per_dB']
    with open(os.path.join(args.out, 'hier_features.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cluster', 'count'] + keys)
        for r in final_rows:
            w.writerow([r['cluster'], r['count']]
                       + [round(r[k], 4) if not np.isnan(r[k]) else '' for k in keys])

    with open(os.path.join(args.out, 'hier_labels.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['wfm_index', 'channel', 'time_s', 'stage1', 'final_cluster'])
        for i, md in enumerate(meta):
            w.writerow([md['index'], md['channel'], md['time'],
                        int(s1_labels[i]), final_labels[i]])

    # ---- map two-stage tags -> integer cluster ids (so we can reuse the exact
    #      ae_deep_cluster figure machinery) ----
    tags = sorted(set(final_labels))
    tag2int = {t: i for i, t in enumerate(tags)}
    labels = np.array([tag2int[t] for t in final_labels], dtype=int)
    lines.append("final cluster id (int) <- two-stage tag:")
    for t in tags:
        lines.append(f"  C{tag2int[t]} <- {t}")
    lines.append("")

    summary = "\n".join(lines)
    with open(os.path.join(args.out, 'hier_summary.txt'), 'w') as f:
        f.write(summary + "\n")
    print("\n" + summary)

    # ---- standard figure set, IDENTICAL to ae_deep_cluster.py ----
    # (loss_curve / latent_scatter / cluster_spectra / cluster_amplitude_vs_freq
    #  / cluster_timeline / cluster_time_vs_freq / prototypes / cluster_labels.csv
    #  / cluster_features.csv / latent_codes.npy / summary.txt)
    from sklearn.preprocessing import StandardScaler
    args.algorithm = 'two-stage'
    space = StandardScaler().fit_transform(latent)
    emb2d, proj_name = ae.embed_projection(args, latent)
    m, valid = ae.metrics_on(space, labels)
    ae.save_outputs(args, X_wave, meta, length, C, latent, loss_curve,
                    space, labels, emb2d, proj_name, m, None, valid)

    # ---- extra: RA–AF mechanism plane (same cluster colors as the above) ----
    _ra_af_plot(args, meta, labels, valid)
    print(f"\nDone. Figures in {args.out}/ match ae_deep_cluster.py "
          f"(cluster ids: {', '.join(f'C{tag2int[t]}={t}' for t in tags)})")


def _ra_af_plot(args, meta, labels, valid):
    """RA–AF mechanism plane, colored by integer cluster id (tab10, matching
    every other figure). RA = rise/amplitude; AF = average frequency.
    tensile (fiber breakage): high AF / low RA; shear (delam/debond): low AF / high RA."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('tab10')

    ra, af, labs = [], [], []
    for i, md in enumerate(meta):
        feat = md['feat']
        afv = next((feat[k] for k in ('avg_freq_kHz', 'centroid_freq_kHz',
                                      'peak_freq_kHz') if k in feat), None)
        if afv is None or 'rise_us' not in feat or not feat.get('amplitude_dB'):
            continue
        ra.append(feat['rise_us'] / feat['amplitude_dB'])
        af.append(float(afv))
        labs.append(int(labels[i]))
    if not ra:
        return
    ra = np.array(ra); af = np.array(af); labs = np.array(labs)
    xhi = float(np.percentile(ra, 99))
    yhi = float(np.percentile(af, 99))
    fig, ax = plt.subplots(figsize=(8, 6))
    for l in valid:
        msk = labs == l
        ax.scatter(ra[msk], af[msk], s=14, alpha=0.55, color=cmap(l % 10),
                   edgecolors='none', label=f'C{l} (n={int(np.sum(msk))})')
    ax.set_xlim(0, xhi); ax.set_ylim(0, yhi)
    ax.set_xlabel('RA value  =  rise time / amplitude  (µs per dB)')
    ax.set_ylabel('Average frequency  AF  (kHz)')
    ax.set_title('RA–AF mechanism plane (color = cluster)')
    ax.text(0.02, 0.97, 'high AF / low RA\n→ tensile (fiber breakage)',
            transform=ax.transAxes, va='top', ha='left', fontsize=8,
            color='#444', bbox=dict(boxstyle='round', fc='white', alpha=0.6))
    ax.text(0.98, 0.03, 'low AF / high RA\n→ shear (delamination/debond)',
            transform=ax.transAxes, va='bottom', ha='right', fontsize=8,
            color='#444', bbox=dict(boxstyle='round', fc='white', alpha=0.6))
    ax.legend(fontsize=8, markerscale=1.6, loc='upper right')
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'hier_ra_af.png'), dpi=150); plt.close()


if __name__ == '__main__':
    main()
