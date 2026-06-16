#!/usr/bin/env python3
"""Standalone deep latent-space clustering for Mistras AE waveforms.

Pipeline (no frontend/backend needed, runs entirely on your machine):

    .DTA waveforms  ->  Autoencoder (CAE / VAE)  ->  latent space
                    ->  clustering (KMeans / GMM / HDBSCAN / DBSCAN)
                    ->  2D projection (PCA / t-SNE) + medoid prototype waveforms

Outputs (saved into --out):
    loss_curve.png        training reconstruction loss per epoch
    latent_scatter.png    2D latent map, colored by cluster
    prototypes.png        the medoid (most representative) waveform of each cluster
    cluster_labels.csv    per-waveform: index, channel, time, cluster, latent dims
    latent_codes.npy      raw latent matrix (N x latent_dim)
    summary.txt           run config + cluster sizes + quality metrics

Example:
    python tools/ae_deep_cluster.py mydata.DTA --model cae --epochs 60 --clusters 4
    python tools/ae_deep_cluster.py mydata.DTA --model vae --algorithm hdbscan --projection tsne
"""

import os
import sys
import csv
import argparse

import numpy as np

# Make the bundled MistrasDTA package importable regardless of CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from MistrasDTA import read_bin, get_waveform_data  # noqa: E402


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def round_up_multiple(n, m):
    return int(np.ceil(n / m) * m)


def load_waveforms(dta_path, channel, max_waveforms, fixed_length, keep_pretrigger):
    """Return (X, meta) where X is (N, L) float32 peak-normalized, meta is a list
    of dicts with index/channel/time/sample_rate for each kept waveform."""
    print(f"[1/5] Reading {dta_path} ...")
    rec, wfm = read_bin(dta_path)
    if not isinstance(wfm, np.recarray) or len(wfm) == 0:
        raise SystemExit("No waveforms found in this file (was waveform streaming enabled?).")
    print(f"      hits={len(rec)}  waveforms={len(wfm)}")

    mask = np.ones(len(wfm), dtype=bool)
    if channel is not None:
        mask &= wfm['CH'] == channel
    idx_all = np.where(mask)[0]
    if len(idx_all) == 0:
        raise SystemExit(f"No waveforms on channel {channel}.")

    if len(idx_all) > max_waveforms:
        sel = np.linspace(0, len(idx_all) - 1, max_waveforms).astype(int)
        idx_all = idx_all[sel]

    L = round_up_multiple(max(64, fixed_length), 8)
    rows, meta = [], []
    for i in idx_all:
        row = wfm[i]
        t, V = get_waveform_data(row)
        if not keep_pretrigger and row['TDLY'] < 0:
            trim = abs(int(row['TDLY']))
            V = V[trim:]
        if len(V) == 0:
            continue
        v = V[:L] if len(V) >= L else np.pad(V, (0, L - len(V)))
        peak = np.max(np.abs(v))
        if peak > 0:
            v = v / peak
        rows.append(v.astype(np.float32))
        meta.append({
            'index': int(i),
            'channel': int(row['CH']),
            'time': float(row['SSSSSSSS.mmmuuun']),
            'sample_rate': float(row['SRATE']),
        })

    if len(rows) < 4:
        raise SystemExit("Not enough valid waveforms (need >= 4).")
    X = np.stack(rows)
    print(f"      using {X.shape[0]} waveforms, length={L} "
          f"(channel={'all' if channel is None else channel}, "
          f"pretrigger={'kept' if keep_pretrigger else 'trimmed'})")
    return X, meta, L


# --------------------------------------------------------------------------- #
# Models (torch imported lazily so --help works without it)
# --------------------------------------------------------------------------- #
def build_models(torch, nn, length, latent_dim):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 16, 9, stride=2, padding=4), nn.ReLU(),
                nn.Conv1d(16, 32, 9, stride=2, padding=4), nn.ReLU(),
                nn.Conv1d(32, 64, 9, stride=2, padding=4), nn.ReLU(),
            )
            self.flat_len = (length // 8) * 64
            self.fc = nn.Linear(self.flat_len, latent_dim)

        def forward(self, x):
            return self.fc(self.conv(x).flatten(1))

    class Decoder(nn.Module):
        def __init__(self, flat_len):
            super().__init__()
            self.length = length
            self.fc = nn.Linear(latent_dim, flat_len)
            self.deconv = nn.Sequential(
                nn.ConvTranspose1d(64, 32, 9, 2, 4, output_padding=1), nn.ReLU(),
                nn.ConvTranspose1d(32, 16, 9, 2, 4, output_padding=1), nn.ReLU(),
                nn.ConvTranspose1d(16, 1, 9, 2, 4, output_padding=1),
            )

        def forward(self, z):
            h = self.fc(z).view(z.size(0), 64, self.length // 8)
            return self.deconv(h)

    class CAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = Encoder()
            self.dec = Decoder(self.enc.flat_len)

        def forward(self, x):
            z = self.enc(x)
            return self.dec(z), z

        def encode(self, x):
            return self.enc(x)

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = Encoder()
            self.fc_mu = nn.Linear(latent_dim, latent_dim)
            self.fc_logvar = nn.Linear(latent_dim, latent_dim)
            self.dec = Decoder(self.enc.flat_len)

        def forward(self, x):
            h = self.enc(x)
            mu, logvar = self.fc_mu(h), self.fc_logvar(h)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            return self.dec(z), mu, logvar

        def encode(self, x):
            return self.fc_mu(self.enc(x))

    return CAE, VAE


def train(args, X, length):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(args.device if args.device != 'auto'
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"[2/5] Training {args.model.upper()} on {device} "
          f"(latent={args.latent_dim}, epochs={args.epochs}) ...")

    CAE, VAE = build_models(torch, nn, length, args.latent_dim)
    net = (VAE() if args.model == 'vae' else CAE()).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    X_t = torch.from_numpy(X).unsqueeze(1).to(device)
    N = X.shape[0]
    n_batches = max(1, N // args.batch_size)

    net.train()
    loss_curve = []
    for epoch in range(args.epochs):
        perm = torch.randperm(N)
        run = 0.0
        for b in range(n_batches):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            xb = X_t[idx]
            opt.zero_grad()
            if args.model == 'vae':
                recon, mu, logvar = net(xb)
                rec = F.mse_loss(recon, xb)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = rec + args.beta * kl
            else:
                recon, _ = net(xb)
                loss = F.mse_loss(recon, xb)
            loss.backward()
            opt.step()
            run += float(loss.item())
        avg = run / n_batches
        loss_curve.append(avg)
        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            print(f"      epoch {epoch + 1:3d}/{args.epochs}  loss={avg:.6f}")

    net.eval()
    with torch.no_grad():
        latent = net.encode(X_t).cpu().numpy()
    return latent, loss_curve


# --------------------------------------------------------------------------- #
# Clustering + outputs
# --------------------------------------------------------------------------- #
def cluster_latent(args, latent):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
    from sklearn.mixture import GaussianMixture

    print(f"[3/5] Clustering latent space with {args.algorithm} ...")
    Z = StandardScaler().fit_transform(latent)
    if args.algorithm == 'kmeans':
        labels = KMeans(args.clusters, random_state=42, n_init=10).fit_predict(Z)
    elif args.algorithm == 'gmm':
        labels = GaussianMixture(args.clusters, random_state=42, n_init=3).fit_predict(Z)
    elif args.algorithm == 'dbscan':
        labels = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit_predict(Z)
    elif args.algorithm == 'hdbscan':
        labels = HDBSCAN(min_cluster_size=max(args.min_samples, 5)).fit_predict(Z)
    else:
        raise SystemExit(f"Unknown algorithm: {args.algorithm}")
    return Z, labels


def metrics(Z, labels):
    from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                                 davies_bouldin_score)
    m = {}
    valid = sorted(l for l in set(labels) if l >= 0)
    nn_mask = labels >= 0
    if len(valid) >= 2 and np.sum(nn_mask) > len(valid):
        m['silhouette'] = silhouette_score(Z[nn_mask], labels[nn_mask])
        m['calinski_harabasz'] = calinski_harabasz_score(Z[nn_mask], labels[nn_mask])
        m['davies_bouldin'] = davies_bouldin_score(Z[nn_mask], labels[nn_mask])
    return m, valid


def project_2d(args, latent):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    N = latent.shape[0]
    if latent.shape[1] <= 2:
        p = latent if latent.shape[1] == 2 else np.column_stack([latent[:, 0], np.zeros(N)])
        return p, 'latent'
    if args.projection == 'tsne':
        perp = float(min(30, max(2, N // 4), N - 1))
        return TSNE(2, random_state=42, perplexity=perp, init='pca').fit_transform(latent), 't-SNE'
    return PCA(2, random_state=42).fit_transform(latent), 'PCA'


def save_outputs(args, X, meta, length, latent, loss_curve, Z, labels,
                 proj, proj_name, m, valid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    cmap = plt.get_cmap('tab10')

    def color(l):
        return (0.3, 0.3, 0.3, 0.4) if l == -1 else cmap(l % 10)

    print(f"[4/5] Writing results to {args.out}/ ...")

    # loss curve
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(loss_curve) + 1), loss_curve, color='#0891b2')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'{args.model.upper()} training loss')
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'loss_curve.png'), dpi=130); plt.close()

    # latent scatter
    plt.figure(figsize=(7, 6))
    for l in sorted(set(labels)):
        pts = proj[labels == l]
        plt.scatter(pts[:, 0], pts[:, 1], s=10, color=color(l),
                    label=('noise' if l == -1 else f'C{l}'))
    plt.xlabel(f'{proj_name}-1'); plt.ylabel(f'{proj_name}-2')
    plt.title(f'Latent space ({proj_name}) — {args.algorithm}')
    plt.legend(markerscale=2, fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'latent_scatter.png'), dpi=130); plt.close()

    # medoid prototype waveforms
    protos = []
    for l in valid:
        members = np.where(labels == l)[0]
        centroid = Z[members].mean(axis=0)
        medoid = members[int(np.argmin(np.linalg.norm(Z[members] - centroid, axis=1)))]
        protos.append((l, medoid))

    if protos:
        ncol = min(3, len(protos))
        nrow = int(np.ceil(len(protos) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.4 * nrow), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        for ax, (l, medoid) in zip(axes.flat, protos):
            ax.axis('on')
            ax.plot(X[medoid], color=color(l), lw=0.8)
            cnt = int(np.sum(labels == l))
            ax.set_title(f'C{l}  n={cnt} ({100*cnt/len(labels):.1f}%)  '
                         f'CH{meta[medoid]["channel"]} #{meta[medoid]["index"]}', fontsize=9)
            ax.tick_params(labelsize=7)
        plt.suptitle('Cluster prototype (medoid) waveforms', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'prototypes.png'), dpi=130); plt.close()

    # per-waveform labels CSV
    with open(os.path.join(args.out, 'cluster_labels.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        header = ['wfm_index', 'channel', 'time_s', 'cluster'] + [f'z{i}' for i in range(latent.shape[1])]
        w.writerow(header)
        for i, md in enumerate(meta):
            w.writerow([md['index'], md['channel'], md['time'], int(labels[i])]
                       + [round(float(z), 6) for z in latent[i]])

    np.save(os.path.join(args.out, 'latent_codes.npy'), latent)

    # summary
    lines = [
        "AE Deep Latent-Space Clustering — summary",
        "=" * 44,
        f"input            : {args.input}",
        f"model            : {args.model}   latent_dim={latent.shape[1]}   epochs={args.epochs}",
        f"waveforms        : {X.shape[0]}   length={length}",
        f"clustering        : {args.algorithm}",
        f"projection       : {proj_name}",
        f"final loss       : {loss_curve[-1]:.6f}",
        f"n_clusters       : {len(valid)}",
        f"noise points     : {int(np.sum(labels == -1))}",
        "",
        "cluster sizes:",
    ]
    for l in valid:
        cnt = int(np.sum(labels == l))
        lines.append(f"  C{l}: {cnt}  ({100*cnt/len(labels):.1f}%)")
    lines.append("")
    lines.append("quality metrics:")
    if m:
        lines.append(f"  silhouette        : {m['silhouette']:.4f}  (higher better, max 1)")
        lines.append(f"  calinski_harabasz : {m['calinski_harabasz']:.1f}  (higher better)")
        lines.append(f"  davies_bouldin    : {m['davies_bouldin']:.4f}  (lower better)")
    else:
        lines.append("  (need >= 2 clusters to compute)")
    summary = "\n".join(lines)
    with open(os.path.join(args.out, 'summary.txt'), 'w') as f:
        f.write(summary + "\n")

    print("[5/5] Done.\n")
    print(summary)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Deep latent-space clustering of Mistras AE waveforms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('input', help='path to a .DTA file')
    ap.add_argument('--out', default='ae_cluster_out', help='output directory')
    # representation learning
    ap.add_argument('--model', choices=['cae', 'vae'], default='cae')
    ap.add_argument('--latent-dim', type=int, default=16, dest='latent_dim')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=64, dest='batch_size')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--beta', type=float, default=1.0, help='VAE KL weight')
    ap.add_argument('--length', type=int, default=1024, dest='fixed_length',
                    help='waveform length (padded/truncated, rounded up to x8)')
    ap.add_argument('--max-waveforms', type=int, default=2000, dest='max_waveforms')
    ap.add_argument('--keep-pretrigger', action='store_true', dest='keep_pretrigger')
    ap.add_argument('--channel', type=int, default=None, help='restrict to one channel')
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    # clustering
    ap.add_argument('--algorithm', choices=['kmeans', 'gmm', 'hdbscan', 'dbscan'],
                    default='kmeans')
    ap.add_argument('--clusters', type=int, default=4, help='for kmeans/gmm')
    ap.add_argument('--eps', type=float, default=0.8, help='for dbscan')
    ap.add_argument('--min-samples', type=int, default=10, dest='min_samples',
                    help='for dbscan/hdbscan')
    ap.add_argument('--projection', choices=['pca', 'tsne'], default='pca')
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit("PyTorch is required. Install with:  pip install torch")

    X, meta, length = load_waveforms(args.input, args.channel, args.max_waveforms,
                                     args.fixed_length, args.keep_pretrigger)
    latent, loss_curve = train(args, X, length)
    Z, labels = cluster_latent(args, latent)
    m, valid = metrics(Z, labels)
    proj, proj_name = project_2d(args, latent)
    save_outputs(args, X, meta, length, latent, loss_curve, Z, labels,
                 proj, proj_name, m, valid)


if __name__ == '__main__':
    main()
