#!/usr/bin/env python3
"""Standalone deep latent-space clustering for Mistras AE waveforms.

Pipeline (runs entirely on your machine, no frontend/backend):

    .DTA waveforms  ->  feature rep (waveform / fft / both)
                    ->  Autoencoder (CAE / VAE)  ->  latent space
                    ->  2D embedding (UMAP / t-SNE / PCA)
                    ->  clustering (KMeans / GMM / HDBSCAN / DBSCAN)
                    ->  medoid prototype waveforms

WHY FREQUENCY MATTERS for AE: damage mechanisms (matrix cracking, fiber
breakage, delamination) separate mainly by *frequency content*. Raw
time-domain autoencoders tend to encode amplitude/decay (continuous) and
give poorly separated clusters. Use --feature both (default) or fft.

Outputs (saved into --out):
    loss_curve.png        training reconstruction loss per epoch
    latent_scatter.png    2D embedding, colored by cluster
    prototypes.png        medoid (representative) waveform of each cluster
    cluster_labels.csv    per-waveform: index, channel, time, cluster, latent
    latent_codes.npy      raw latent matrix (N x latent_dim)
    summary.txt           run config + cluster sizes + quality metrics

Examples:
    python tools/ae_deep_cluster.py data.DTA --feature both --clusters 4
    python tools/ae_deep_cluster.py data.DTA --feature fft --algorithm hdbscan --projection umap
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


# Parametric AE hit features (Mistras field -> readable label) used to give
# the deep clusters a physical interpretation.
FEATURE_FIELDS = [
    ('AMP', 'amplitude_dB'), ('ENER', 'energy'), ('ABS-ENERGY', 'abs_energy'),
    ('RISE', 'rise_us'), ('DURATION', 'duration_us'), ('COUN', 'counts'),
    ('A-FRQ', 'avg_freq_kHz'), ('P-FRQ', 'peak_freq_kHz'),
    ('FRQ-C', 'centroid_freq_kHz'), ('R-FRQ', 'rev_freq_kHz'),
    ('I-FRQ', 'init_freq_kHz'),
]


def _extract_feats(rec_row):
    if rec_row is None:
        return {}
    names = rec_row.dtype.names
    return {label: float(rec_row[f]) for f, label in FEATURE_FIELDS if f in names}


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def round_up_multiple(n, m):
    return int(np.ceil(n / m) * m)


def _waveform_to_features(v, L, feature):
    """Build the model input channels for one (peak-normalized) waveform v.

    Returns an array of shape (C, L):
        waveform -> 1 channel  (time domain)
        fft      -> 1 channel  (log-magnitude spectrum, resampled to L)
        both     -> 2 channels (time + spectrum)
    """
    time_ch = v  # already length L, peak-normalized

    if feature == 'waveform':
        return time_ch[None, :]

    # log-magnitude spectrum, resampled to length L so conv arch is unchanged
    spec = np.log1p(np.abs(np.fft.rfft(v)))          # length L//2 + 1
    xp = np.linspace(0, 1, len(spec))
    xq = np.linspace(0, 1, L)
    spec_rs = np.interp(xq, xp, spec).astype(np.float32)
    smax = np.max(np.abs(spec_rs))
    if smax > 0:
        spec_rs = spec_rs / smax

    if feature == 'fft':
        return spec_rs[None, :]
    return np.stack([time_ch, spec_rs])              # both -> (2, L)


def load_waveforms(dta_path, channel, max_waveforms, fixed_length,
                   keep_pretrigger, feature):
    """Return (X, X_wave, meta, L, C).

    X      : model input (N, C, L) for the chosen feature
    X_wave : peak-normalized time-domain waveforms (N, L) for plotting
    """
    print(f"[1/5] Reading {dta_path} ...")
    rec, wfm = read_bin(dta_path)
    if not isinstance(wfm, np.recarray) or len(wfm) == 0:
        raise SystemExit("No waveforms found in this file.")
    print(f"      hits={len(rec)}  waveforms={len(wfm)}")

    mask = np.ones(len(wfm), dtype=bool)
    if channel is not None:
        mask &= wfm['CH'] == channel
    idx_all = np.where(mask)[0]
    if len(idx_all) == 0:
        raise SystemExit(f"No waveforms on channel {channel}.")

    # ---------- auto-detect waveform length ----------
    if fixed_length <= 0:
        sample = idx_all[np.linspace(0, len(idx_all) - 1, min(100, len(idx_all))).astype(int)]
        lengths = []
        for i in sample:
            _, V = get_waveform_data(wfm[i])
            if not keep_pretrigger and wfm[i]['TDLY'] < 0:
                V = V[abs(int(wfm[i]['TDLY'])):]
            lengths.append(len(V))
        med = int(np.median(lengths))
        L = round_up_multiple(max(64, med), 16)
        print(f"      auto-detected waveform length: median={med} -> padded to {L}")
    else:
        L = round_up_multiple(max(64, fixed_length), 16)

    if len(idx_all) > max_waveforms:
        sel = np.linspace(0, len(idx_all) - 1, max_waveforms).astype(int)
        idx_all = idx_all[sel]

    # Align parametric hit features (in `rec`) to each waveform for physical
    # interpretation. wfm and rec are usually 1:1 by index; otherwise match by
    # nearest timestamp.
    same_len = isinstance(rec, np.recarray) and len(rec) == len(wfm)
    rec_times = (rec['SSSSSSSS.mmmuuun']
                 if isinstance(rec, np.recarray) and len(rec)
                 and 'SSSSSSSS.mmmuuun' in rec.dtype.names and not same_len
                 else None)

    feats, waves, meta = [], [], []
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
        v = v.astype(np.float32)

        rec_row = None
        if same_len:
            rec_row = rec[i]
        elif rec_times is not None:
            rec_row = rec[int(np.argmin(np.abs(rec_times - float(row['SSSSSSSS.mmmuuun']))))]

        feats.append(_waveform_to_features(v, L, feature))
        waves.append(v)
        meta.append({
            'index': int(i),
            'channel': int(row['CH']),
            'time': float(row['SSSSSSSS.mmmuuun']),
            'sample_rate': float(row['SRATE']),
            'feat': _extract_feats(rec_row),
        })

    if len(feats) < 4:
        raise SystemExit("Not enough valid waveforms (need >= 4).")
    X = np.stack(feats)            # (N, C, L)
    X_wave = np.stack(waves)       # (N, L)
    C = X.shape[1]
    print(f"      using {X.shape[0]} waveforms, length={L}, feature={feature} "
          f"({C} channel{'s' if C > 1 else ''}), "
          f"pretrigger={'kept' if keep_pretrigger else 'trimmed'}")
    return X, X_wave, meta, L, C


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_models(torch, nn, length, latent_dim, in_ch):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(in_ch, 32, 7, stride=2, padding=3),
                nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 64, 7, stride=2, padding=3),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 128, 5, stride=2, padding=2),
                nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 128, 5, stride=2, padding=2),
                nn.BatchNorm1d(128), nn.ReLU(),
            )
            self.flat_len = (length // 16) * 128
            self.fc = nn.Linear(self.flat_len, latent_dim)

        def forward(self, x):
            return self.fc(self.conv(x).flatten(1))

    class Decoder(nn.Module):
        def __init__(self, flat_len):
            super().__init__()
            self.length = length
            self.fc = nn.Linear(latent_dim, flat_len)
            self.deconv = nn.Sequential(
                nn.ConvTranspose1d(128, 128, 5, 2, 2, output_padding=1),
                nn.BatchNorm1d(128), nn.ReLU(),
                nn.ConvTranspose1d(128, 64, 5, 2, 2, output_padding=1),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.ConvTranspose1d(64, 32, 7, 2, 3, output_padding=1),
                nn.BatchNorm1d(32), nn.ReLU(),
                nn.ConvTranspose1d(32, in_ch, 7, 2, 3, output_padding=1),
            )

        def forward(self, z):
            h = self.fc(z).view(z.size(0), 128, self.length // 16)
            return self.deconv(h)[:, :, :self.length]

    class CAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = Encoder(); self.dec = Decoder(self.enc.flat_len)
        def forward(self, x):
            z = self.enc(x); return self.dec(z), z
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


def train(args, X, length, C):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(args.device if args.device != 'auto'
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"[2/5] Training {args.model.upper()} on {device} "
          f"(latent={args.latent_dim}, epochs={args.epochs}, lr={args.lr}) ...")

    CAE, VAE = build_models(torch, nn, length, args.latent_dim, C)
    net = (VAE() if args.model == 'vae' else CAE()).to(device)
    print(f"      model parameters: {sum(p.numel() for p in net.parameters()):,}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    X_t = torch.from_numpy(X).to(device)            # (N, C, L)
    N = X.shape[0]
    bs = min(args.batch_size, N)
    n_batches = max(1, N // bs)

    net.train()
    loss_curve, best, patience = [], float('inf'), 0
    for epoch in range(args.epochs):
        perm = torch.randperm(N, device=device)
        run = 0.0
        for b in range(n_batches):
            xb = X_t[perm[b * bs:(b + 1) * bs]]
            opt.zero_grad()
            if args.model == 'vae':
                recon, mu, logvar = net(xb)
                loss = F.mse_loss(recon, xb) + args.beta * (
                    -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
            else:
                recon, _ = net(xb)
                loss = F.mse_loss(recon, xb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            run += float(loss.item())
        sched.step()
        avg = run / n_batches
        loss_curve.append(avg)
        if avg < best * 0.999:
            best, patience = avg, 0
        else:
            patience += 1
        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            print(f"      epoch {epoch + 1:3d}/{args.epochs}  loss={avg:.6f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")
        if args.early_stop > 0 and patience >= args.early_stop:
            print(f"      early stop at epoch {epoch + 1}")
            break

    net.eval()
    with torch.no_grad():
        latent = net.encode(X_t).cpu().numpy()
    print(f"      final loss={loss_curve[-1]:.6f}  trained {len(loss_curve)} epochs")
    return latent, loss_curve


# --------------------------------------------------------------------------- #
# Embedding + clustering
# --------------------------------------------------------------------------- #
def embed_2d(args, latent):
    from sklearn.decomposition import PCA
    N = latent.shape[0]
    if latent.shape[1] <= 2:
        p = latent if latent.shape[1] == 2 else np.column_stack([latent[:, 0], np.zeros(N)])
        return p, 'latent'

    if args.projection == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42,
                                n_neighbors=args.umap_neighbors, min_dist=args.umap_mindist)
            return reducer.fit_transform(latent), 'UMAP'
        except ImportError:
            print("      [tip] umap-learn not installed; falling back to t-SNE. "
                  "For best AE clustering:  pip install umap-learn")
            args.projection = 'tsne'

    if args.projection == 'tsne':
        from sklearn.manifold import TSNE
        perp = float(min(30, max(2, N // 4), N - 1))
        return TSNE(2, random_state=42, perplexity=perp, init='pca').fit_transform(latent), 't-SNE'

    return PCA(2, random_state=42).fit_transform(latent), 'PCA'


def cluster(args, latent, emb2d):
    """Return (space, labels) where `space` is the array clustering ran on
    (used for metrics). KMeans/GMM use the latent; density methods use the
    2D embedding by default (curse-of-dimensionality + matches AE literature
    latent->UMAP->HDBSCAN)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.mixture import GaussianMixture

    print(f"[3/5] Clustering with {args.algorithm} ...")
    Zlat = StandardScaler().fit_transform(latent)
    Zemb = StandardScaler().fit_transform(emb2d)

    density = args.algorithm in ('hdbscan', 'dbscan')
    use_embed = density and args.density_space == 'embed'
    space = Zemb if use_embed else Zlat
    where = '2D embedding' if use_embed else 'latent'

    if args.algorithm == 'kmeans':
        labels = KMeans(args.clusters, random_state=42, n_init=10).fit_predict(space)
    elif args.algorithm == 'gmm':
        labels = GaussianMixture(args.clusters, random_state=42, n_init=3).fit_predict(space)
    elif args.algorithm == 'dbscan':
        labels = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit_predict(space)
    elif args.algorithm == 'hdbscan':
        from sklearn.cluster import HDBSCAN
        labels = HDBSCAN(min_cluster_size=max(args.min_cluster_size, 2),
                         min_samples=args.min_samples).fit_predict(space)
    else:
        raise SystemExit(f"Unknown algorithm: {args.algorithm}")

    valid = sorted(l for l in set(labels) if l >= 0)
    noise = int(np.sum(labels == -1))
    print(f"      ran on {where}: clusters={len(valid)}  noise={noise}/{len(labels)}")

    if density and len(valid) == 0:
        print("\n      WARNING: 0 clusters. Try:")
        print("        --projection umap              (pip install umap-learn; best for AE)")
        print("        --min-cluster-size 15          (lower = more clusters)")
        print("        --feature both  or  --feature fft")
        print("        --algorithm kmeans --clusters 4   (baseline that always returns clusters)\n")
    return space, labels


def metrics_on(space, labels):
    from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                                 davies_bouldin_score)
    m = {}
    valid = sorted(l for l in set(labels) if l >= 0)
    nn_mask = labels >= 0
    if len(valid) >= 2 and np.sum(nn_mask) > len(valid):
        m['silhouette'] = silhouette_score(space[nn_mask], labels[nn_mask])
        m['calinski_harabasz'] = calinski_harabasz_score(space[nn_mask], labels[nn_mask])
        m['davies_bouldin'] = davies_bouldin_score(space[nn_mask], labels[nn_mask])
    return m, valid


def scan_k(latent, kmax):
    """Sweep KMeans k and report latent-space silhouette to help pick k."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    Z = StandardScaler().fit_transform(latent)
    print("      k-scan (KMeans, silhouette on latent):")
    best_k, best_s = None, -1
    for k in range(2, kmax + 1):
        lab = KMeans(k, random_state=42, n_init=10).fit_predict(Z)
        s = silhouette_score(Z, lab)
        flag = ''
        if s > best_s:
            best_s, best_k, flag = s, k, '  <- best'
        print(f"        k={k}: silhouette={s:.4f}{flag}")
    print(f"      suggested clusters: {best_k}\n")


def latent_diagnostic(latent):
    """Print how much variance the top latent PCs explain — tells you whether
    real structure exists or the latent is one diffuse blob."""
    from sklearn.decomposition import PCA
    k = min(5, latent.shape[1])
    pca = PCA(k, random_state=42).fit(latent)
    ev = pca.explained_variance_ratio_
    print(f"      latent PCA top-{k} explained variance: "
          + ", ".join(f"{v*100:.0f}%" for v in ev)
          + f"  (cum {ev.sum()*100:.0f}%)")


def characterize(args, X_wave, meta, length, labels, valid, plt, color):
    """Physically interpret clusters: mean FFT spectrum per cluster + a table
    of parametric AE hit features (peak frequency, energy, amplitude, ...).
    This is what turns black-box clusters into damage-mode statements."""
    out_lines = []

    # ----- mean spectrum per cluster (the key frequency-separation plot) -----
    sr = float(np.median([md['sample_rate'] for md in meta]))
    freqs_khz = np.fft.rfftfreq(length, 1.0 / sr) / 1000.0
    spectra = np.abs(np.fft.rfft(X_wave, axis=1))      # (N, L//2+1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for l in valid:
        ms = spectra[labels == l].mean(axis=0)
        ax.plot(freqs_khz, ms, color=color(l), lw=1.3, label=f'C{l} (n={int(np.sum(labels==l))})')
    ax.set_xlabel('Frequency (kHz)'); ax.set_ylabel('Mean |FFT| (norm. waveforms)')
    ax.set_title('Cluster mean frequency spectra — physical signature')
    ax.legend(fontsize=9); ax.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'cluster_spectra.png'), dpi=150); plt.close()

    # ----- parametric feature table per cluster -----
    keys = [lbl for _, lbl in FEATURE_FIELDS
            if any(lbl in md.get('feat', {}) for md in meta)]
    if keys:
        rows = []
        for l in valid:
            members = [i for i in range(len(meta)) if labels[i] == l]
            row = {'cluster': l, 'count': len(members)}
            for k in keys:
                vals = np.array([meta[i]['feat'][k] for i in members
                                 if k in meta[i]['feat']], dtype=float)
                row[k] = (float(np.mean(vals)), float(np.std(vals))) if len(vals) else (np.nan, np.nan)
            rows.append(row)

        with open(os.path.join(args.out, 'cluster_features.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            header = ['cluster', 'count'] + [f'{k}_mean' for k in keys] + [f'{k}_std' for k in keys]
            w.writerow(header)
            for r in rows:
                w.writerow([r['cluster'], r['count']]
                           + [round(r[k][0], 3) for k in keys]
                           + [round(r[k][1], 3) for k in keys])

        # compact text table for the summary, highlighting frequency + energy
        hi = [k for k in ('peak_freq_kHz', 'centroid_freq_kHz', 'avg_freq_kHz',
                          'amplitude_dB', 'energy', 'duration_us', 'rise_us') if k in keys]
        out_lines.append("")
        out_lines.append("physical interpretation (mean per cluster):")
        out_lines.append("  cluster  " + "  ".join(f"{k:>16}" for k in hi))
        for r in rows:
            out_lines.append(f"  C{r['cluster']:<6} "
                             + "  ".join(f"{r[k][0]:>16.2f}" for k in hi))
        out_lines.append("  -> low centroid/peak freq usually = delamination/debonding;")
        out_lines.append("     high freq = matrix cracking / fiber breakage (verify with your material).")
    return out_lines


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def save_outputs(args, X_wave, meta, length, C, latent, loss_curve, space, labels,
                 emb2d, proj_name, m, m_embed, valid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    cmap = plt.get_cmap('tab10')

    def color(l):
        return (0.3, 0.3, 0.3, 0.4) if l == -1 else cmap(l % 10)

    print(f"[4/5] Writing results to {args.out}/ ...")

    # loss curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(loss_curve) + 1), loss_curve, color='#0891b2', lw=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title(f'{args.model.upper()} training loss  (final={loss_curve[-1]:.6f})')
    ax.grid(alpha=0.3)
    if loss_curve[-1] > loss_curve[0] * 0.5:
        ax.annotate('still dropping — try more --epochs',
                    xy=(len(loss_curve), loss_curve[-1]), fontsize=9, color='red',
                    xytext=(-120, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='red'))
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'loss_curve.png'), dpi=150); plt.close()

    # 2D embedding scatter
    fig, ax = plt.subplots(figsize=(8, 7))
    for l in sorted(set(labels)):
        pts = emb2d[labels == l]
        ax.scatter(pts[:, 0], pts[:, 1], s=12, color=color(l), alpha=0.7,
                   edgecolors='none',
                   label=('noise' if l == -1 else f'C{l} (n={int(np.sum(labels == l))})'))
    ax.set_xlabel(f'{proj_name}-1'); ax.set_ylabel(f'{proj_name}-2')
    title = f'{proj_name} embedding — {args.algorithm} on {args.feature}'
    if 'silhouette' in m:
        title += f'  sil(latent)={m["silhouette"]:.3f}'
    ax.set_title(title); ax.legend(markerscale=2, fontsize=9, loc='best')
    ax.grid(alpha=0.2); plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'latent_scatter.png'), dpi=150); plt.close()

    # medoid prototype waveforms (always time-domain for interpretability)
    protos = []
    for l in valid:
        members = np.where(labels == l)[0]
        centroid = space[members].mean(axis=0)
        medoid = members[int(np.argmin(np.linalg.norm(space[members] - centroid, axis=1)))]
        protos.append((l, medoid))

    if protos:
        ncol = min(3, len(protos)); nrow = int(np.ceil(len(protos) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 2.5 * nrow), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        for ax, (l, medoid) in zip(axes.flat, protos):
            ax.axis('on')
            ax.plot(X_wave[medoid], color=color(l), lw=0.8)
            cnt = int(np.sum(labels == l))
            ax.set_title(f'C{l}  n={cnt} ({100*cnt/len(labels):.1f}%)  '
                         f'CH{meta[medoid]["channel"]} #{meta[medoid]["index"]}', fontsize=9)
            ax.set_xlabel('sample'); ax.set_ylabel('norm. V'); ax.tick_params(labelsize=7)
        plt.suptitle('Cluster prototype (medoid) waveforms', fontsize=12, y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'prototypes.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # labels CSV
    with open(os.path.join(args.out, 'cluster_labels.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['wfm_index', 'channel', 'time_s', 'sample_rate', 'cluster']
                   + [f'z{i}' for i in range(latent.shape[1])])
        for i, md in enumerate(meta):
            w.writerow([md['index'], md['channel'], md['time'], md['sample_rate'],
                        int(labels[i])] + [round(float(z), 6) for z in latent[i]])

    np.save(os.path.join(args.out, 'latent_codes.npy'), latent)

    # physical characterization (spectra + parametric features)
    char_lines = characterize(args, X_wave, meta, length, labels, valid, plt, color)

    # summary
    lines = [
        "AE Deep Latent-Space Clustering — summary",
        "=" * 44,
        f"input            : {args.input}",
        f"feature          : {args.feature} ({C} ch)",
        f"model            : {args.model}   latent_dim={latent.shape[1]}   "
        f"epochs={len(loss_curve)}/{args.epochs}",
        f"waveforms        : {X_wave.shape[0]}   length={length}",
        f"clustering       : {args.algorithm}   embedding={proj_name}",
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
    lines.append("quality metrics (report the LATENT one in papers):")
    if m:
        lines.append(f"  silhouette (latent)     : {m['silhouette']:.4f}  (higher=better, max 1)")
        lines.append(f"  calinski_harabasz       : {m['calinski_harabasz']:.1f}  (higher=better)")
        lines.append(f"  davies_bouldin          : {m['davies_bouldin']:.4f}  (lower=better)")
    else:
        lines.append("  (need >= 2 non-noise clusters)")
    if m_embed:
        lines.append(f"  silhouette ({proj_name}, optimistic): {m_embed['silhouette']:.4f}  "
                     f"(inflated by {proj_name}; for visualization only)")
    lines += char_lines
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

    g = ap.add_argument_group('representation learning')
    g.add_argument('--feature', choices=['waveform', 'fft', 'both'], default='both',
                   help='input representation; fft/both separate AE damage modes better')
    g.add_argument('--model', choices=['cae', 'vae'], default='cae')
    g.add_argument('--latent-dim', type=int, default=16, dest='latent_dim')
    g.add_argument('--epochs', type=int, default=100)
    g.add_argument('--early-stop', type=int, default=20, dest='early_stop',
                   help='stop if loss does not improve for N epochs (0=off)')
    g.add_argument('--batch-size', type=int, default=64, dest='batch_size')
    g.add_argument('--lr', type=float, default=5e-4)
    g.add_argument('--beta', type=float, default=1.0, help='VAE KL weight')
    g.add_argument('--length', type=int, default=0, dest='fixed_length',
                   help='waveform length; 0=auto-detect from data')
    g.add_argument('--max-waveforms', type=int, default=5000, dest='max_waveforms')
    g.add_argument('--keep-pretrigger', action='store_true', dest='keep_pretrigger')
    g.add_argument('--channel', type=int, default=None, help='restrict to one channel')
    g.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')

    g = ap.add_argument_group('embedding + clustering')
    g.add_argument('--projection', choices=['umap', 'tsne', 'pca'], default='pca',
                   help='2D embedding; umap recommended for density clustering')
    g.add_argument('--umap-neighbors', type=int, default=15, dest='umap_neighbors')
    g.add_argument('--umap-mindist', type=float, default=0.1, dest='umap_mindist')
    g.add_argument('--algorithm', choices=['kmeans', 'gmm', 'hdbscan', 'dbscan'],
                   default='kmeans')
    g.add_argument('--clusters', type=int, default=4, help='for kmeans/gmm')
    g.add_argument('--density-space', choices=['embed', 'latent'], default='embed',
                   dest='density_space',
                   help='space hdbscan/dbscan run on (embed=2D, robust)')
    g.add_argument('--eps', type=float, default=0.3, help='for dbscan')
    g.add_argument('--min-samples', type=int, default=5, dest='min_samples',
                   help='hdbscan/dbscan: lower = less conservative')
    g.add_argument('--min-cluster-size', type=int, default=30, dest='min_cluster_size',
                   help='hdbscan: lower = more clusters')
    g.add_argument('--scan-k', type=int, default=0, dest='scan_k',
                   help='sweep KMeans k=2..N on the latent and report silhouette, then continue')
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit("PyTorch required.  pip install torch")

    from sklearn.preprocessing import StandardScaler

    X, X_wave, meta, length, C = load_waveforms(
        args.input, args.channel, args.max_waveforms,
        args.fixed_length, args.keep_pretrigger, args.feature)
    latent, loss_curve = train(args, X, length, C)
    latent_diagnostic(latent)
    if args.scan_k >= 2:
        scan_k(latent, args.scan_k)
    emb2d, proj_name = embed_2d(args, latent)
    space, labels = cluster(args, latent, emb2d)

    # Honest metrics on the latent space (comparable across runs); the
    # embedding-space silhouette is reported separately and flagged optimistic.
    m, valid = metrics_on(StandardScaler().fit_transform(latent), labels)
    m_embed = None
    if args.algorithm in ('hdbscan', 'dbscan') and args.density_space == 'embed':
        m_embed, _ = metrics_on(StandardScaler().fit_transform(emb2d), labels)

    save_outputs(args, X_wave, meta, length, C, latent, loss_curve,
                 space, labels, emb2d, proj_name, m, m_embed, valid)


if __name__ == '__main__':
    main()
