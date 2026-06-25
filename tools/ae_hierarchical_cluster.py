#!/usr/bin/env python3
"""Two-stage hierarchical clustering for Mistras AE waveforms.

Pipeline:

    Stage 1 — Waveform shape grouping:
        .DTA waveforms  ->  peak-normalize + pad
                        ->  CAE/VAE (time-domain only)  ->  latent codes
                        ->  coarse clustering  ->  K1 shape groups

    Stage 2 — Within-group refinement:
        For each shape group:
            extract FFT spectral features + parametric AE hit features
            ->  standardize  ->  fine clustering  ->  sub-labels

    Final label = (stage1_group, stage2_sub)

Rationale: waveforms from the same damage mechanism share a similar
time-domain shape (rise pattern, decay envelope, oscillation character).
Grouping by shape first avoids mixing fundamentally different source types.
Within each shape group, frequency and parametric features tease apart
subtypes that the time-domain autoencoder treats as equivalent.

Outputs (saved into --out):
    loss_curve.png              CAE/VAE training loss
    stage1_scatter.png          stage-1 coarse groups in 2D embedding
    stage2_scatter.png          final hierarchical labels in 2D embedding
    stage1_prototypes.png       medoid waveform per coarse group
    stage2_prototypes.png       medoid waveform per final sub-cluster
    cluster_spectra.png         mean frequency spectrum per final cluster
    cluster_timeline.png        event amplitude vs time, colored by final label
    cluster_amplitude_vs_freq.png   amplitude vs peak-frequency scatter
    cluster_labels.csv          per-waveform labels (stage1, stage2, final)
    latent_codes.npy            raw latent matrix
    summary.txt                 run config + cluster sizes + quality metrics

Examples:
    python tools/ae_hierarchical_cluster.py data.DTA
    python tools/ae_hierarchical_cluster.py data.DTA --stage1-clusters 3 --stage2-clusters 2
    python tools/ae_hierarchical_cluster.py data.DTA --stage2-features fft+param --algorithm hdbscan
"""

import os
import sys
import csv
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


def _extract_feats(rec_row):
    if rec_row is None:
        return {}
    names = rec_row.dtype.names
    return {label: float(rec_row[f]) for f, label in FEATURE_FIELDS if f in names}


# --------------------------------------------------------------------------- #
# Entropy
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Denoising (reused from ae_deep_cluster)
# --------------------------------------------------------------------------- #
def _wavelet_denoise(v, wavelet='db4', level=0, mode='soft'):
    import pywt
    n = len(v)
    if n < 8:
        return v
    v = np.array(v, dtype=np.float64)
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(n, w.dec_len)
    lvl = max_level if level <= 0 else min(level, max_level)
    if lvl < 1:
        return v
    coeffs = pywt.wavedec(v, w, mode='periodization', level=lvl)
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail)) / 0.6745 if detail.size else 0.0
    if sigma > 0:
        uthresh = sigma * np.sqrt(2.0 * np.log(n))
        coeffs[1:] = [pywt.threshold(c, uthresh, mode=mode) for c in coeffs[1:]]
    rec = pywt.waverec(coeffs, w, mode='periodization')
    return rec[:n].astype(np.float32)


def _bandpass(v, sr, low_hz, high_hz, order=4):
    from scipy.signal import butter, sosfiltfilt
    n = len(v)
    if n < 3 * (order + 1):
        return v
    nyq = sr / 2.0
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 0.999)
    if low >= high:
        return v
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    pad = min(n - 1, 3 * (sos.shape[0] + 1))
    return sosfiltfilt(sos, v, padlen=pad).astype(np.float32)


def make_denoiser(args):
    if getattr(args, 'denoise', 'none') == 'none':
        return None
    do_band = 'bandpass' in args.denoise
    do_wave = 'wavelet' in args.denoise
    low_hz = args.denoise_band[0] * 1000.0
    high_hz = args.denoise_band[1] * 1000.0
    if do_wave:
        try:
            import pywt  # noqa: F401
        except ImportError:
            print("      [warning] PyWavelets not installed; wavelet denoising disabled.")
            do_wave = False
            if not do_band:
                return None

    def denoiser(v, sr):
        if do_band:
            v = _bandpass(v, sr, low_hz, high_hz, order=args.denoise_order)
        if do_wave:
            v = _wavelet_denoise(v, wavelet=args.denoise_wavelet,
                                 level=args.denoise_level, mode=args.denoise_mode)
        return v
    return denoiser


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def round_up_multiple(n, m):
    return int(np.ceil(n / m) * m)


def load_waveforms(dta_path, channel, max_waveforms, fixed_length,
                   keep_pretrigger, denoiser=None):
    print(f"[1/6] Reading {dta_path} ...")
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

    same_len = isinstance(rec, np.recarray) and len(rec) == len(wfm)
    rec_times = (rec['SSSSSSSS.mmmuuun']
                 if isinstance(rec, np.recarray) and len(rec)
                 and 'SSSSSSSS.mmmuuun' in rec.dtype.names and not same_len
                 else None)

    waves, meta = [], []
    for i in idx_all:
        row = wfm[i]
        t, V = get_waveform_data(row)
        if not keep_pretrigger and row['TDLY'] < 0:
            trim = abs(int(row['TDLY']))
            V = V[trim:]
        if len(V) == 0:
            continue
        V_raw = V
        if denoiser is not None:
            V = denoiser(V, float(row['SRATE']))
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

        entropy_val = compute_waveform_entropy(V_raw)
        hit_feats = _extract_feats(rec_row)
        hit_feats['entropy_nats'] = entropy_val

        waves.append(v)
        meta.append({
            'index': int(i),
            'channel': int(row['CH']),
            'time': float(row['SSSSSSSS.mmmuuun']),
            'sample_rate': float(row['SRATE']),
            'feat': hit_feats,
        })

    if len(waves) < 4:
        raise SystemExit("Not enough valid waveforms (need >= 4).")
    X_wave = np.stack(waves)
    print(f"      using {X_wave.shape[0]} waveforms, length={L}, "
          f"pretrigger={'kept' if keep_pretrigger else 'trimmed'}")
    return X_wave, meta, L


# --------------------------------------------------------------------------- #
# CAE/VAE models (time-domain only, 1 input channel)
# --------------------------------------------------------------------------- #
def build_models(torch, nn, length, latent_dim):
    in_ch = 1

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


def train_autoencoder(args, X_wave, length):
    """Train on time-domain waveforms only (1 channel) to learn shape."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(args.device if args.device != 'auto'
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"[2/6] Training {args.model.upper()} on {device} "
          f"(time-domain only, latent={args.latent_dim}, epochs={args.epochs}, lr={args.lr}) ...")

    CAE, VAE = build_models(torch, nn, length, args.latent_dim)
    net = (VAE() if args.model == 'vae' else CAE()).to(device)
    print(f"      model parameters: {sum(p.numel() for p in net.parameters()):,}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    X_t = torch.from_numpy(X_wave[:, None, :]).to(device)  # (N, 1, L)
    N = X_wave.shape[0]
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
# Stage 1: coarse clustering on time-domain latent
# --------------------------------------------------------------------------- #
def stage1_cluster(args, latent):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    print(f"[3/6] Stage 1 — coarse shape clustering ({args.stage1_algorithm}, "
          f"k={args.stage1_clusters}) ...")
    Z = StandardScaler().fit_transform(latent)

    if args.stage1_algorithm == 'kmeans':
        labels = KMeans(args.stage1_clusters, random_state=42, n_init=10).fit_predict(Z)
    elif args.stage1_algorithm == 'gmm':
        labels = GaussianMixture(args.stage1_clusters, random_state=42, n_init=3).fit_predict(Z)
    elif args.stage1_algorithm == 'hdbscan':
        from sklearn.cluster import HDBSCAN
        labels = HDBSCAN(min_cluster_size=max(args.stage1_min_cluster_size, 2),
                         min_samples=args.stage1_min_samples).fit_predict(Z)
    else:
        raise SystemExit(f"Unknown stage1 algorithm: {args.stage1_algorithm}")

    valid = sorted(l for l in set(labels) if l >= 0)
    noise = int(np.sum(labels == -1))
    print(f"      coarse groups: {len(valid)}  noise: {noise}/{len(labels)}")
    for l in valid:
        cnt = int(np.sum(labels == l))
        print(f"        G{l}: {cnt} ({100 * cnt / len(labels):.1f}%)")
    return Z, labels


# --------------------------------------------------------------------------- #
# Stage 2: within-group refinement using FFT + parametric features
# --------------------------------------------------------------------------- #
def _build_stage2_features(X_wave, meta, indices, length, feature_mode):
    """Build a feature matrix for the given subset indices.

    feature_mode:
        'fft'       — FFT magnitude spectrum (resampled to 64 bins)
        'param'     — parametric AE hit features only
        'fft+param' — concatenation of both (recommended)
    """
    N_FFT_BINS = 64
    features = []
    feat_names = []

    if 'fft' in feature_mode:
        spectra = np.abs(np.fft.rfft(X_wave[indices], axis=1))  # (n, L//2+1)
        xp = np.linspace(0, 1, spectra.shape[1])
        xq = np.linspace(0, 1, N_FFT_BINS)
        spec_binned = np.array([np.interp(xq, xp, s) for s in spectra], dtype=np.float32)
        smax = np.max(np.abs(spec_binned), axis=1, keepdims=True)
        smax[smax == 0] = 1.0
        spec_binned = spec_binned / smax
        features.append(spec_binned)
        feat_names += [f'fft_bin{i}' for i in range(N_FFT_BINS)]

    if 'param' in feature_mode:
        param_keys = [lbl for _, lbl in FEATURE_FIELDS
                      if any(lbl in meta[j]['feat'] for j in indices)]
        param_keys.append('entropy_nats')
        param_mat = []
        for j in indices:
            row = [meta[j]['feat'].get(k, np.nan) for k in param_keys]
            param_mat.append(row)
        param_mat = np.array(param_mat, dtype=np.float32)
        col_mean = np.nanmean(param_mat, axis=0)
        nan_mask = np.isnan(param_mat)
        param_mat[nan_mask] = np.take(col_mean, np.where(nan_mask)[1]) if nan_mask.any() else 0
        features.append(param_mat)
        feat_names += param_keys

    if not features:
        raise SystemExit(f"No features for stage2 mode '{feature_mode}'")

    return np.hstack(features), feat_names


def stage2_refine(args, X_wave, meta, length, stage1_labels):
    """Within each stage-1 group, sub-cluster on FFT + parametric features."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    print(f"[4/6] Stage 2 — within-group refinement ({args.stage2_algorithm}, "
          f"k={args.stage2_clusters}, features={args.stage2_features}) ...")

    N = len(stage1_labels)
    final_labels = np.full(N, -1, dtype=int)
    next_label = 0
    group_map = {}  # final_label -> (stage1_group, stage2_sub)

    valid_groups = sorted(l for l in set(stage1_labels) if l >= 0)

    for g in valid_groups:
        members = np.where(stage1_labels == g)[0]
        n_members = len(members)

        if n_members < max(4, args.stage2_clusters):
            for idx in members:
                final_labels[idx] = next_label
            group_map[next_label] = (g, 0)
            print(f"      G{g}: {n_members} samples -> 1 sub-cluster (too few to split)")
            next_label += 1
            continue

        F_mat, feat_names = _build_stage2_features(
            X_wave, meta, members, length, args.stage2_features)
        F_scaled = StandardScaler().fit_transform(F_mat)

        k2 = min(args.stage2_clusters, n_members // 2)
        k2 = max(k2, 2)

        if args.stage2_algorithm == 'kmeans':
            sub_labels = KMeans(k2, random_state=42, n_init=10).fit_predict(F_scaled)
        elif args.stage2_algorithm == 'gmm':
            sub_labels = GaussianMixture(k2, random_state=42, n_init=3).fit_predict(F_scaled)
        elif args.stage2_algorithm == 'hdbscan':
            from sklearn.cluster import HDBSCAN
            sub_labels = HDBSCAN(
                min_cluster_size=max(args.stage2_min_cluster_size, 2),
                min_samples=max(args.stage2_min_samples, 2)).fit_predict(F_scaled)
        else:
            raise SystemExit(f"Unknown stage2 algorithm: {args.stage2_algorithm}")

        sub_valid = sorted(l for l in set(sub_labels) if l >= 0)
        sub_noise = int(np.sum(sub_labels == -1))

        for s in sub_valid:
            sub_members = members[sub_labels == s]
            for idx in sub_members:
                final_labels[idx] = next_label
            group_map[next_label] = (g, s)
            next_label += 1

        if sub_noise > 0:
            noise_members = members[sub_labels == -1]
            for idx in noise_members:
                final_labels[idx] = -1

        print(f"      G{g}: {n_members} samples -> {len(sub_valid)} sub-clusters"
              + (f" + {sub_noise} noise" if sub_noise > 0 else ""))

    noise_from_s1 = np.sum(stage1_labels == -1)
    total_clusters = len(set(final_labels) - {-1})
    total_noise = int(np.sum(final_labels == -1))
    print(f"      final: {total_clusters} clusters, {total_noise} noise "
          f"(s1 noise: {noise_from_s1})")
    return final_labels, group_map


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #
def embed_2d(args, latent):
    from sklearn.decomposition import PCA
    N = latent.shape[0]
    dim = 2

    if args.projection == 'umap':
        try:
            import umap
            return umap.UMAP(n_components=dim, random_state=42,
                             n_neighbors=args.umap_neighbors,
                             min_dist=args.umap_mindist).fit_transform(latent), 'UMAP'
        except ImportError:
            print("      [tip] umap-learn not installed; falling back to t-SNE.")
            args.projection = 'tsne'

    if args.projection == 'tsne':
        from sklearn.manifold import TSNE
        perp = float(min(30, max(2, N // 4), N - 1))
        return TSNE(dim, random_state=42, perplexity=perp, init='pca').fit_transform(latent), 't-SNE'

    return PCA(dim, random_state=42).fit_transform(latent), 'PCA'


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(space, labels):
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
    print(f"      suggested stage1 clusters: {best_k}\n")


# --------------------------------------------------------------------------- #
# Output / visualization
# --------------------------------------------------------------------------- #
def save_outputs(args, X_wave, meta, length, latent, loss_curve, stage1_labels,
                 final_labels, group_map, emb2d, proj_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler

    os.makedirs(args.out, exist_ok=True)
    cmap = plt.get_cmap('tab20')

    def color(l):
        return (0.3, 0.3, 0.3, 0.4) if l == -1 else cmap(l % 20)

    print(f"[5/6] Writing results to {args.out}/ ...")

    # --- loss curve ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(loss_curve) + 1), loss_curve, color='#0891b2', lw=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title(f'{args.model.upper()} training loss (time-domain, final={loss_curve[-1]:.6f})')
    ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'loss_curve.png'), dpi=150); plt.close()

    # --- stage 1 scatter ---
    fig, ax = plt.subplots(figsize=(8, 7))
    for l in sorted(set(stage1_labels)):
        pts = emb2d[stage1_labels == l]
        ax.scatter(pts[:, 0], pts[:, 1], s=12, color=color(l), alpha=0.7,
                   edgecolors='none',
                   label=('noise' if l == -1 else f'G{l} (n={int(np.sum(stage1_labels == l))})'))
    ax.set_xlabel(f'{proj_name}-1'); ax.set_ylabel(f'{proj_name}-2')
    ax.set_title('Stage 1: coarse shape groups (time-domain latent)')
    ax.legend(markerscale=2, fontsize=9); ax.grid(alpha=0.2); plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'stage1_scatter.png'), dpi=150); plt.close()

    # --- stage 2 (final) scatter ---
    fig, ax = plt.subplots(figsize=(8, 7))
    final_valid = sorted(l for l in set(final_labels) if l >= 0)
    for l in sorted(set(final_labels)):
        pts = emb2d[final_labels == l]
        if l == -1:
            lbl = 'noise'
        else:
            g, s = group_map.get(l, ('?', '?'))
            lbl = f'G{g}.{s} (n={int(np.sum(final_labels == l))})'
        ax.scatter(pts[:, 0], pts[:, 1], s=12, color=color(l), alpha=0.7,
                   edgecolors='none', label=lbl)
    ax.set_xlabel(f'{proj_name}-1'); ax.set_ylabel(f'{proj_name}-2')

    s1_m, _ = compute_metrics(StandardScaler().fit_transform(latent), stage1_labels)
    s2_m, _ = compute_metrics(StandardScaler().fit_transform(latent), final_labels)
    title = 'Stage 2: final hierarchical clusters'
    if 'silhouette' in s1_m:
        title += f'\nsil(s1)={s1_m["silhouette"]:.3f}'
    if 'silhouette' in s2_m:
        title += f'  sil(final)={s2_m["silhouette"]:.3f}'
    ax.set_title(title, fontsize=11)
    ax.legend(markerscale=2, fontsize=8, loc='best', ncol=2); ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'stage2_scatter.png'), dpi=150); plt.close()

    # --- stage 1 prototypes ---
    _plot_prototypes(X_wave, meta, stage1_labels, latent, 'stage1_prototypes.png',
                     'Stage 1 prototype waveforms (medoid per group)', 'G',
                     args.out, plt, color)

    # --- stage 2 prototypes ---
    _plot_prototypes(X_wave, meta, final_labels, latent, 'stage2_prototypes.png',
                     'Stage 2 prototype waveforms (medoid per final cluster)', 'C',
                     args.out, plt, color, group_map=group_map)

    # --- cluster spectra (final labels) ---
    sr = float(np.median([md['sample_rate'] for md in meta]))
    freqs_khz = np.fft.rfftfreq(length, 1.0 / sr) / 1000.0
    spectra = np.abs(np.fft.rfft(X_wave, axis=1))

    fig, ax = plt.subplots(figsize=(8, 5))
    for l in final_valid:
        ms = spectra[final_labels == l].mean(axis=0)
        g, s = group_map.get(l, ('?', '?'))
        ax.plot(freqs_khz, ms, color=color(l), lw=1.3,
                label=f'G{g}.{s} (n={int(np.sum(final_labels == l))})')
    ax.set_xlabel('Frequency (kHz)'); ax.set_ylabel('Mean |FFT|')
    ax.set_title('Mean frequency spectra per final cluster')
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'cluster_spectra.png'), dpi=150); plt.close()

    # --- amplitude vs frequency scatter ---
    _plot_amp_vs_freq(meta, final_labels, group_map, args.out, plt, color)

    # --- timeline ---
    _plot_timeline(meta, final_labels, group_map, args.out, plt, color)

    # --- entropy distribution ---
    _plot_entropy(meta, final_labels, final_valid, group_map, args.out, plt, color)

    # --- labels CSV ---
    with open(os.path.join(args.out, 'cluster_labels.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['wfm_index', 'channel', 'time_s', 'sample_rate',
                     'stage1_group', 'final_cluster', 'group_sub',
                     'entropy_nats']
                    + [f'z{i}' for i in range(latent.shape[1])])
        for i, md in enumerate(meta):
            s1 = int(stage1_labels[i])
            fl = int(final_labels[i])
            gs = f"G{group_map[fl][0]}.{group_map[fl][1]}" if fl in group_map else "noise"
            ent = md.get('feat', {}).get('entropy_nats', '')
            if isinstance(ent, float):
                ent = round(ent, 6)
            w.writerow([md['index'], md['channel'], md['time'], md['sample_rate'],
                        s1, fl, gs, ent]
                       + [round(float(z), 6) for z in latent[i]])

    np.save(os.path.join(args.out, 'latent_codes.npy'), latent)

    # --- feature table per final cluster ---
    char_lines = _feature_table(meta, final_labels, final_valid, group_map)

    # --- summary ---
    lines = [
        "AE Two-Stage Hierarchical Clustering — summary",
        "=" * 50,
        f"input            : {args.input}",
        f"denoise          : {args.denoise}",
        f"model            : {args.model}   latent_dim={latent.shape[1]}   "
        f"epochs={len(loss_curve)}/{args.epochs}",
        f"waveforms        : {X_wave.shape[0]}   length={length}",
        f"final loss       : {loss_curve[-1]:.6f}",
        "",
        "Stage 1 (time-domain shape):",
        f"  algorithm      : {args.stage1_algorithm}   k={args.stage1_clusters}",
        f"  coarse groups  : {len(set(stage1_labels) - {-1})}",
    ]
    for l in sorted(set(stage1_labels)):
        if l == -1:
            continue
        cnt = int(np.sum(stage1_labels == l))
        lines.append(f"    G{l}: {cnt} ({100 * cnt / len(stage1_labels):.1f}%)")

    lines += [
        "",
        "Stage 2 (within-group refinement):",
        f"  algorithm      : {args.stage2_algorithm}   k={args.stage2_clusters}",
        f"  features       : {args.stage2_features}",
        f"  final clusters : {len(final_valid)}",
        f"  noise points   : {int(np.sum(final_labels == -1))}",
    ]
    for l in final_valid:
        cnt = int(np.sum(final_labels == l))
        g, s = group_map.get(l, ('?', '?'))
        lines.append(f"    G{g}.{s}: {cnt} ({100 * cnt / len(final_labels):.1f}%)")

    lines += ["", "Quality metrics (latent space):"]
    if s1_m:
        lines.append(f"  stage1 silhouette     : {s1_m['silhouette']:.4f}")
        lines.append(f"  stage1 calinski_harabasz: {s1_m['calinski_harabasz']:.1f}")
        lines.append(f"  stage1 davies_bouldin : {s1_m['davies_bouldin']:.4f}")
    if s2_m:
        lines.append(f"  final  silhouette     : {s2_m['silhouette']:.4f}")
        lines.append(f"  final  calinski_harabasz: {s2_m['calinski_harabasz']:.1f}")
        lines.append(f"  final  davies_bouldin : {s2_m['davies_bouldin']:.4f}")
    lines += char_lines

    summary = "\n".join(lines)
    with open(os.path.join(args.out, 'summary.txt'), 'w') as f:
        f.write(summary + "\n")

    print("[6/6] Done.\n")
    print(summary)


def _plot_prototypes(X_wave, meta, labels, latent, fname, title, prefix, out_dir,
                     plt, color, group_map=None):
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(latent)
    valid = sorted(l for l in set(labels) if l >= 0)
    protos = []
    for l in valid:
        members = np.where(labels == l)[0]
        centroid = Z[members].mean(axis=0)
        medoid = members[int(np.argmin(np.linalg.norm(Z[members] - centroid, axis=1)))]
        protos.append((l, medoid))

    if not protos:
        return
    ncol = min(3, len(protos))
    nrow = int(np.ceil(len(protos) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 2.5 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, (l, medoid) in zip(axes.flat, protos):
        ax.axis('on')
        ax.plot(X_wave[medoid], color=color(l), lw=0.8)
        cnt = int(np.sum(labels == l))
        if group_map and l in group_map:
            g, s = group_map[l]
            lbl = f'G{g}.{s}'
        else:
            lbl = f'{prefix}{l}'
        ax.set_title(f'{lbl}  n={cnt} ({100 * cnt / len(labels):.1f}%)  '
                     f'CH{meta[medoid]["channel"]} #{meta[medoid]["index"]}', fontsize=9)
        ax.set_xlabel('sample'); ax.set_ylabel('norm. V'); ax.tick_params(labelsize=7)
    plt.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight'); plt.close()


def _plot_amp_vs_freq(meta, labels, group_map, out_dir, plt, color):
    freqs, amps, clus = [], [], []
    for md, label in zip(meta, labels):
        feat = md.get('feat', {})
        if 'amplitude_dB' not in feat:
            continue
        freq = next((feat[k] for k in ('peak_freq_kHz', 'centroid_freq_kHz', 'avg_freq_kHz')
                     if k in feat), None)
        if freq is None:
            continue
        freqs.append(float(freq)); amps.append(float(feat['amplitude_dB'])); clus.append(label)

    if not freqs:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    freqs, amps, clus = np.array(freqs), np.array(amps), np.array(clus)
    for l in sorted(set(clus)):
        m = clus == l
        if l == -1:
            lbl = 'noise'
        else:
            g, s = group_map.get(l, ('?', '?'))
            lbl = f'G{g}.{s} (n={int(np.sum(m))})'
        ax.scatter(freqs[m], amps[m], s=18, alpha=0.7, color=color(l), label=lbl)
    ax.set_xlabel('Frequency (kHz)'); ax.set_ylabel('Amplitude (dB)')
    ax.set_title('Amplitude vs frequency by final cluster')
    ax.legend(markerscale=1.5, fontsize=8, ncol=2); ax.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cluster_amplitude_vs_freq.png'), dpi=150); plt.close()


def _plot_timeline(meta, labels, group_map, out_dir, plt, color):
    times = np.array([md['time'] for md in meta], dtype=float)
    amps = np.array([md.get('feat', {}).get('amplitude_dB', np.nan) for md in meta], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    for l in sorted(set(labels)):
        m = (labels == l) & ~np.isnan(amps)
        if np.sum(m) == 0:
            continue
        if l == -1:
            lbl = 'noise'
        else:
            g, s = group_map.get(l, ('?', '?'))
            lbl = f'G{g}.{s}'
        ax.scatter(times[m], amps[m], s=24, alpha=0.7, color=color(l),
                   edgecolors='none', label=lbl)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude (dB)')
    ax.set_title('AE hit amplitude vs time by final cluster')
    ax.legend(markerscale=1.0, fontsize=8, ncol=2); ax.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cluster_timeline.png'), dpi=150); plt.close()


def _plot_entropy(meta, labels, valid, group_map, out_dir, plt, color):
    entropies = np.array([md.get('feat', {}).get('entropy_nats', np.nan) for md in meta])
    if np.all(np.isnan(entropies)):
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    box_data = [entropies[labels == l] for l in valid]
    box_data = [d[~np.isnan(d)] for d in box_data]
    bp = ax.boxplot(box_data,
                    labels=[f'G{group_map[l][0]}.{group_map[l][1]}' if l in group_map else str(l)
                            for l in valid],
                    patch_artist=True)
    for patch, l in zip(bp['boxes'], valid):
        patch.set_facecolor(color(l)); patch.set_alpha(0.6)
    ax.set_ylabel('Entropy (nats)')
    ax.set_title('Waveform entropy per final cluster')
    ax.grid(alpha=0.25)

    ax = axes[1]
    amps_e = np.array([md.get('feat', {}).get('amplitude_dB', np.nan) for md in meta])
    has_both = ~np.isnan(entropies) & ~np.isnan(amps_e)
    if np.sum(has_both) > 0:
        for l in sorted(set(labels)):
            m = (labels == l) & has_both
            if np.sum(m) > 0:
                lbl = 'noise' if l == -1 else f'G{group_map[l][0]}.{group_map[l][1]}'
                ax.scatter(entropies[m], amps_e[m], s=18, alpha=0.7, color=color(l), label=lbl)
        ax.set_xlabel('Entropy (nats)'); ax.set_ylabel('Amplitude (dB)')
        ax.set_title('Entropy vs amplitude by final cluster')
        ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.25)
    else:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cluster_entropy.png'), dpi=150); plt.close()


def _feature_table(meta, labels, valid, group_map):
    keys = [lbl for _, lbl in FEATURE_FIELDS
            if any(lbl in md.get('feat', {}) for md in meta)]
    if not keys:
        return []

    rows = []
    for l in valid:
        members = [i for i in range(len(meta)) if labels[i] == l]
        g, s = group_map.get(l, ('?', '?'))
        row = {'label': f'G{g}.{s}', 'count': len(members)}
        for k in keys:
            vals = np.array([meta[i]['feat'][k] for i in members if k in meta[i]['feat']],
                            dtype=float)
            row[k] = (float(np.mean(vals)), float(np.std(vals))) if len(vals) else (np.nan, np.nan)
        rows.append(row)

    hi = [k for k in ('peak_freq_kHz', 'centroid_freq_kHz', 'avg_freq_kHz',
                       'amplitude_dB', 'energy', 'duration_us', 'rise_us') if k in keys]
    out = ["", "Physical interpretation (mean per final cluster):",
           "  cluster  " + "  ".join(f"{k:>16}" for k in hi)]
    for r in rows:
        out.append(f"  {r['label']:<8} "
                   + "  ".join(f"{r[k][0]:>16.2f}" for k in hi))
    out.append("  -> low centroid/peak freq = delamination/debonding;")
    out.append("     high freq = matrix cracking / fiber breakage.")
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Two-stage hierarchical clustering of Mistras AE waveforms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('input', nargs='?', default=None, help='path to a .DTA file (optional with --load-cache)')
    ap.add_argument('--out', default='ae_hierarchical_out', help='output directory')

    g = ap.add_argument_group('preprocessing / denoising')
    g.add_argument('--denoise', choices=['none', 'wavelet', 'bandpass', 'wavelet+bandpass'],
                   default='none')
    g.add_argument('--denoise-wavelet', default='db4', dest='denoise_wavelet')
    g.add_argument('--denoise-level', type=int, default=0, dest='denoise_level')
    g.add_argument('--denoise-mode', choices=['soft', 'hard'], default='soft', dest='denoise_mode')
    g.add_argument('--denoise-band', type=float, nargs=2, default=[20.0, 400.0],
                   metavar=('LOW_kHz', 'HIGH_kHz'), dest='denoise_band')
    g.add_argument('--denoise-order', type=int, default=4, dest='denoise_order')

    g = ap.add_argument_group('autoencoder (stage 1 representation)')
    g.add_argument('--model', choices=['cae', 'vae'], default='cae')
    g.add_argument('--latent-dim', type=int, default=16, dest='latent_dim')
    g.add_argument('--epochs', type=int, default=100)
    g.add_argument('--early-stop', type=int, default=20, dest='early_stop')
    g.add_argument('--batch-size', type=int, default=64, dest='batch_size')
    g.add_argument('--lr', type=float, default=5e-4)
    g.add_argument('--beta', type=float, default=1.0, help='VAE KL weight')
    g.add_argument('--length', type=int, default=0, dest='fixed_length',
                   help='waveform length; 0=auto-detect')
    g.add_argument('--max-waveforms', type=int, default=5000, dest='max_waveforms')
    g.add_argument('--keep-pretrigger', action='store_true', dest='keep_pretrigger')
    g.add_argument('--channel', type=int, default=None)
    g.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')

    g = ap.add_argument_group('stage 1 — coarse shape clustering')
    g.add_argument('--stage1-algorithm', choices=['kmeans', 'gmm', 'hdbscan'],
                   default='kmeans', dest='stage1_algorithm')
    g.add_argument('--stage1-clusters', type=int, default=3, dest='stage1_clusters',
                   help='number of coarse shape groups (kmeans/gmm)')
    g.add_argument('--stage1-min-cluster-size', type=int, default=30, dest='stage1_min_cluster_size')
    g.add_argument('--stage1-min-samples', type=int, default=5, dest='stage1_min_samples')
    g.add_argument('--scan-k', type=int, default=0, dest='scan_k',
                   help='sweep KMeans k=2..N on latent before stage1 clustering')

    g = ap.add_argument_group('stage 2 — within-group refinement')
    g.add_argument('--stage2-algorithm', choices=['kmeans', 'gmm', 'hdbscan'],
                   default='kmeans', dest='stage2_algorithm')
    g.add_argument('--stage2-clusters', type=int, default=2, dest='stage2_clusters',
                   help='sub-clusters per group (kmeans/gmm)')
    g.add_argument('--stage2-features', choices=['fft', 'param', 'fft+param'],
                   default='fft+param', dest='stage2_features',
                   help='features for within-group refinement')
    g.add_argument('--stage2-min-cluster-size', type=int, default=15, dest='stage2_min_cluster_size')
    g.add_argument('--stage2-min-samples', type=int, default=3, dest='stage2_min_samples')

    g = ap.add_argument_group('embedding / projection')
    g.add_argument('--projection', choices=['umap', 'tsne', 'pca'], default='pca')
    g.add_argument('--umap-neighbors', type=int, default=15, dest='umap_neighbors')
    g.add_argument('--umap-mindist', type=float, default=0.1, dest='umap_mindist')

    g = ap.add_argument_group('preprocessing cache')
    g.add_argument('--save-cache', default=None, dest='save_cache', metavar='PATH',
                   help='save preprocessed data (after denoise) to .npz for reuse')
    g.add_argument('--load-cache', default=None, dest='load_cache', metavar='PATH',
                   help='load preprocessed data from .npz, skip DTA reading/denoise')

    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit("PyTorch required.  pip install torch")

    if args.load_cache:
        import json
        data = np.load(args.load_cache)
        X_wave, length = data['X_wave'], int(data['L'])
        meta_path = args.load_cache.replace('.npz', '_meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        print(f"[1/6] Loaded from cache: {args.load_cache}")
        print(f"      {X_wave.shape[0]} waveforms, length={length}")
    else:
        if not args.input:
            raise SystemExit("Error: .DTA file path required (or use --load-cache)")
        denoiser = make_denoiser(args)
        if denoiser is not None:
            print(f"[0/6] Denoising: {args.denoise}")

        X_wave, meta, length = load_waveforms(
            args.input, args.channel, args.max_waveforms,
            args.fixed_length, args.keep_pretrigger, denoiser)

        if args.save_cache:
            import json
            np.savez_compressed(args.save_cache, X_wave=X_wave, L=np.array(length))
            meta_path = args.save_cache.replace('.npz', '_meta.json')
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
            print(f"      cache saved: {args.save_cache}")

    latent, loss_curve = train_autoencoder(args, X_wave, length)

    if args.scan_k >= 2:
        scan_k(latent, args.scan_k)

    Z_s1, stage1_labels = stage1_cluster(args, latent)
    final_labels, group_map = stage2_refine(args, X_wave, meta, length, stage1_labels)

    emb2d, proj_name = embed_2d(args, latent)

    save_outputs(args, X_wave, meta, length, latent, loss_curve,
                 stage1_labels, final_labels, group_map, emb2d, proj_name)


if __name__ == '__main__':
    main()
