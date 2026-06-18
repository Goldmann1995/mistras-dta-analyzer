#!/usr/bin/env python3
"""STCD-AE — Spectral-Temporal Contrastive multi-view clustering for AE waveforms.

A *standalone* research script (no frontend/backend). Its novelty over the
ordinary "autoencoder -> latent -> k-means" pipeline (tools/ae_deep_cluster.py)
is that it learns ONE shared latent space in which *several different
representations of the same acoustic-emission event are aligned*:

    each AE hit  ->  3 physical "views"
        time     : raw normalized waveform           (1D)
        freq     : log-magnitude FFT spectrum        (1D)
        tf       : time-frequency spectrogram        (2D)

    one encoder per view  ->  z_time, z_freq, z_tf   (same latent dim)

    cross-view contrastive loss (InfoNCE / CLIP-style) pulls the THREE views
    of the SAME event to the SAME point, and pushes different events apart:

        maximize  sim(z_time[i], z_freq[i])   (same event, different view)
        minimize  sim(z_time[i], z_freq[j!=i])

    => the views "agree" in latent space. Clustering then runs on the aligned
       consensus  z = normalize(mean(z_time, z_freq, z_tf)).

Why this is a real innovation for AE (not just SimCLR re-skinned):
  * In SimCLR/MoCo the positive pair is two RANDOM AUGMENTATIONS of one image
    (redundant information). Here the positive pair is two DETERMINISTIC,
    information-COMPLEMENTARY physical transforms of one signal. Alignment
    therefore forces the latent to keep only what is *invariant across domains*
    — the damage-source signature — and discard view-specific nuisances
    (windowing, amplitude/decay from propagation distance). That is exactly the
    quantity that separates CFRP damage modes.
  * Optional physics-disentanglement (--disentangle): split the fused latent
    into z_source / z_propagation and use the parametric AE features
    (peak/centroid frequency vs amplitude/energy/duration) as weak supervision
    + a gradient-reversal adversary so the clustering subspace z_source is
    distance-invariant.

Outputs (into --out):
    loss_curves.png      reconstruction / contrastive / total loss per epoch
    alignment_curve.png  cross-view agreement (same-event vs diff-event cos)
    alignment_2d.png     the 3 views of each event drawn as a triangle that
                         collapses to a point  ==  visual proof of alignment
    latent_scatter.png   2D embedding of the aligned consensus, by cluster
    per_view_scatter.png each view embedded separately, colored by cluster
    cluster_spectra.png  mean FFT per cluster  (physical signature)
    prototypes.png       medoid waveform per cluster
    cluster_labels.csv   wfm index, channel, time, cluster, latent dims
    cluster_features.csv mean +/- std parametric features per cluster
    latent_codes.npy     aligned consensus latent (N x d)
    summary.txt          config + cluster sizes + honest quality metrics

Examples:
    python tools/stcd_ae.py data.DTA --epochs 120 --clusters 4 --projection umap
    python tools/stcd_ae.py data.DTA --views time,freq --algorithm hdbscan
    python tools/stcd_ae.py data.DTA --disentangle --source-dim 16
    python tools/stcd_ae.py --self-test          # synthetic sanity check, no .DTA
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


# Parametric AE hit features (Mistras field -> readable label). Used both for
# physical interpretation of clusters and (with --disentangle) as weak
# supervision. "source" fields are dominated by the damage mechanism; "prop"
# fields are dominated by propagation distance / path.
FEATURE_FIELDS = [
    ('AMP', 'amplitude_dB'), ('ENER', 'energy'), ('ABS-ENERGY', 'abs_energy'),
    ('RISE', 'rise_us'), ('DURATION', 'duration_us'), ('COUN', 'counts'),
    ('A-FRQ', 'avg_freq_kHz'), ('P-FRQ', 'peak_freq_kHz'),
    ('FRQ-C', 'centroid_freq_kHz'), ('R-FRQ', 'rev_freq_kHz'),
    ('I-FRQ', 'init_freq_kHz'),
]
SOURCE_FEATS = ['peak_freq_kHz', 'centroid_freq_kHz', 'avg_freq_kHz', 'init_freq_kHz']
PROP_FEATS = ['amplitude_dB', 'energy', 'abs_energy', 'duration_us', 'rise_us']


def _extract_feats(rec_row):
    if rec_row is None:
        return {}
    names = rec_row.dtype.names
    return {label: float(rec_row[f]) for f, label in FEATURE_FIELDS if f in names}


def round_up_multiple(n, m):
    return int(np.ceil(n / m) * m)


# --------------------------------------------------------------------------- #
# Views: build time / freq / time-frequency representations from a waveform
# --------------------------------------------------------------------------- #
def _resize2d(img, out_hw):
    """Bilinear resize a 2D array to (H, W) using only numpy (interp on axes)."""
    H, W = out_hw
    h, w = img.shape
    yi = np.linspace(0, h - 1, H)
    xi = np.linspace(0, w - 1, W)
    # interp along width then height
    tmp = np.empty((h, W), dtype=np.float32)
    xs = np.arange(w)
    for r in range(h):
        tmp[r] = np.interp(xi, xs, img[r])
    out = np.empty((H, W), dtype=np.float32)
    ys = np.arange(h)
    for c in range(W):
        out[:, c] = np.interp(yi, ys, tmp[:, c])
    return out


def _spectrogram(v, sr, out_hw):
    """Log-magnitude STFT spectrogram of waveform v, resized to out_hw, [0,1]."""
    from scipy.signal import stft
    nper = int(min(64, max(16, len(v) // 4)))
    nover = nper * 3 // 4
    _, _, Z = stft(v, fs=sr, nperseg=nper, noverlap=nover, boundary=None, padded=False)
    S = np.log1p(np.abs(Z)).astype(np.float32)
    if S.shape[1] < 2:  # too few time frames -> tile
        S = np.repeat(S, 2, axis=1)
    S = _resize2d(S, out_hw)
    smax = float(S.max())
    if smax > 0:
        S = S / smax
    return S


def _spectrum(v, L):
    """Log-magnitude FFT spectrum resampled to length L, peak-normalized."""
    spec = np.log1p(np.abs(np.fft.rfft(v))).astype(np.float32)
    xp = np.linspace(0, 1, len(spec))
    xq = np.linspace(0, 1, L)
    spec = np.interp(xq, xp, spec).astype(np.float32)
    smax = float(np.max(np.abs(spec)))
    if smax > 0:
        spec = spec / smax
    return spec


def build_views(waves, srates, views, L, tf_hw):
    """waves: list/array of peak-normalized waveforms (each length L).
    Returns dict view-name -> ndarray, plus X_wave (N, L)."""
    N = len(waves)
    out = {}
    if 'time' in views:
        out['time'] = np.stack(waves)[:, None, :].astype(np.float32)        # (N,1,L)
    if 'freq' in views:
        out['freq'] = np.stack([_spectrum(v, L) for v in waves])[:, None, :].astype(np.float32)
    if 'tf' in views:
        tf = np.stack([_spectrogram(waves[i], srates[i], tf_hw) for i in range(N)])
        out['tf'] = tf[:, None, :, :].astype(np.float32)                    # (N,1,H,W)
    return out, np.stack(waves).astype(np.float32)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_dta(dta_path, channel, max_waveforms, fixed_length, keep_pretrigger):
    """Return (waves, srates, meta, L) of peak-normalized fixed-length waveforms."""
    from MistrasDTA import read_bin, get_waveform_data
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

    waves, srates, meta = [], [], []
    for i in idx_all:
        row = wfm[i]
        t, V = get_waveform_data(row)
        if not keep_pretrigger and row['TDLY'] < 0:
            V = V[abs(int(row['TDLY'])):]
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

        waves.append(v)
        srates.append(float(row['SRATE']))
        meta.append({
            'index': int(i), 'channel': int(row['CH']),
            'time': float(row['SSSSSSSS.mmmuuun']), 'sample_rate': float(row['SRATE']),
            'feat': _extract_feats(rec_row),
        })

    if len(waves) < 8:
        raise SystemExit("Not enough valid waveforms (need >= 8).")
    return waves, srates, meta, L


def synth_dataset(n=480, L=384, sr=1e6, k=4, seed=0):
    """Synthetic AE-like data for --self-test: K damage modes that differ by
    FREQUENCY (the signal) but share random amplitude/decay (the nuisance that
    the alignment is supposed to ignore). Returns waves, srates, meta, L, truth."""
    rng = np.random.default_rng(seed)
    base_freqs = np.linspace(0.08, 0.42, k) * sr   # Hz, well separated
    t = np.arange(L) / sr
    waves, srates, meta, truth = [], [], [], []
    for i in range(n):
        c = i % k
        f0 = base_freqs[c] * (1 + 0.04 * rng.standard_normal())
        decay = rng.uniform(L * 0.1, L * 0.5)            # nuisance
        amp = rng.uniform(0.3, 1.0)                       # nuisance
        env = np.exp(-np.arange(L) / decay)
        sig = amp * env * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 6.28))
        sig = sig + 0.05 * rng.standard_normal(L)
        peak = np.max(np.abs(sig))
        if peak > 0:
            sig = sig / peak
        waves.append(sig.astype(np.float32))
        srates.append(sr)
        meta.append({'index': i, 'channel': 1, 'time': i * 1e-3, 'sample_rate': sr,
                     'feat': {'peak_freq_kHz': f0 / 1000.0,
                              'centroid_freq_kHz': f0 / 1000.0,
                              'amplitude_dB': 20 * np.log10(amp + 1e-6) + 60,
                              'energy': amp ** 2 * 100,
                              'duration_us': decay}})
        truth.append(c)
    return waves, srates, meta, L, np.array(truth)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_modules(torch, nn, L, tf_hw, d):
    class Enc1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 16, 7, 2, 3), nn.BatchNorm1d(16), nn.ReLU(),
                nn.Conv1d(16, 32, 7, 2, 3), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 64, 7, 2, 3), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 64, 7, 2, 3), nn.BatchNorm1d(64), nn.ReLU(),
            )
            self.flat = (L // 16) * 64
            self.fc = nn.Linear(self.flat, d)

        def forward(self, x):
            return self.fc(self.net(x).flatten(1))

    class Dec1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(d, (L // 16) * 64)
            self.net = nn.Sequential(
                nn.ConvTranspose1d(64, 64, 7, 2, 3, output_padding=1), nn.BatchNorm1d(64), nn.ReLU(),
                nn.ConvTranspose1d(64, 32, 7, 2, 3, output_padding=1), nn.BatchNorm1d(32), nn.ReLU(),
                nn.ConvTranspose1d(32, 16, 7, 2, 3, output_padding=1), nn.BatchNorm1d(16), nn.ReLU(),
                nn.ConvTranspose1d(16, 1, 7, 2, 3, output_padding=1),
            )

        def forward(self, z):
            h = self.fc(z).view(z.size(0), 64, L // 16)
            return self.net(h)[:, :, :L]

    class Enc2D(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Conv2d(16, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            )
            self.h8 = (tf_hw[0] // 8, tf_hw[1] // 8)
            self.flat = 64 * self.h8[0] * self.h8[1]
            self.fc = nn.Linear(self.flat, d)

        def forward(self, x):
            return self.fc(self.net(x).flatten(1))

    class Dec2D(nn.Module):
        def __init__(self):
            super().__init__()
            self.h8 = (tf_hw[0] // 8, tf_hw[1] // 8)
            self.fc = nn.Linear(d, 64 * self.h8[0] * self.h8[1])
            self.net = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.ConvTranspose2d(16, 1, 4, 2, 1),
            )

        def forward(self, z):
            h = self.fc(z).view(z.size(0), 64, self.h8[0], self.h8[1])
            return self.net(h)[:, :, :tf_hw[0], :tf_hw[1]]

    return {'time': (Enc1D, Dec1D), 'freq': (Enc1D, Dec1D), 'tf': (Enc2D, Dec2D)}


class _GradReverse:
    """Gradient reversal (Ganin & Lempitsky 2015) for the disentangle adversary."""
    @staticmethod
    def apply(torch, x, lambd):
        class _F(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.view_as(inp)

            @staticmethod
            def backward(ctx, grad):
                return -lambd * grad
        return _F.apply(x)


def info_nce(torch, F, za, zb, tau):
    """Symmetric CLIP-style cross-view contrastive loss on L2-normalized z.

    Aligns views, but is an INSTANCE-DISCRIMINATION objective: every event is
    its own class, so it also repels events that belong to the same damage mode
    -> it ALIGNS well yet ANTI-CLUSTERS. Kept as an ablation (--align-loss
    infonce); the default vicreg avoids this."""
    logits = za @ zb.t() / tau
    labels = torch.arange(za.size(0), device=za.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def vicreg(torch, F, za, zb, sim_w=25.0, std_w=25.0, cov_w=1.0):
    """Non-contrastive cross-view alignment (Bardes et al. 2022, VICReg).

    invariance : pull the two views of the SAME event together (MSE)
    variance   : keep each latent dim's batch std >= 1  (prevents collapse)
    covariance : decorrelate latent dims                (spreads information)

    Crucially there is NO term that repels different events, so events of the
    same damage mode are free to stay together -> the latent CLUSTERS while the
    views ALIGN. This is the key to good downstream clustering."""
    B, D = za.shape
    sim = F.mse_loss(za, zb)
    std_a = torch.sqrt(za.var(dim=0) + 1e-4)
    std_b = torch.sqrt(zb.var(dim=0) + 1e-4)
    std = 0.5 * (F.relu(1.0 - std_a).mean() + F.relu(1.0 - std_b).mean())
    za_c = za - za.mean(0)
    zb_c = zb - zb.mean(0)
    cov_a = (za_c.t() @ za_c) / (B - 1)
    cov_b = (zb_c.t() @ zb_c) / (B - 1)
    off = ~torch.eye(D, dtype=torch.bool, device=za.device)
    cov = (cov_a[off].pow(2).sum() + cov_b[off].pow(2).sum()) / D
    return sim_w * sim + std_w * std + cov_w * cov


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args, view_arrays, meta, L, tf_hw, view_names):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(args.device if args.device != 'auto'
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    d = args.latent_dim
    print(f"[3/6] Training STCD-AE on {device}  views={view_names}  "
          f"latent={d}  epochs={args.epochs}  lr={args.lr}")

    factory = build_modules(torch, nn, L, tf_hw, d)
    encoders = nn.ModuleDict({v: factory[v][0]() for v in view_names}).to(device)
    decoders = nn.ModuleDict({v: factory[v][1]() for v in view_names}).to(device)

    params = list(encoders.parameters()) + list(decoders.parameters())

    # optional physics-disentanglement heads
    disent = args.disentangle
    if disent:
        ds = min(args.source_dim, d - 1)
        # predict source-frequency feats from z_source; prop feats from z_prop;
        # adversary tries to predict prop feats from z_source (reversed grad).
        src_keys = [k for k in SOURCE_FEATS if any(k in m['feat'] for m in meta)]
        prop_keys = [k for k in PROP_FEATS if any(k in m['feat'] for m in meta)]
        if not src_keys or not prop_keys:
            print("      [disentangle] missing parametric features; disabling.")
            disent = False
    if disent:
        head_src = nn.Sequential(nn.Linear(ds, 32), nn.ReLU(), nn.Linear(32, len(src_keys))).to(device)
        head_prop = nn.Sequential(nn.Linear(d - ds, 32), nn.ReLU(), nn.Linear(32, len(prop_keys))).to(device)
        head_adv = nn.Sequential(nn.Linear(ds, 32), nn.ReLU(), nn.Linear(32, len(prop_keys))).to(device)
        params += list(head_src.parameters()) + list(head_prop.parameters()) + list(head_adv.parameters())

        def _targets(keys):
            M = np.array([[m['feat'].get(k, np.nan) for k in keys] for m in meta], dtype=np.float32)
            mu = np.nanmean(M, axis=0); sd = np.nanstd(M, axis=0) + 1e-6
            M = (M - mu) / sd
            return torch.from_numpy(np.nan_to_num(M)).to(device)
        Y_src, Y_prop = _targets(src_keys), _targets(prop_keys)
        print(f"      [disentangle] z_source dim={ds} predicts {src_keys}; "
              f"z_prop dim={d-ds} predicts {prop_keys}; adversary on z_source")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    Xt = {v: torch.from_numpy(view_arrays[v]).to(device) for v in view_names}
    N = next(iter(Xt.values())).shape[0]
    bs = min(args.batch_size, N)
    nb = max(1, N // bs)
    pairs = [(view_names[i], view_names[j])
             for i in range(len(view_names)) for j in range(i + 1, len(view_names))]

    # fixed eval subset for the alignment metric
    eval_n = min(512, N)
    eval_idx = torch.arange(eval_n, device=device)

    hist = {'total': [], 'recon': [], 'contrast': [], 'cos_same': [], 'cos_diff': [], 'sil': []}
    n_params = sum(p.numel() for p in params)
    print(f"      trainable parameters: {n_params:,}  ({len(pairs)} view-pair alignment term(s))")

    from sklearn.cluster import KMeans as _KM
    from sklearn.preprocessing import StandardScaler as _SS
    from sklearn.metrics import silhouette_score as _sil

    def extract():
        """Full-set embeddings: raw per-view H, normalized Zn, fused consensus."""
        encoders.eval()
        with torch.no_grad():
            H = {v: encoders[v](Xt[v]).cpu().numpy() for v in view_names}
        Zn = {v: H[v] / (np.linalg.norm(H[v], axis=1, keepdims=True) + 1e-9) for v in view_names}
        consensus = np.mean([Zn[v] for v in view_names], axis=0)
        fused_raw = np.mean([H[v] for v in view_names], axis=0)
        return {'H': H, 'per_view': Zn, 'consensus': consensus, 'fused_raw': fused_raw}

    def selection_silhouette(consensus):
        """Unsupervised model-selection score: KMeans silhouette on the fused
        latent. Lets us keep the best epoch and stay immune to overtraining
        (no labels needed)."""
        try:
            Z = _SS().fit_transform(consensus)
            lab = _KM(max(2, args.clusters), n_init=5, random_state=42).fit_predict(Z)
            return float(_sil(Z, lab))
        except Exception:
            return -1.0

    ckpts = []                       # (epoch, silhouette, align-gap, snapshot)
    ckpt_every = max(1, args.epochs // 15)

    for epoch in range(args.epochs):
        encoders.train(); decoders.train()
        perm = torch.randperm(N, device=device)
        e_tot = e_rec = e_con = 0.0
        for b in range(nb):
            bidx = perm[b * bs:(b + 1) * bs]
            if bidx.numel() < 2:
                continue
            opt.zero_grad()

            h = {v: encoders[v](Xt[v][bidx]) for v in view_names}        # raw latents
            z = {v: F.normalize(h[v], dim=1) for v in view_names}        # normalized

            recon = sum(F.mse_loss(decoders[v](h[v]), Xt[v][bidx]) for v in view_names) / len(view_names)
            if not pairs:
                align = torch.zeros((), device=device)
            elif args.align_loss == 'infonce':
                align = sum(info_nce(torch, F, z[a], z[b2], args.tau) for a, b2 in pairs) / len(pairs)
            else:  # vicreg (default): aligns without repelling same-mode events
                align = sum(vicreg(torch, F, h[a], h[b2]) for a, b2 in pairs) / len(pairs)

            loss = args.recon_weight * recon + args.contrast_weight * align

            if disent:
                fused = torch.stack([h[v] for v in view_names], 0).mean(0)   # (B, d)
                z_src, z_prop = fused[:, :ds], fused[:, ds:]
                l_src = F.mse_loss(head_src(z_src), Y_src[bidx])
                l_prop = F.mse_loss(head_prop(z_prop), Y_prop[bidx])
                adv_in = _GradReverse.apply(torch, z_src, args.adv_lambda)
                l_adv = F.mse_loss(head_adv(adv_in), Y_prop[bidx])
                loss = loss + args.disent_weight * (l_src + l_prop + l_adv)

            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            e_tot += float(loss); e_rec += float(recon); e_con += float(align)
        sched.step()

        # ---- alignment diagnostic on the fixed eval subset ----
        encoders.eval()
        with torch.no_grad():
            ze = {v: F.normalize(encoders[v](Xt[v][eval_idx]), dim=1) for v in view_names}
            if pairs:
                cs = np.mean([float((ze[a] * ze[b2]).sum(1).mean()) for a, b2 in pairs])
                cd = np.mean([float((ze[a] @ ze[b2].t()).mean()) for a, b2 in pairs])  # ~ off-diagonal avg
            else:
                cs = cd = float('nan')
        hist['total'].append(e_tot / nb); hist['recon'].append(e_rec / nb)
        hist['contrast'].append(e_con / nb); hist['cos_same'].append(cs); hist['cos_diff'].append(cd)

        # ---- unsupervised checkpoint: snapshot for later model selection ----
        if (epoch + 1) % ckpt_every == 0 or epoch == args.epochs - 1:
            snap = extract()
            sil = selection_silhouette(snap['consensus'])
            hist['sil'].append((epoch + 1, sil))
            ckpts.append({'epoch': epoch + 1, 'sil': sil, 'gap': cs - cd, 'snap': snap})

        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            cur = max((c['sil'] for c in ckpts), default=-1.0)
            print(f"      epoch {epoch+1:3d}/{args.epochs}  total={e_tot/nb:.4f}  "
                  f"recon={e_rec/nb:.4f}  align={e_con/nb:.4f}  "
                  f"cross-view(same-diff)={cs-cd:+.3f}  best_sil={cur:.3f}")

    # ---- model selection: pick the checkpoint best for BOTH clustering AND
    # alignment (normalized silhouette + normalized cross-view gap). This finds
    # the "knee": clusters are well separated yet the views are well aligned,
    # and it is immune to over/under-training. ----
    def _norm(a):
        a = np.asarray(a, float); rng = a.max() - a.min()
        return (a - a.min()) / rng if rng > 1e-9 else np.ones_like(a)
    sils = [c['sil'] for c in ckpts]
    gaps = [c['gap'] for c in ckpts]
    score = _norm(sils) + _norm(gaps)
    chosen = ckpts[int(np.argmax(score))]
    snap = chosen['snap']
    H, Zn = snap['H'], snap['per_view']
    consensus, fused_raw = snap['consensus'], snap['fused_raw']

    cluster_latent, latent_kind = consensus, 'aligned-fused'
    if disent:
        cluster_latent, latent_kind = fused_raw[:, :ds], 'z_source'

    print(f"      selected epoch {chosen['epoch']} "
          f"(silhouette={chosen['sil']:.3f}, align gap={chosen['gap']:+.3f}) "
          f"by combined clustering+alignment score")
    return {'per_view': Zn, 'H': H, 'consensus': consensus, 'fused_raw': fused_raw,
            'cluster_latent': cluster_latent, 'latent_kind': latent_kind,
            'hist': hist, 'best_epoch': chosen['epoch']}


# --------------------------------------------------------------------------- #
# Embedding + clustering (shared with ae_deep_cluster conventions)
# --------------------------------------------------------------------------- #
def embed_2d(args, latent, fit_extra=None):
    """2D embedding. If fit_extra is given (stacked per-view points), fit the
    reducer jointly so views share ONE coordinate frame (for alignment_2d)."""
    from sklearn.decomposition import PCA
    N = latent.shape[0]
    data = latent if fit_extra is None else np.vstack([latent, fit_extra])
    if latent.shape[1] <= 2:
        p = data[:, :2] if data.shape[1] >= 2 else np.column_stack([data[:, 0], np.zeros(len(data))])
        return p[:N], (p[N:] if fit_extra is not None else None), 'latent'
    if args.projection == 'umap':
        try:
            import umap
            r = umap.UMAP(n_components=2, random_state=42,
                          n_neighbors=args.umap_neighbors, min_dist=args.umap_mindist)
            p = r.fit_transform(data)
            return p[:N], (p[N:] if fit_extra is not None else None), 'UMAP'
        except ImportError:
            print("      [tip] umap-learn not installed; using PCA. pip install umap-learn")
            args.projection = 'pca'
    if args.projection == 'tsne':
        from sklearn.manifold import TSNE
        perp = float(min(30, max(2, len(data) // 4), len(data) - 1))
        p = TSNE(2, random_state=42, perplexity=perp, init='pca').fit_transform(data)
        return p[:N], (p[N:] if fit_extra is not None else None), 't-SNE'
    p = PCA(2, random_state=42).fit_transform(data)
    return p[:N], (p[N:] if fit_extra is not None else None), 'PCA'


def cluster(args, latent, emb2d):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.mixture import GaussianMixture
    print(f"[4/6] Clustering ({args.algorithm}) on the aligned latent ...")
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
    print(f"      ran on {where}: clusters={len(valid)}  noise={int(np.sum(labels==-1))}/{len(labels)}")
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
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    Z = StandardScaler().fit_transform(latent)
    print("      k-scan (KMeans, silhouette on aligned latent):")
    best_k, best_s = None, -1
    for k in range(2, kmax + 1):
        lab = KMeans(k, random_state=42, n_init=10).fit_predict(Z)
        s = silhouette_score(Z, lab)
        flag = ''
        if s > best_s:
            best_s, best_k, flag = s, k, '  <- best'
        print(f"        k={k}: silhouette={s:.4f}{flag}")
    print(f"      suggested clusters: {best_k}\n")


# --------------------------------------------------------------------------- #
# Visualization + outputs
# --------------------------------------------------------------------------- #
def save_outputs(args, res, view_names, X_wave, meta, L, labels, space, emb2d,
                 proj_name, m, valid, truth=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(args.out, exist_ok=True)
    cmap = plt.get_cmap('tab10')

    def color(l):
        return (0.4, 0.4, 0.4, 0.35) if l == -1 else cmap(l % 10)

    print(f"[5/6] Writing figures + tables to {args.out}/ ...")
    hist = res['hist']
    ep = range(1, len(hist['total']) + 1)

    # 1) loss curves
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, hist['total'], label='total', color='#111827', lw=1.6)
    ax.plot(ep, hist['recon'], label='reconstruction', color='#0891b2', lw=1.2)
    ax.plot(ep, hist['contrast'], label='cross-view contrastive', color='#db2777', lw=1.2)
    ax.set_xlabel('epoch'); ax.set_ylabel('loss'); ax.set_title('STCD-AE training losses')
    ax.legend(fontsize=9); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'loss_curves.png'), dpi=150); plt.close()

    # 2) alignment curve — the core "views align in latent space" evidence
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, hist['cos_same'], label='same event, different views', color='#16a34a', lw=1.6)
    ax.plot(ep, hist['cos_diff'], label='different events', color='#9ca3af', lw=1.4)
    gap = np.array(hist['cos_same']) - np.array(hist['cos_diff'])
    ax.fill_between(ep, hist['cos_diff'], hist['cos_same'], color='#16a34a', alpha=0.12)
    be = res.get('best_epoch')
    if be:
        ax.axvline(be, color='#6b7280', ls='--', lw=1, label=f'selected epoch {be}')
    ax.set_xlabel('epoch'); ax.set_ylabel('mean cosine similarity')
    ax.set_title(f'Cross-view latent alignment  (final gap = {gap[-1]:+.3f})')
    ax.legend(fontsize=9, loc='best'); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'alignment_curve.png'), dpi=150); plt.close()

    # 3) alignment_2d — the 3 views of each event as a collapsing triangle
    if len(view_names) >= 2:
        N = X_wave.shape[0]
        stacked = np.vstack([res['per_view'][v] for v in view_names])  # (V*N, d)
        _, views2d, pname2 = embed_2d(args, res['consensus'], fit_extra=stacked)
        V = len(view_names)
        per_view_2d = [views2d[i * N:(i + 1) * N] for i in range(V)]
        markers = ['o', '^', 's', 'D', 'P']
        vpal = ['#2563eb', '#ea580c', '#16a34a', '#9333ea', '#0d9488']
        rng = np.random.default_rng(0)
        show = rng.choice(N, size=min(args.align_lines, N), replace=False)
        be = res.get('best_epoch', len(hist['cos_same']))
        gap = hist['cos_same'][be - 1] - hist['cos_diff'][be - 1]

        fig, ax = plt.subplots(figsize=(8, 7))
        for s in show:                       # gray links join one event's views
            poly = np.array([per_view_2d[k][s] for k in range(V)] + [per_view_2d[0][s]])
            ax.plot(poly[:, 0], poly[:, 1], color='#9ca3af', lw=0.3, alpha=0.45, zorder=1)
        for k, v in enumerate(view_names):    # COLOR BY VIEW -> overlap = alignment
            pts = per_view_2d[k]
            ax.scatter(pts[:, 0], pts[:, 1], s=11, marker=markers[k % len(markers)],
                       color=vpal[k % len(vpal)], alpha=0.55, edgecolors='none',
                       label=f'view: {v}', zorder=2)
        ax.set_title('Cross-view latent alignment\n'
                     'each view in its own color; the views of one event are joined by a gray link '
                     f'(short links = aligned; gap={gap:+.2f})')
        ax.set_xlabel(f'{pname2}-1'); ax.set_ylabel(f'{pname2}-2')
        ax.legend(fontsize=9, markerscale=1.6, loc='best'); ax.grid(alpha=0.2)
        plt.tight_layout(); plt.savefig(os.path.join(args.out, 'alignment_2d.png'), dpi=150); plt.close()

        # 5) per-view scatter small multiples colored by the SAME cluster labels
        fig, axes = plt.subplots(1, V, figsize=(4.6 * V, 4.4), squeeze=False)
        for k, v in enumerate(view_names):
            ax = axes[0, k]; pts = per_view_2d[k]
            for l in sorted(set(labels)):
                mlab = labels == l
                ax.scatter(pts[mlab, 0], pts[mlab, 1], s=8, color=color(l), alpha=0.6, edgecolors='none')
            ax.set_title(f'view: {v}'); ax.set_xticks([]); ax.set_yticks([])
        plt.suptitle('Each view embedded separately, colored by the shared clusters', y=1.02)
        plt.tight_layout(); plt.savefig(os.path.join(args.out, 'per_view_scatter.png'),
                                        dpi=150, bbox_inches='tight'); plt.close()

    # 4) consensus latent scatter colored by cluster
    fig, ax = plt.subplots(figsize=(8, 7))
    for l in sorted(set(labels)):
        pts = emb2d[labels == l]
        ax.scatter(pts[:, 0], pts[:, 1], s=12, color=color(l), alpha=0.75, edgecolors='none',
                   label=('noise' if l == -1 else f'C{l} (n={int(np.sum(labels==l))})'))
    title = f'{proj_name} of {res["latent_kind"]} latent — {args.algorithm}'
    if 'silhouette' in m:
        title += f'   sil={m["silhouette"]:.3f}'
    ax.set_title(title); ax.set_xlabel(f'{proj_name}-1'); ax.set_ylabel(f'{proj_name}-2')
    ax.legend(markerscale=2, fontsize=9, loc='best'); ax.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, 'latent_scatter.png'), dpi=150); plt.close()

    # 6) cluster mean spectra (physical signature)
    sr = float(np.median([md['sample_rate'] for md in meta]))
    freqs_khz = np.fft.rfftfreq(L, 1.0 / sr) / 1000.0
    spectra = np.abs(np.fft.rfft(X_wave, axis=1))
    fig, ax = plt.subplots(figsize=(8, 5))
    for l in valid:
        ax.plot(freqs_khz, spectra[labels == l].mean(0), color=color(l), lw=1.3,
                label=f'C{l} (n={int(np.sum(labels==l))})')
    ax.set_xlabel('Frequency (kHz)'); ax.set_ylabel('Mean |FFT|')
    ax.set_title('Cluster mean frequency spectra — physical signature')
    ax.legend(fontsize=9); ax.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'cluster_spectra.png'), dpi=150); plt.close()

    # 7) medoid prototypes
    protos = []
    for l in valid:
        members = np.where(labels == l)[0]
        cen = space[members].mean(0)
        protos.append((l, members[int(np.argmin(np.linalg.norm(space[members] - cen, axis=1)))]))
    if protos:
        ncol = min(3, len(protos)); nrow = int(np.ceil(len(protos) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 2.5 * nrow), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        for ax, (l, md_i) in zip(axes.flat, protos):
            ax.axis('on'); ax.plot(X_wave[md_i], color=color(l), lw=0.8)
            cnt = int(np.sum(labels == l))
            ax.set_title(f'C{l}  n={cnt} ({100*cnt/len(labels):.1f}%)  '
                         f'CH{meta[md_i]["channel"]} #{meta[md_i]["index"]}', fontsize=9)
            ax.set_xlabel('sample'); ax.set_ylabel('norm. V'); ax.tick_params(labelsize=7)
        plt.suptitle('Cluster prototype (medoid) waveforms', y=1.01)
        plt.tight_layout(); plt.savefig(os.path.join(args.out, 'prototypes.png'),
                                        dpi=150, bbox_inches='tight'); plt.close()

    # tables
    latent = res['cluster_latent']
    with open(os.path.join(args.out, 'cluster_labels.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['wfm_index', 'channel', 'time_s', 'sample_rate', 'cluster']
                   + [f'z{i}' for i in range(latent.shape[1])])
        for i, md in enumerate(meta):
            w.writerow([md['index'], md['channel'], md['time'], md['sample_rate'], int(labels[i])]
                       + [round(float(z), 6) for z in latent[i]])
    np.save(os.path.join(args.out, 'latent_codes.npy'), latent)

    char_lines = _characterize(args, meta, labels, valid)

    # summary
    lines = [
        "STCD-AE — Spectral-Temporal Contrastive multi-view AE clustering",
        "=" * 62,
        f"input            : {args.input or 'SELF-TEST (synthetic)'}",
        f"views            : {', '.join(view_names)}",
        f"latent / cluster : dim={res['consensus'].shape[1]}  clustered on={res['latent_kind']}"
        f"{' (disentangled)' if args.disentangle else ''}",
        f"clustering       : {args.algorithm}  embedding={proj_name}",
        f"waveforms        : {X_wave.shape[0]}  length={L}",
        f"final align gap  : {hist['cos_same'][-1]-hist['cos_diff'][-1]:+.3f}  "
        f"(same-view minus diff-view cosine; higher=better aligned)",
        f"n_clusters       : {len(valid)}   noise={int(np.sum(labels==-1))}",
        "",
        "cluster sizes:",
    ]
    for l in valid:
        cnt = int(np.sum(labels == l))
        lines.append(f"  C{l}: {cnt}  ({100*cnt/len(labels):.1f}%)")
    lines.append("")
    lines.append("quality metrics (report silhouette on the aligned latent):")
    if m:
        lines.append(f"  silhouette        : {m['silhouette']:.4f}  (higher=better, max 1)")
        lines.append(f"  calinski_harabasz : {m['calinski_harabasz']:.1f}  (higher=better)")
        lines.append(f"  davies_bouldin    : {m['davies_bouldin']:.4f}  (lower=better)")
    else:
        lines.append("  (need >= 2 non-noise clusters)")
    if truth is not None:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        lines.append("")
        lines.append("self-test recovery of known synthetic damage modes:")
        lines.append(f"  ARI = {adjusted_rand_score(truth, labels):.3f}   "
                     f"NMI = {normalized_mutual_info_score(truth, labels):.3f}  (1.0 = perfect)")
    lines += char_lines
    summary = "\n".join(lines)
    with open(os.path.join(args.out, 'summary.txt'), 'w') as f:
        f.write(summary + "\n")
    print("[6/6] Done.\n")
    print(summary)


def _characterize(args, meta, labels, valid):
    keys = [lbl for _, lbl in FEATURE_FIELDS if any(lbl in md.get('feat', {}) for md in meta)]
    if not keys:
        return []
    rows = []
    for l in valid:
        members = [i for i in range(len(meta)) if labels[i] == l]
        row = {'cluster': l, 'count': len(members)}
        for k in keys:
            vals = np.array([meta[i]['feat'][k] for i in members if k in meta[i]['feat']], dtype=float)
            row[k] = (float(np.mean(vals)), float(np.std(vals))) if len(vals) else (np.nan, np.nan)
        rows.append(row)
    with open(os.path.join(args.out, 'cluster_features.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cluster', 'count'] + [f'{k}_mean' for k in keys] + [f'{k}_std' for k in keys])
        for r in rows:
            w.writerow([r['cluster'], r['count']]
                       + [round(r[k][0], 3) for k in keys] + [round(r[k][1], 3) for k in keys])
    hi = [k for k in ('peak_freq_kHz', 'centroid_freq_kHz', 'avg_freq_kHz',
                      'amplitude_dB', 'energy', 'duration_us') if k in keys]
    out = ["", "physical interpretation (mean per cluster):",
           "  cluster  " + "  ".join(f"{k:>16}" for k in hi)]
    for r in rows:
        out.append(f"  C{r['cluster']:<6} " + "  ".join(f"{r[k][0]:>16.2f}" for k in hi))
    out.append("  -> low peak/centroid freq ~ delamination/debonding;")
    out.append("     high freq ~ matrix cracking / fiber breakage (verify for your material).")
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="STCD-AE: cross-view contrastive multi-view clustering of AE waveforms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('input', nargs='?', default=None, help='path to a .DTA file')
    ap.add_argument('--out', default='stcd_out', help='output directory')
    ap.add_argument('--self-test', action='store_true', dest='self_test',
                    help='run on synthetic data (no .DTA) to validate the install')

    g = ap.add_argument_group('views + representation')
    g.add_argument('--views', default='time,freq,tf',
                   help='comma list of: time,freq,tf  (>=2 needed to align)')
    g.add_argument('--tf-size', type=int, default=32, dest='tf_size',
                   help='spectrogram side (divisible by 8) for the tf view')
    g.add_argument('--latent-dim', type=int, default=16, dest='latent_dim',
                   help='keep modest: VICReg spreads ALL dims, so excess dims add '
                        'noise and hurt clustering (16 works well; try 8-24)')
    g.add_argument('--length', type=int, default=0, dest='fixed_length',
                   help='waveform length; 0=auto-detect')
    g.add_argument('--max-waveforms', type=int, default=4000, dest='max_waveforms')
    g.add_argument('--keep-pretrigger', action='store_true', dest='keep_pretrigger')
    g.add_argument('--channel', type=int, default=None)

    g = ap.add_argument_group('training')
    g.add_argument('--epochs', type=int, default=120)
    g.add_argument('--batch-size', type=int, default=128, dest='batch_size')
    g.add_argument('--lr', type=float, default=1e-3)
    g.add_argument('--align-loss', choices=['vicreg', 'infonce'], default='vicreg',
                   dest='align_loss',
                   help='cross-view alignment: vicreg (non-contrastive, clusters well) '
                        'or infonce (instance-discrimination, aligns but anti-clusters)')
    g.add_argument('--tau', type=float, default=0.2, help='InfoNCE temperature')
    g.add_argument('--recon-weight', type=float, default=1.0, dest='recon_weight')
    g.add_argument('--contrast-weight', type=float, default=1.0, dest='contrast_weight')
    g.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')

    g = ap.add_argument_group('physics-disentanglement (optional)')
    g.add_argument('--disentangle', action='store_true',
                   help='split latent into z_source/z_prop with weak supervision + adversary')
    g.add_argument('--source-dim', type=int, default=16, dest='source_dim',
                   help='dims of z_source (clustered when --disentangle)')
    g.add_argument('--disent-weight', type=float, default=0.3, dest='disent_weight')
    g.add_argument('--adv-lambda', type=float, default=0.5, dest='adv_lambda')

    g = ap.add_argument_group('embedding + clustering')
    g.add_argument('--projection', choices=['umap', 'tsne', 'pca'], default='umap')
    g.add_argument('--umap-neighbors', type=int, default=15, dest='umap_neighbors')
    g.add_argument('--umap-mindist', type=float, default=0.1, dest='umap_mindist')
    g.add_argument('--algorithm', choices=['kmeans', 'gmm', 'hdbscan', 'dbscan'], default='kmeans')
    g.add_argument('--clusters', type=int, default=4)
    g.add_argument('--density-space', choices=['embed', 'latent'], default='embed', dest='density_space')
    g.add_argument('--eps', type=float, default=0.3)
    g.add_argument('--min-samples', type=int, default=5, dest='min_samples')
    g.add_argument('--min-cluster-size', type=int, default=30, dest='min_cluster_size')
    g.add_argument('--align-lines', type=int, default=150, dest='align_lines',
                   help='how many events to connect across views in alignment_2d.png')
    g.add_argument('--scan-k', type=int, default=0, dest='scan_k')
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit("PyTorch required.  pip install -r tools/requirements.txt")

    view_names = [v.strip() for v in args.views.split(',') if v.strip()]
    for v in view_names:
        if v not in ('time', 'freq', 'tf'):
            raise SystemExit(f"Unknown view '{v}'. Choose from: time, freq, tf")
    if len(view_names) < 2:
        raise SystemExit("Need >= 2 views to align (e.g. --views time,freq).")
    args.tf_size = round_up_multiple(max(8, args.tf_size), 8)  # 2D enc/dec round-trip
    tf_hw = (args.tf_size, args.tf_size)

    truth = None
    if args.self_test or args.input is None:
        if args.input is None and not args.self_test:
            raise SystemExit("Provide a .DTA file, or use --self-test.")
        print("[1/6] Generating synthetic AE dataset (--self-test) ...")
        waves, srates, meta, L, truth = synth_dataset()
        args.input = None
        if args.epochs == 120:      # default -> keep the sanity check fast
            args.epochs = 60
    else:
        waves, srates, meta, L = load_dta(
            args.input, args.channel, args.max_waveforms, args.fixed_length, args.keep_pretrigger)

    print(f"[2/6] Building views {view_names} (L={L}, tf={tf_hw}) ...")
    view_arrays, X_wave = build_views(waves, srates, view_names, L, tf_hw)

    res = train(args, view_arrays, meta, L, tf_hw, view_names)

    if args.scan_k >= 2:
        scan_k(res['cluster_latent'], args.scan_k)

    emb2d, _, proj_name = embed_2d(args, res['cluster_latent'])
    space, labels = cluster(args, res['cluster_latent'], emb2d)

    from sklearn.preprocessing import StandardScaler
    m, valid = metrics_on(StandardScaler().fit_transform(res['cluster_latent']), labels)

    save_outputs(args, res, view_names, X_wave, meta, L, labels, space, emb2d,
                 proj_name, m, valid, truth=truth)


if __name__ == '__main__':
    main()
