"""Deep-learning latent-space clustering for AE waveforms.

Implements the mainstream "raw waveform -> deep latent representation ->
latent-space clustering" pipeline from the AE literature:

  - CAE  : 1D Convolutional Autoencoder  (cf. Krishna et al. 2026; Wang 2023)
  - VAE  : Variational Autoencoder        (cf. Cui & Yan 2026, AVAE)

After unsupervised representation learning, the latent codes are clustered
with K-Means / GMM / HDBSCAN, projected to 2D (PCA or t-SNE) for
visualization, and each cluster is summarized by its medoid waveform
(prototype) for physical interpretation.

PyTorch is imported lazily so the rest of the backend runs without it.
"""

import numpy as np
from typing import Optional

from MistrasDTA import get_waveform_data


def _round_up_multiple(n: int, m: int) -> int:
    return int(np.ceil(n / m) * m)


def _prepare_waveforms(
    wfm_data,
    channel: Optional[int],
    max_waveforms: int,
    fixed_length: int,
    keep_pretrigger: bool,
):
    """Return (X, indices) where X is (N, L) float32, peak-normalized per row."""
    mask = np.ones(len(wfm_data), dtype=bool)
    if channel is not None:
        mask &= wfm_data['CH'] == channel
    idx_all = np.where(mask)[0]

    if len(idx_all) == 0:
        raise ValueError("No waveforms match the selection")

    if len(idx_all) > max_waveforms:
        sel = np.linspace(0, len(idx_all) - 1, max_waveforms).astype(int)
        idx_all = idx_all[sel]

    L = fixed_length
    rows = []
    kept = []
    for i in idx_all:
        row = wfm_data[i]
        t, V = get_waveform_data(row)
        if not keep_pretrigger and row['TDLY'] < 0:
            trim = abs(int(row['TDLY']))
            V = V[trim:]
        if len(V) == 0:
            continue
        if len(V) >= L:
            v = V[:L]
        else:
            v = np.pad(V, (0, L - len(V)))
        peak = np.max(np.abs(v))
        if peak > 0:
            v = v / peak
        rows.append(v.astype(np.float32))
        kept.append(int(i))

    if len(rows) < 4:
        raise ValueError("Not enough valid waveforms for deep clustering")

    return np.stack(rows), np.array(kept)


def compute_deep_clustering(
    wfm_data,
    rec_data,
    model: str = 'cae',
    latent_dim: int = 16,
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    fixed_length: int = 1024,
    max_waveforms: int = 2000,
    keep_pretrigger: bool = False,
    algorithm: str = 'kmeans',
    n_clusters: int = 4,
    eps: float = 0.8,
    min_samples: int = 10,
    projection: str = 'pca',
    beta: float = 1.0,
    channel: Optional[int] = None,
) -> dict:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        raise ValueError(
            "PyTorch is required for deep clustering. Install with: pip install torch"
        )

    from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    )

    torch.manual_seed(42)
    np.random.seed(42)

    L = _round_up_multiple(max(64, fixed_length), 8)
    X, indices = _prepare_waveforms(
        wfm_data, channel, max_waveforms, L, keep_pretrigger,
    )
    N = X.shape[0]

    device = torch.device('cpu')
    X_t = torch.from_numpy(X).unsqueeze(1).to(device)  # (N, 1, L)

    # ---- Model definitions ----
    class Encoder(nn.Module):
        def __init__(self, length, ldim):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 16, 9, stride=2, padding=4), nn.ReLU(),
                nn.Conv1d(16, 32, 9, stride=2, padding=4), nn.ReLU(),
                nn.Conv1d(32, 64, 9, stride=2, padding=4), nn.ReLU(),
            )
            self.flat_len = (length // 8) * 64
            self.length = length
            self.fc = nn.Linear(self.flat_len, ldim)

        def forward(self, x):
            h = self.conv(x).flatten(1)
            return self.fc(h), h

    class Decoder(nn.Module):
        def __init__(self, length, ldim, flat_len):
            super().__init__()
            self.length = length
            self.fc = nn.Linear(ldim, flat_len)
            self.deconv = nn.Sequential(
                nn.ConvTranspose1d(64, 32, 9, stride=2, padding=4, output_padding=1), nn.ReLU(),
                nn.ConvTranspose1d(32, 16, 9, stride=2, padding=4, output_padding=1), nn.ReLU(),
                nn.ConvTranspose1d(16, 1, 9, stride=2, padding=4, output_padding=1),
            )

        def forward(self, z):
            h = self.fc(z).view(z.size(0), 64, self.length // 8)
            return self.deconv(h)

    class CAE(nn.Module):
        def __init__(self, length, ldim):
            super().__init__()
            self.enc = Encoder(length, ldim)
            self.dec = Decoder(length, ldim, self.enc.flat_len)

        def forward(self, x):
            z, _ = self.enc(x)
            return self.dec(z), z

        def encode(self, x):
            return self.enc(x)[0]

    class VAE(nn.Module):
        def __init__(self, length, ldim):
            super().__init__()
            self.enc = Encoder(length, ldim)
            self.fc_mu = nn.Linear(ldim, ldim)
            self.fc_logvar = nn.Linear(ldim, ldim)
            self.dec = Decoder(length, ldim, self.enc.flat_len)

        def forward(self, x):
            h, _ = self.enc(x)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            return self.dec(z), mu, logvar

        def encode(self, x):
            h, _ = self.enc(x)
            return self.fc_mu(h)

    net = (VAE(L, latent_dim) if model == 'vae' else CAE(L, latent_dim)).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # ---- Training ----
    net.train()
    loss_curve = []
    n_batches = max(1, N // batch_size)
    for epoch in range(epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            xb = X_t[idx]
            optimizer.zero_grad()
            if model == 'vae':
                recon, mu, logvar = net(xb)
                rec_loss = F.mse_loss(recon, xb, reduction='mean')
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = rec_loss + beta * kl
            else:
                recon, _ = net(xb)
                loss = F.mse_loss(recon, xb, reduction='mean')
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        loss_curve.append(round(epoch_loss / n_batches, 6))

    # ---- Latent extraction ----
    net.eval()
    with torch.no_grad():
        latent = net.encode(X_t).cpu().numpy()

    latent_scaled = StandardScaler().fit_transform(latent)

    # ---- Latent-space clustering ----
    if algorithm == 'kmeans':
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(latent_scaled)
    elif algorithm == 'gmm':
        labels = GaussianMixture(n_components=n_clusters, random_state=42, n_init=3).fit_predict(latent_scaled)
    elif algorithm == 'dbscan':
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(latent_scaled)
    elif algorithm == 'hdbscan':
        labels = HDBSCAN(min_cluster_size=max(min_samples, 5)).fit_predict(latent_scaled)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    valid_labels = sorted(l for l in set(labels) if l >= 0)
    noise_points = int(np.sum(labels == -1))

    # ---- Metrics ----
    metrics = {}
    nn_mask = labels >= 0
    if len(valid_labels) >= 2 and np.sum(nn_mask) > len(valid_labels):
        metrics['silhouette'] = float(silhouette_score(latent_scaled[nn_mask], labels[nn_mask]))
        metrics['calinski_harabasz'] = float(calinski_harabasz_score(latent_scaled[nn_mask], labels[nn_mask]))
        metrics['davies_bouldin'] = float(davies_bouldin_score(latent_scaled[nn_mask], labels[nn_mask]))

    # ---- 2D projection for visualization ----
    if latent.shape[1] <= 2:
        proj = latent[:, :2] if latent.shape[1] == 2 else np.column_stack([latent[:, 0], np.zeros(N)])
        proj_name = 'latent'
    elif projection == 'tsne':
        perplexity = float(min(30, max(2, N // 4), N - 1))
        proj = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca').fit_transform(latent)
        proj_name = 't-SNE'
    else:
        proj = PCA(n_components=2, random_state=42).fit_transform(latent)
        proj_name = 'PCA'

    scatter_data = [
        {'x': float(proj[i, 0]), 'y': float(proj[i, 1]),
         'cluster': int(labels[i]), 'index': int(indices[i])}
        for i in range(N)
    ]

    # ---- Cluster stats + medoid prototypes ----
    cluster_stats = []
    prototypes = []
    for label in valid_labels:
        cmask = labels == label
        members = np.where(cmask)[0]
        centroid = latent_scaled[cmask].mean(axis=0)
        dists = np.linalg.norm(latent_scaled[members] - centroid, axis=1)
        medoid_local = members[int(np.argmin(dists))]
        medoid_global_idx = int(indices[medoid_local])

        count = int(cmask.sum())
        stat = {
            'label': int(label),
            'count': count,
            'percentage': round(100.0 * count / N, 1),
            'medoid_index': medoid_global_idx,
        }
        cluster_stats.append(stat)

        # medoid waveform for plotting (downsampled)
        wf_row = wfm_data[medoid_global_idx]
        t, V = get_waveform_data(wf_row)
        if not keep_pretrigger and wf_row['TDLY'] < 0:
            trim = abs(int(wf_row['TDLY']))
            t = t[trim:] - t[trim]
            V = V[trim:]
        step = max(1, len(V) // 800)
        prototypes.append({
            'cluster': int(label),
            'index': medoid_global_idx,
            'channel': int(wf_row['CH']),
            'sample_rate': float(wf_row['SRATE']),
            'time': t[::step].tolist(),
            'waveform': V[::step].tolist(),
        })

    return {
        'model': model,
        'latent_dim': int(latent.shape[1]),
        'n_waveforms': N,
        'waveform_length': L,
        'epochs': epochs,
        'final_loss': loss_curve[-1] if loss_curve else 0.0,
        'loss_curve': loss_curve,
        'algorithm': algorithm,
        'n_clusters': len(valid_labels),
        'noise_points': noise_points,
        'projection': proj_name,
        'scatter_data': scatter_data,
        'cluster_stats': cluster_stats,
        'prototypes': prototypes,
        'metrics': metrics,
    }
