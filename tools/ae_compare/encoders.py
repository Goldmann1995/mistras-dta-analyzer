"""The 6 feature extractors (M1..M6) under one interface.

Every extractor implements::

    name              short id, e.g. "M2_CAE"
    requires_torch    bool
    available()       -> (ok: bool, reason: str)
    fit_transform(ds, cfg) -> latent (N, d) float32

so the pipeline can treat them uniformly. PyTorch is imported lazily inside the
deep methods, so M1 (and the rest of the framework) runs with torch absent.

    M1  physical parameters     (~10-15 scalar AE features, no training)
    M2  CAE                     1D conv autoencoder on time+FFT (the existing one)
    M3  CAE+CWT                 2D conv autoencoder on Morlet scalograms
    M4  VAE                     1D conv variational autoencoder on time+FFT
    M5  SimCLR                  contrastive learning with signal augmentations
    M6  TF-C                    time/frequency dual-view consistency contrastive
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #
class FeatureExtractor:
    name = "base"
    requires_torch = False

    def available(self):
        if self.requires_torch:
            try:
                import torch  # noqa: F401
            except ImportError:
                return False, "PyTorch not installed (pip install torch)"
        return True, ""

    def fit_transform(self, ds, cfg):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# M1 — physical parameters
# --------------------------------------------------------------------------- #
class PhysicalFeatures(FeatureExtractor):
    name = "M1_physical"
    requires_torch = False

    def available(self):
        return True, ""

    def fit_transform(self, ds, cfg):
        if ds.phys is None:
            return None
        from sklearn.preprocessing import StandardScaler
        return StandardScaler().fit_transform(ds.phys).astype(np.float32)


# --------------------------------------------------------------------------- #
# Shared torch building blocks
# --------------------------------------------------------------------------- #
def _device(cfg):
    import torch
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def _build_1d(nn, length, latent_dim, in_ch):
    """1D conv encoder/decoder pair (same arch as tools/ae_deep_cluster.py)."""
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

    return Encoder, Decoder


def _batches(n, bs, rng):
    perm = rng.permutation(n)
    for b in range(max(1, n // bs)):
        yield perm[b * bs:(b + 1) * bs]


def _train_loop(net, X_t, cfg, loss_fn, tag):
    """Generic AdamW + cosine training loop. ``loss_fn(net, xb) -> scalar``."""
    import torch
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
    N = X_t.shape[0]
    bs = min(cfg.batch_size, N)
    rng = np.random.default_rng(42)
    net.train()
    curve, best, patience = [], float("inf"), 0
    for epoch in range(cfg.epochs):
        run, nb = 0.0, 0
        for idx in _batches(N, bs, rng):
            xb = X_t[torch.from_numpy(idx).to(X_t.device)]
            opt.zero_grad()
            loss = loss_fn(net, xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            run += float(loss.item()); nb += 1
        sched.step()
        avg = run / max(nb, 1)
        curve.append(avg)
        if avg < best * 0.999:
            best, patience = avg, 0
        else:
            patience += 1
        if cfg.verbose and ((epoch + 1) % max(1, cfg.epochs // 5) == 0 or epoch == 0):
            print(f"    [{tag}] epoch {epoch + 1}/{cfg.epochs} loss={avg:.5f}")
        if cfg.early_stop > 0 and patience >= cfg.early_stop:
            if cfg.verbose:
                print(f"    [{tag}] early stop @ epoch {epoch + 1}")
            break
    return curve


# --------------------------------------------------------------------------- #
# M2 — CAE   /   M4 — VAE  (1D conv on time+FFT)
# --------------------------------------------------------------------------- #
class CAEEncoder(FeatureExtractor):
    name = "M2_CAE"
    requires_torch = True
    feature = "both"

    def fit_transform(self, ds, cfg):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        torch.manual_seed(42)
        dev = _device(cfg)
        X = ds.representation(self.feature)
        in_ch = X.shape[1]
        Encoder, Decoder = _build_1d(nn, ds.length, cfg.latent_dim, in_ch)

        class CAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = Encoder(); self.dec = Decoder(self.enc.flat_len)

            def forward(self, x):
                z = self.enc(x); return self.dec(z), z

        net = CAE().to(dev)
        X_t = torch.from_numpy(X).float().to(dev)

        def loss_fn(net, xb):
            recon, _ = net(xb)
            return F.mse_loss(recon, xb)

        _train_loop(net, X_t, cfg, loss_fn, self.name)
        net.eval()
        with torch.no_grad():
            return net.enc(X_t).cpu().numpy().astype(np.float32)


class VAEEncoder(FeatureExtractor):
    name = "M4_VAE"
    requires_torch = True
    feature = "both"

    def fit_transform(self, ds, cfg):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        torch.manual_seed(42)
        dev = _device(cfg)
        X = ds.representation(self.feature)
        in_ch = X.shape[1]
        Encoder, Decoder = _build_1d(nn, ds.length, cfg.latent_dim, in_ch)

        class VAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = Encoder()
                self.fc_mu = nn.Linear(cfg.latent_dim, cfg.latent_dim)
                self.fc_lv = nn.Linear(cfg.latent_dim, cfg.latent_dim)
                self.dec = Decoder(self.enc.flat_len)

            def forward(self, x):
                h = self.enc(x)
                mu, lv = self.fc_mu(h), self.fc_lv(h)
                z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
                return self.dec(z), mu, lv

            def encode(self, x):
                return self.fc_mu(self.enc(x))

        net = VAE().to(dev)
        X_t = torch.from_numpy(X).float().to(dev)
        beta = getattr(cfg, "beta", 1.0)

        def loss_fn(net, xb):
            recon, mu, lv = net(xb)
            kld = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            return F.mse_loss(recon, xb) + beta * kld

        _train_loop(net, X_t, cfg, loss_fn, self.name)
        net.eval()
        with torch.no_grad():
            return net.encode(X_t).cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# M3 — CAE on CWT scalograms (2D conv)
# --------------------------------------------------------------------------- #
class CWTCAEEncoder(FeatureExtractor):
    name = "M3_CWT_CAE"
    requires_torch = True

    def available(self):
        ok, why = super().available()
        if not ok:
            return ok, why
        try:
            import pywt  # noqa: F401
        except ImportError:
            return False, "PyWavelets not installed (pip install PyWavelets)"
        return True, ""

    def fit_transform(self, ds, cfg):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        torch.manual_seed(42)
        dev = _device(cfg)
        n_scales = getattr(cfg, "cwt_scales", 32)
        X = ds.representation("scalogram", n_scales=n_scales)   # (N,1,S,L)
        S, L = X.shape[2], X.shape[3]

        class Enc2D(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(1, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
                    nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                    nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                )
                self.hs, self.ws = S // 8, L // 8
                self.flat = 64 * self.hs * self.ws
                self.fc = nn.Linear(self.flat, cfg.latent_dim)

            def forward(self, x):
                return self.fc(self.conv(x).flatten(1))

        class Dec2D(nn.Module):
            def __init__(self, enc):
                super().__init__()
                self.hs, self.ws = enc.hs, enc.ws
                self.fc = nn.Linear(cfg.latent_dim, enc.flat)
                self.deconv = nn.Sequential(
                    nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),
                    nn.BatchNorm2d(32), nn.ReLU(),
                    nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1),
                    nn.BatchNorm2d(16), nn.ReLU(),
                    nn.ConvTranspose2d(16, 1, 3, 2, 1, output_padding=1),
                )

            def forward(self, z):
                h = self.fc(z).view(z.size(0), 64, self.hs, self.ws)
                return self.deconv(h)

        class CAE2D(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = Enc2D(); self.dec = Dec2D(self.enc)

            def forward(self, x):
                z = self.enc(x)
                r = self.dec(z)
                r = F.interpolate(r, size=(x.shape[2], x.shape[3]),
                                  mode="bilinear", align_corners=False)
                return r, z

        net = CAE2D().to(dev)
        X_t = torch.from_numpy(X).float().to(dev)

        def loss_fn(net, xb):
            recon, _ = net(xb)
            return F.mse_loss(recon, xb)

        _train_loop(net, X_t, cfg, loss_fn, self.name)
        net.eval()
        with torch.no_grad():
            # batched encode to keep memory modest
            outs = []
            for i in range(0, X_t.shape[0], 256):
                outs.append(net.enc(X_t[i:i + 256]).cpu().numpy())
        return np.concatenate(outs).astype(np.float32)


# --------------------------------------------------------------------------- #
# augmentations for contrastive methods
# --------------------------------------------------------------------------- #
def _augment(torch, x):
    """Two stochastic AE-appropriate views: jitter, scaling, time-masking."""
    b, c, L = x.shape
    # amplitude scaling
    scale = (0.8 + 0.4 * torch.rand(b, 1, 1, device=x.device))
    xa = x * scale
    # gaussian jitter
    xa = xa + 0.02 * torch.randn_like(xa)
    # random time mask
    w = max(1, L // 10)
    start = torch.randint(0, max(1, L - w), (1,)).item()
    xa[:, :, start:start + w] = 0.0
    return xa


def _nt_xent(torch, z1, z2, temp=0.5):
    """NT-Xent / InfoNCE over a batch of positive pairs (z1[i], z2[i])."""
    import torch.nn.functional as F
    b = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], 0), dim=1)
    sim = z @ z.t() / temp
    sim.fill_diagonal_(-1e9)
    targets = torch.arange(b, device=z.device)
    targets = torch.cat([targets + b, targets], 0)
    return F.cross_entropy(sim, targets)


# --------------------------------------------------------------------------- #
# M5 — SimCLR on waveforms (augmentation-based contrastive)
# --------------------------------------------------------------------------- #
class SimCLREncoder(FeatureExtractor):
    name = "M5_SimCLR"
    requires_torch = True
    feature = "both"

    def fit_transform(self, ds, cfg):
        import torch
        import torch.nn as nn
        torch.manual_seed(42)
        dev = _device(cfg)
        X = ds.representation(self.feature)
        in_ch = X.shape[1]
        Encoder, _ = _build_1d(nn, ds.length, cfg.latent_dim, in_ch)

        class SimCLR(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = Encoder()
                self.proj = nn.Sequential(
                    nn.Linear(cfg.latent_dim, cfg.latent_dim), nn.ReLU(),
                    nn.Linear(cfg.latent_dim, cfg.latent_dim))

            def forward(self, x):
                return self.proj(self.enc(x))

        net = SimCLR().to(dev)
        X_t = torch.from_numpy(X).float().to(dev)
        temp = getattr(cfg, "temp", 0.5)

        def loss_fn(net, xb):
            return _nt_xent(torch, net(_augment(torch, xb)),
                            net(_augment(torch, xb)), temp)

        _train_loop(net, X_t, cfg, loss_fn, self.name)
        net.eval()
        with torch.no_grad():
            return net.enc(X_t).cpu().numpy().astype(np.float32)   # backbone, not proj head


# --------------------------------------------------------------------------- #
# M6 — TF-C : time / frequency dual-view consistency contrastive
# --------------------------------------------------------------------------- #
class TFCEncoder(FeatureExtractor):
    name = "M6_TFC"
    requires_torch = True

    def fit_transform(self, ds, cfg):
        import torch
        import torch.nn as nn
        torch.manual_seed(42)
        dev = _device(cfg)
        Xt = ds.representation("waveform")     # (N,1,L)
        Xf = ds.representation("fft")          # (N,1,L)
        EncT, _ = _build_1d(nn, ds.length, cfg.latent_dim, 1)
        EncF, _ = _build_1d(nn, ds.length, cfg.latent_dim, 1)

        class TFC(nn.Module):
            def __init__(self):
                super().__init__()
                self.et, self.ef = EncT(), EncF()
                self.pt = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.latent_dim),
                                        nn.ReLU(), nn.Linear(cfg.latent_dim, cfg.latent_dim))
                self.pf = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.latent_dim),
                                        nn.ReLU(), nn.Linear(cfg.latent_dim, cfg.latent_dim))

            def embed(self, xt, xf):
                zt, zf = self.et(xt), self.ef(xf)
                return zt, zf

        net = TFC().to(dev)
        Xt_t = torch.from_numpy(Xt).float().to(dev)
        Xf_t = torch.from_numpy(Xf).float().to(dev)
        temp = getattr(cfg, "temp", 0.5)
        # train on (time, freq) pairs of the same event -> consistency + per-view aug
        N = Xt_t.shape[0]
        bs = min(cfg.batch_size, N)
        rng = np.random.default_rng(42)

        import torch.nn.functional as F  # noqa: F401

        def step(net, idx):
            xt = Xt_t[idx]; xf = Xf_t[idx]
            zt, zf = net.embed(xt, xf)
            ht, hf = net.pt(zt), net.pf(zf)
            # cross-view: same event time<->freq are positives
            l_cross = _nt_xent(torch, ht, hf, temp)
            # within-view augmentation consistency
            zt2, _ = net.embed(_augment(torch, xt), xf)
            l_time = _nt_xent(torch, net.pt(zt), net.pt(zt2), temp)
            return l_cross + 0.5 * l_time

        # custom loop (two inputs) — reuse optimizer/scheduler pattern
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
        net.train()
        best, patience = float("inf"), 0
        for epoch in range(cfg.epochs):
            run, nb = 0.0, 0
            for idx in _batches(N, bs, rng):
                idx_t = torch.from_numpy(idx).to(dev)
                opt.zero_grad()
                loss = step(net, idx_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                run += float(loss.item()); nb += 1
            sched.step()
            avg = run / max(nb, 1)
            if avg < best * 0.999:
                best, patience = avg, 0
            else:
                patience += 1
            if cfg.verbose and ((epoch + 1) % max(1, cfg.epochs // 5) == 0 or epoch == 0):
                print(f"    [{self.name}] epoch {epoch + 1}/{cfg.epochs} loss={avg:.5f}")
            if cfg.early_stop > 0 and patience >= cfg.early_stop:
                break
        net.eval()
        with torch.no_grad():
            zt, zf = net.embed(Xt_t, Xf_t)
            # fused consensus representation
            z = torch.cat([zt, zf], dim=1)
        return z.cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
REGISTRY = {
    "M1": PhysicalFeatures,
    "M2": CAEEncoder,
    "M3": CWTCAEEncoder,
    "M4": VAEEncoder,
    "M5": SimCLREncoder,
    "M6": TFCEncoder,
}


def build(method_id):
    if method_id not in REGISTRY:
        raise ValueError(f"unknown method {method_id}; choose from {list(REGISTRY)}")
    return REGISTRY[method_id]()
