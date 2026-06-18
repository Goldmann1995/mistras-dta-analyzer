"""Plots: per-method 2D embedding, RA-AF damage-mode map, and the summary bar.

All figures are written with the Agg backend so this works headless on a server.
"""

import os

import numpy as np


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _project_2d(Z, method="umap", seed=42):
    """Reduce to 2D for visualization (UMAP -> t-SNE -> PCA fallback)."""
    if Z.shape[1] <= 2:
        return Z, "raw"
    if method == "umap":
        try:
            import umap
            return umap.UMAP(n_components=2, random_state=seed).fit_transform(Z), "UMAP"
        except ImportError:
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE
        perp = float(min(30, max(2, Z.shape[0] // 4), Z.shape[0] - 1))
        return TSNE(2, random_state=seed, perplexity=perp, init="pca").fit_transform(Z), "t-SNE"
    from sklearn.decomposition import PCA
    return PCA(2, random_state=seed).fit_transform(Z), "PCA"


def _color(plt, l):
    cmap = plt.get_cmap("tab10")
    return (0.3, 0.3, 0.3, 0.4) if l == -1 else cmap(int(l) % 10)


def embedding_plot(out_dir, method_name, Z, labels, proj="umap", title_extra="", suffix=""):
    plt = _mpl()
    emb, pname = _project_2d(Z, proj)
    fig, ax = plt.subplots(figsize=(7, 6))
    for l in sorted(set(labels)):
        pts = emb[labels == l]
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color=_color(plt, l), alpha=0.75,
                   edgecolors="none",
                   label=("noise" if l == -1 else f"C{l} (n={int(np.sum(labels == l))})"))
    ax.set_xlabel(f"{pname}-1"); ax.set_ylabel(f"{pname}-2")
    ax.set_title(f"{method_name} — {pname} embedding {title_extra}")
    ax.legend(markerscale=1.6, fontsize=8, loc="best"); ax.grid(alpha=0.2)
    plt.tight_layout()
    p = os.path.join(out_dir, f"embed_{method_name}{suffix}.png")
    plt.savefig(p, dpi=150); plt.close()
    return p


def ra_af_plot(out_dir, method_name, ra, af, labels, suffix=""):
    """RA (rise/amp) vs AF (avg freq) coloured by cluster — Aggelis-style
    damage-mode validation. Tensile/matrix cracking sits high-AF/low-RA;
    shear/delamination sits low-AF/high-RA."""
    if ra is None or af is None:
        return None
    mask = np.isfinite(ra) & np.isfinite(af)
    if np.sum(mask) < 3:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 6))
    for l in sorted(set(labels)):
        m = mask & (labels == l)
        if np.sum(m) == 0:
            continue
        ax.scatter(ra[m], af[m], s=16, color=_color(plt, l), alpha=0.7,
                   edgecolors="none",
                   label=("noise" if l == -1 else f"C{l}"))
    ax.set_xlabel("RA = rise time / amplitude  (us/dB)")
    ax.set_ylabel("AF = average frequency  (kHz)")
    ax.set_title(f"{method_name} — RA-AF damage-mode map")
    ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.25)
    plt.tight_layout()
    p = os.path.join(out_dir, f"ra_af_{method_name}{suffix}.png")
    plt.savefig(p, dpi=150); plt.close()
    return p


def amplitude_vs_freq_plot(out_dir, method_name, freqs, amps, labels, suffix=""):
    """AE hit amplitude (dB) vs frequency (kHz), coloured by cluster."""
    if freqs is None or amps is None:
        return None
    mask = np.isfinite(freqs) & np.isfinite(amps)
    if np.sum(mask) < 3:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 5))
    for l in sorted(set(labels)):
        m = mask & (labels == l)
        if np.sum(m) == 0:
            continue
        ax.scatter(freqs[m], amps[m], s=16, alpha=0.7, color=_color(plt, l),
                   edgecolors="none",
                   label=("noise" if l == -1 else f"C{l} (n={int(np.sum(m))})"))
    ax.set_xlabel("Frequency (kHz)"); ax.set_ylabel("Amplitude (dB)")
    ax.set_title(f"{method_name} — AE amplitude vs frequency by cluster")
    ax.legend(markerscale=1.5, fontsize=8, loc="best"); ax.grid(alpha=0.25)
    plt.tight_layout()
    p = os.path.join(out_dir, f"amp_freq_{method_name}{suffix}.png")
    plt.savefig(p, dpi=150); plt.close()
    return p


def time_vs_freq_plot(out_dir, method_name, times, freqs, labels, suffix=""):
    """AE hit time (s) vs frequency (kHz), coloured by cluster — shows how the
    clusters evolve over the loading history."""
    if times is None or freqs is None:
        return None
    mask = np.isfinite(times) & np.isfinite(freqs)
    if np.sum(mask) < 3:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(10, 5))
    for l in sorted(set(labels)):
        m = mask & (labels == l)
        if np.sum(m) == 0:
            continue
        ax.scatter(times[m], freqs[m], s=16, alpha=0.7, color=_color(plt, l),
                   edgecolors="none",
                   label=("noise" if l == -1 else f"C{l} (n={int(np.sum(m))})"))
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (kHz)")
    ax.set_title(f"{method_name} — AE time vs frequency by cluster")
    ax.legend(markerscale=1.5, fontsize=8, loc="best"); ax.grid(alpha=0.25)
    plt.tight_layout()
    p = os.path.join(out_dir, f"time_freq_{method_name}{suffix}.png")
    plt.savefig(p, dpi=150); plt.close()
    return p


def summary_bar(out_dir, results, clusterer="kmeans"):
    """Grouped bar of silhouette per method (mean +/- std) for quick comparison."""
    plt = _mpl()
    names, mean, std = [], [], []
    for name, res in results.items():
        agg = res["clusters"].get(clusterer)
        if not agg or agg.get("silhouette_mean") is None:
            continue
        names.append(name)
        mean.append(agg["silhouette_mean"])
        std.append(agg["silhouette_std"] or 0.0)
    if not names:
        return None
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(names)), 5))
    x = np.arange(len(names))
    ax.bar(x, mean, yerr=std, capsize=4, color="#0891b2", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Silhouette (higher = better)")
    ax.set_title(f"Feature-method comparison — {clusterer} (mean +/- std over runs)")
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    p = os.path.join(out_dir, f"comparison_{clusterer}.png")
    plt.savefig(p, dpi=150); plt.close()
    return p
