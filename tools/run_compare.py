#!/usr/bin/env python3
"""Unified comparison of 6 AE feature-extraction methods (M1..M6).

Runs every method on ONE dataset with identical preprocessing, the same two
clusterers (KMeans swept over k, and HDBSCAN), and the same internal metrics
(silhouette / Davies-Bouldin / Calinski-Harabasz), then writes per-method
embedding + RA-AF plots and a comparison table.

Methods:
    M1  physical parameters (domain baseline, no training)
    M2  CAE                 1D conv autoencoder on time+FFT
    M3  CAE+CWT             2D conv autoencoder on Morlet scalograms
    M4  VAE                 1D conv variational autoencoder
    M5  SimCLR              augmentation-based contrastive
    M6  TF-C                time/frequency dual-view consistency contrastive

Examples:
    # real data
    python tools/run_compare.py data.DTA --methods M1 M2 M4 --epochs 60
    python tools/run_compare.py data.DTA --methods all --denoise wavelet --out cmp_out

    # synthetic self-test (no .DTA needed; reports ARI/NMI vs ground truth)
    python tools/run_compare.py --synthetic --epochs 30
"""

import os
import sys
import argparse
from types import SimpleNamespace

# allow `python tools/run_compare.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.ae_compare import data as data_mod      # noqa: E402
from tools.ae_compare.pipeline import run_comparison  # noqa: E402
from tools.ae_compare.encoders import REGISTRY      # noqa: E402

ALL_METHODS = ["M1", "M2", "M3", "M4", "M5", "M6"]


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="path to a .DTA file (omit with --synthetic)")
    ap.add_argument("--synthetic", action="store_true",
                    help="generate labelled synthetic AE data instead of reading a .DTA")
    ap.add_argument("--out", default="ae_compare_out", help="output directory")
    ap.add_argument("--methods", nargs="+", default=["all"],
                    help="subset of {M1..M6} or 'all'")

    g = ap.add_argument_group("data / preprocessing")
    g.add_argument("--channel", type=int, default=None)
    g.add_argument("--max-hits", type=int, default=4000, dest="max_hits")
    g.add_argument("--length", type=int, default=0, dest="fixed_length",
                   help="waveform length; 0 = auto-detect")
    g.add_argument("--keep-pretrigger", action="store_true", dest="keep_pretrigger")
    g.add_argument("--denoise", default="none",
                   choices=["none", "wavelet", "bandpass", "wavelet+bandpass"])
    g.add_argument("--denoise-band", type=float, nargs=2, default=[20.0, 400.0],
                   dest="denoise_band", metavar=("LOW_kHz", "HIGH_kHz"))

    g = ap.add_argument_group("synthetic")
    g.add_argument("--syn-n", type=int, default=600, dest="syn_n")
    g.add_argument("--syn-classes", type=int, default=4, dest="syn_classes")
    g.add_argument("--syn-length", type=int, default=512, dest="syn_length")
    g.add_argument("--syn-noise", type=float, default=0.05, dest="syn_noise")
    g.add_argument("--seed", type=int, default=0)

    g = ap.add_argument_group("representation learning (deep methods)")
    g.add_argument("--latent-dim", type=int, default=32, dest="latent_dim")
    g.add_argument("--epochs", type=int, default=60)
    g.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--beta", type=float, default=0.1,
                   help="VAE KL weight (low avoids posterior collapse on short AE bursts)")
    g.add_argument("--temp", type=float, default=0.5, help="contrastive temperature")
    g.add_argument("--early-stop", type=int, default=15, dest="early_stop")
    g.add_argument("--cwt-scales", type=int, default=32, dest="cwt_scales")
    g.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    g.add_argument("--quiet", action="store_true", help="suppress per-epoch logs")

    g = ap.add_argument_group("clustering + comparison")
    g.add_argument("--common-dim", type=int, default=16, dest="common_dim",
                   help="PCA all latents to this dim for a fair comparison (0=off)")
    g.add_argument("--k-min", type=int, default=2, dest="k_min")
    g.add_argument("--k-max", type=int, default=5, dest="k_max")
    g.add_argument("--n-runs", type=int, default=5, dest="n_runs",
                   help="repeat clustering N times, report mean +/- std")
    g.add_argument("--min-cluster-size", type=int, default=30, dest="min_cluster_size")
    g.add_argument("--min-samples", type=int, default=5, dest="min_samples")
    g.add_argument("--projection", choices=["umap", "tsne", "pca"], default="umap")
    return ap


def resolve_methods(req):
    if "all" in req:
        return ALL_METHODS
    bad = [m for m in req if m not in REGISTRY]
    if bad:
        raise SystemExit(f"unknown methods {bad}; choose from {ALL_METHODS} or 'all'")
    return req


def main():
    args = build_parser().parse_args()
    methods = resolve_methods(args.methods)

    if args.synthetic:
        ds = data_mod.make_synthetic(
            n=args.syn_n, length=args.syn_length, n_classes=args.syn_classes,
            noise=args.syn_noise, seed=args.seed)
    else:
        if not args.input:
            raise SystemExit("provide a .DTA path, or use --synthetic")
        denoiser = data_mod.make_denoiser(args.denoise, band=tuple(args.denoise_band))
        ds = data_mod.load_dta(args.input, channel=args.channel,
                               max_hits=args.max_hits, fixed_length=args.fixed_length,
                               keep_pretrigger=args.keep_pretrigger, denoiser=denoiser)

    cfg = SimpleNamespace(
        out=args.out, methods=methods,
        latent_dim=args.latent_dim, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, beta=args.beta, temp=args.temp, early_stop=args.early_stop,
        cwt_scales=args.cwt_scales, device=args.device, verbose=not args.quiet,
        common_dim=args.common_dim, k_min=args.k_min, k_max=args.k_max,
        n_runs=args.n_runs, min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples, projection=args.projection,
    )
    run_comparison(ds, cfg)


if __name__ == "__main__":
    main()
