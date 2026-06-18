#!/usr/bin/env python3
"""Standard self-test for the AE comparison framework (tools/ae_compare).

Runs the full pipeline on labelled SYNTHETIC composite-AE data — no .DTA file
needed — and checks that:

  1. the framework runs end to end and writes its expected output files;
  2. M1 (physical features) recovers the known damage modes with high ARI
     (a real signal => the metrics/clustering plumbing is correct);
  3. any deep method that can run (torch installed) also clears a sane bar.

Exit code 0 = PASS, 1 = FAIL. Designed to run on CPU in well under a minute.

Usage:
    python tools/test_compare.py            # auto: M1 + deep methods if torch
    python tools/test_compare.py --fast     # M1 only, tiny data (seconds)
"""

import os
import sys
import shutil
import argparse
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.ae_compare import data as data_mod          # noqa: E402
from tools.ae_compare.encoders import REGISTRY         # noqa: E402
from tools.ae_compare.pipeline import run_comparison   # noqa: E402

# Pass criteria
ARI_PASS_M1 = 0.60     # physical features must clearly recover the 4 modes
ARI_PASS_DEEP = 0.30   # deep nets w/ tiny epochs: looser, just "learned something"


def torch_available():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="M1 only, tiny synthetic data")
    ap.add_argument("--out", default="ae_compare_selftest")
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)

    has_torch = torch_available()
    if args.fast or not has_torch:
        methods = ["M1"]
        n, length = 300, 256
        if not has_torch and not args.fast:
            print("[selftest] PyTorch not installed -> testing M1 only "
                  "(deep methods M2-M6 require torch).")
    else:
        methods = list(REGISTRY.keys())
        n, length = 500, 384

    print(f"[selftest] methods={methods}  n={n}  length={length}  "
          f"torch={'yes' if has_torch else 'no'}")

    ds = data_mod.make_synthetic(n=n, length=length, n_classes=4, noise=0.05, seed=0)

    cfg = SimpleNamespace(
        out=args.out, methods=methods,
        latent_dim=16, epochs=args.epochs, batch_size=64, lr=1e-3, beta=0.1,
        temp=0.5, early_stop=10, cwt_scales=24, device="cpu", verbose=False,
        common_dim=8, k_min=2, k_max=5, n_runs=3,
        min_cluster_size=20, min_samples=5, projection="pca",  # pca: no umap dep
    )
    results = run_comparison(ds, cfg)

    # ---- assertions ----
    failures = []

    expected_files = ["comparison.csv", "results.json", "comparison_kmeans.png"]
    for fn in expected_files:
        if not os.path.exists(os.path.join(args.out, fn)):
            failures.append(f"missing output file: {fn}")

    def ari_of(name):
        """Best ARI across both clusterers. KMeans selects k by silhouette,
        which can prefer fewer-but-cleaner clusters than ground truth; HDBSCAN
        finds its own k. 'Did the features recover the modes?' is best answered
        by the stronger of the two."""
        res = results.get(name, {})
        if "clusters" not in res:
            return None
        aris = [agg.get("ari_mean") for agg in res["clusters"].values()
                if agg.get("ari_mean") is not None]
        return max(aris) if aris else None

    m1 = ari_of("M1_physical")
    if m1 is None:
        failures.append("M1_physical produced no ARI (pipeline broken)")
    elif m1 < ARI_PASS_M1:
        failures.append(f"M1_physical ARI={m1:.3f} < {ARI_PASS_M1} "
                        "(framework not recovering known modes)")
    else:
        print(f"[selftest] OK  M1_physical ARI={m1:.3f} >= {ARI_PASS_M1}")

    for mid, cls in REGISTRY.items():
        if mid == "M1" or mid not in methods:
            continue
        name = cls.name
        res = results.get(name, {})
        if "skipped" in res:
            print(f"[selftest] -- {name} skipped: {res['skipped']}")
            continue
        ari = ari_of(name)
        if ari is None:
            failures.append(f"{name} ran but produced no ARI")
        elif ari < ARI_PASS_DEEP:
            failures.append(f"{name} ARI={ari:.3f} < {ARI_PASS_DEEP}")
        else:
            print(f"[selftest] OK  {name} ARI={ari:.3f} >= {ARI_PASS_DEEP}")

    print("\n" + "=" * 60)
    if failures:
        print("SELF-TEST FAILED:")
        for f in failures:
            print("  - " + f)
        print("=" * 60)
        sys.exit(1)
    print("SELF-TEST PASSED — framework is working end to end.")
    print(f"Inspect plots/tables in: {args.out}/")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
