# AE feature-extraction comparison framework

Compares **6 feature-extraction methods** for clustering *unlabeled* acoustic-emission
(AE) hits from composites (CFRP/GFRP), under identical data, preprocessing,
clusterers and metrics. Only the feature extractor changes — everything else is
held constant, so differences are attributable to the representation.

| ID | Method | Input | Training | Role |
|----|--------|-------|----------|------|
| M1 | Physical parameters | ~10–15 scalar AE features | none | domain baseline |
| M2 | CAE | time + FFT (1D, 2ch) | reconstruction | DL baseline |
| M3 | CAE + CWT | Morlet scalogram (2D) | reconstruction | M2 contrast |
| M4 | VAE | time + FFT (1D, 2ch) | ELBO | probabilistic latent |
| M5 | SimCLR | time + FFT + augmentations | NT-Xent contrastive | self-supervised |
| M6 | TF-C | time / freq dual view | cross-view consistency | signal-specific |

Two clusterers run on every latent as an internal cross-check: **KMeans** (swept
k = 2..5, best by silhouette) and **HDBSCAN** (auto k, robust to noise). Each
configuration is repeated `--n-runs` times; metrics are reported as mean ± std.

Internal metrics (no labels): **Silhouette** (↑), **Davies–Bouldin** (↓),
**Calinski–Harabasz** (↑). On synthetic data, **ARI/NMI** vs ground truth are
also reported.

For a fair comparison, every latent is standardized and PCA-reduced to a common
width (`--common-dim`, default 16) so M1's ~12 native dims aren't penalized
against wider deep latents.

## Install

```bash
pip install -r tools/requirements.txt   # numpy scipy sklearn matplotlib PyWavelets torch (+umap-learn)
```

`torch` is only needed for M2–M6; M1 and the whole framework run without it.
`umap-learn` is optional (falls back to t-SNE → PCA for the 2D plots).

## Run

```bash
# Synthetic self-test — no .DTA needed, reports ARI/NMI vs ground truth
python tools/run_compare.py --synthetic --epochs 40

# Your data, all methods
python tools/run_compare.py path/to/data.DTA --methods all --denoise wavelet --out cmp_out

# A subset (e.g. skip the slow contrastive ones)
python tools/run_compare.py data.DTA --methods M1 M2 M3 M4 --epochs 80
```

Key flags: `--channel`, `--max-hits`, `--length`, `--denoise {none,wavelet,bandpass,wavelet+bandpass}`,
`--latent-dim`, `--epochs`, `--common-dim`, `--k-min/--k-max`, `--n-runs`,
`--projection {umap,tsne,pca}`, `--device {auto,cpu,cuda}`.

## Outputs (`--out`, default `ae_compare_out/`)

- `comparison.csv` — every (method × clusterer) row with all metrics
- `results.json` — full machine-readable results
- `comparison_kmeans.png`, `comparison_hdbscan.png` — silhouette bar charts
- `embed_<method>.png` — 2D embedding per method, colored by cluster
- `ra_af_<method>.png` — RA–AF damage-mode map (physical validation)
- `latent_<method>.npy` — raw latent matrix per method

## Standard test

```bash
bash tools/run_compare_test.sh            # full self-test (deep methods if torch present)
bash tools/run_compare_test.sh --fast     # M1 only, seconds
bash tools/run_compare_test.sh --install  # pip install deps, then test
```

The test builds labelled synthetic composite-AE data and asserts the pipeline
recovers the known damage modes (ARI). Exit 0 = PASS, 1 = FAIL.

## Notes / honest caveats

- The synthetic generator is a sanity harness, **not** a physical simulator —
  on real data the internal metrics + the RA–AF map are the evidence to trust.
- KMeans selecting k via silhouette can prefer fewer, cleaner clusters than the
  true number of damage modes; report HDBSCAN alongside it.
- JEPA is intentionally omitted: training I-JEPA from scratch on 1e4–1e5 hits is
  not realistic and would underperform TF-C/SimCLR here.
