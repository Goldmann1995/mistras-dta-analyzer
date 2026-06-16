# AE Deep Latent-Space Clustering (standalone)

A single self-contained script — **no frontend, no backend, no server**.
It reads your `.DTA` file directly, trains an autoencoder on the raw AE
waveforms, clusters the events in the learned latent space, and writes
figures + a labels table to disk.

```
.DTA waveforms → autoencoder (CAE/VAE) → latent space
              → clustering (KMeans/GMM/HDBSCAN/DBSCAN)
              → 2D map (PCA/t-SNE) + medoid prototype waveforms
```

## 1. Install dependencies

From the repo root (use your `mistrasdta` conda env):

```bash
pip install -r tools/requirements.txt
```

(That's `numpy scipy scikit-learn matplotlib torch`. `torch` is the only
big one, ~1–2 min to install. CPU-only is fine.)

## 2. Run / train

The only required argument is your data file:

```bash
python tools/ae_deep_cluster.py path/to/yourdata.DTA
```

Common variations:

```bash
# more training, 4 clusters with K-Means (default)
python tools/ae_deep_cluster.py data.DTA --epochs 80 --clusters 4

# Variational Autoencoder + GMM
python tools/ae_deep_cluster.py data.DTA --model vae --algorithm gmm --clusters 5

# density-based clustering that auto-discovers the number of clusters
python tools/ae_deep_cluster.py data.DTA --algorithm hdbscan --min-samples 15

# only one sensor, keep pre-trigger, t-SNE map, GPU if available
python tools/ae_deep_cluster.py data.DTA --channel 1 --keep-pretrigger \
    --projection tsne --device auto
```

See every option:

```bash
python tools/ae_deep_cluster.py --help
```

### What "training" means here
The autoencoder learns, fully unsupervised, to compress each waveform into a
small latent vector (`--latent-dim`, default 16) and reconstruct it. You do
**not** need labels. Watch `loss` drop in the console / `loss_curve.png`; if
it's still falling at the end, raise `--epochs`.

## 3. Outputs

Everything lands in `--out` (default `ae_cluster_out/`):

| file | meaning |
|------|---------|
| `loss_curve.png`     | training reconstruction loss per epoch |
| `latent_scatter.png` | 2D latent map, points colored by cluster |
| `prototypes.png`     | the medoid (most representative) waveform of each cluster |
| `cluster_labels.csv` | per-waveform: wfm index, channel, time, cluster, latent dims |
| `latent_codes.npy`   | raw latent matrix `(N, latent_dim)` for your own analysis |
| `summary.txt`        | config, cluster sizes, quality metrics |

`cluster_labels.csv` is the key artifact — join it back to your hit table by
`wfm_index` to interpret each cluster physically (frequency, energy, etc.).

## 4. Tuning cheatsheet

| Goal | Knob |
|------|------|
| Loss not converged | `--epochs` ↑ |
| Richer / coarser features | `--latent-dim` ↑ / ↓ |
| Don't know cluster count | `--algorithm hdbscan` |
| Fix cluster count | `--algorithm kmeans --clusters N` |
| Noisy/short signals | `--length` to match your record size |
| Too slow | `--max-waveforms` ↓, keep `--projection pca` |
| Best separation view | `--projection tsne` (slower) |

## 5. Interpretability tip
The latent clusters are a "black box". To get **explainable rules** like
`IF peak_freq > 300 kHz AND energy < 50 → cluster 0`, take
`cluster_labels.csv`, attach the parametric hit features, and fit a shallow
decision tree on `cluster` (e.g. `sklearn.tree.DecisionTreeClassifier`).
The in-app "Clustering" tab already does this for parametric features.
