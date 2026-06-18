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

---

# STCD-AE — multi-view contrastive alignment clustering (`stcd_ae.py`)

A second, **research-grade** standalone script. Where `ae_deep_cluster.py`
trains one autoencoder on one representation, **STCD-AE learns ONE shared
latent space in which several physical "views" of the same AE event are
aligned**, then clusters on the aligned consensus. This is the part you asked
for: *几个表示在潜空间内对齐 + 可视化*.

```
each hit ─► time  (waveform)   ─► encoder_T ┐
         ─► freq  (FFT)         ─► encoder_F ├─► aligned latent ─► cluster
         ─► tf    (spectrogram) ─► encoder_S ┘        │
   cross-view alignment pulls the 3 views of ONE event together,
   reconstruction keeps the codes physical, no view is allowed to dominate.
```

### Why it is genuinely new (not just SimCLR re-skinned)
* **Positive pairs are deterministic, information-complementary physical
  transforms** (time / spectrum / spectrogram) of the *same* signal — not two
  random augmentations of one image. Alignment therefore forces the latent to
  keep only the cross-domain-invariant *damage-source* signature and drop
  view-specific nuisances (windowing, propagation amplitude/decay).
* **Default alignment loss is non-contrastive (VICReg).** Plain InfoNCE is
  instance discrimination — it aligns views but *repels events of the same
  damage mode*, so it aligns yet **anti-clusters**. VICReg aligns (invariance)
  + prevents collapse (variance) + decorrelates (covariance) with **no
  same-mode repulsion**, so the latent clusters *and* aligns. (`--align-loss
  infonce` is kept as the ablation that demonstrates this.)
* **Fusion beats every single view.** On the synthetic check, time alone /
  freq alone reach ARI ≈ 0.5–0.75; the aligned fusion reaches **ARI ≈ 1.0**.
* **Unsupervised model selection.** It snapshots every few epochs and keeps the
  one with the best *combined* (cluster silhouette + cross-view alignment)
  score — no labels, immune to over/under-training.
* **Optional physics-disentanglement** (`--disentangle`): splits the latent
  into `z_source` / `z_propagation`, uses the parametric AE features
  (peak/centroid freq vs amplitude/energy/duration) as weak supervision plus a
  gradient-reversal adversary, and clusters on the distance-invariant
  `z_source`.

### Validate the install (no data needed)
```bash
python tools/stcd_ae.py --self-test
```
Builds a synthetic 4-mode dataset where **frequency is the true signal** and
amplitude/decay are nuisances (your exact situation), trains, and prints ARI/NMI
against the known modes (expect ≈ 1.0).

### Run on your data
```bash
# recommended starting point (UMAP needs: pip install umap-learn)
python tools/stcd_ae.py data.DTA --views time,freq,tf --clusters 4 --projection umap

# lighter / faster: two 1-D views only (no spectrogram)
python tools/stcd_ae.py data.DTA --views time,freq --clusters 4

# let the data choose k
python tools/stcd_ae.py data.DTA --scan-k 8 --algorithm hdbscan --projection umap

# physics-disentangled: cluster on the propagation-invariant source subspace
python tools/stcd_ae.py data.DTA --disentangle --source-dim 8 --clusters 4

# ablation for your paper: alignment that anti-clusters
python tools/stcd_ae.py data.DTA --align-loss infonce
```

### Outputs (into `--out`, default `stcd_out/`)
| file | meaning |
|------|---------|
| `alignment_2d.png`     | **headline figure** — each view in its own color, the 3 views of one event joined by a gray link; short links + overlapping colors = aligned |
| `alignment_curve.png`  | same-event vs different-event cosine over epochs (the gap = alignment quality); selected epoch marked |
| `per_view_scatter.png` | each view embedded separately, colored by the shared clusters — shows which view separates which modes |
| `latent_scatter.png`   | 2D embedding of the aligned consensus, colored by cluster |
| `loss_curves.png`      | reconstruction / alignment / total loss |
| `cluster_spectra.png`  | mean FFT per cluster (physical signature) |
| `prototypes.png`       | medoid waveform per cluster |
| `cluster_labels.csv` / `cluster_features.csv` / `latent_codes.npy` / `summary.txt` | as in `ae_deep_cluster.py` |

### Key knobs
| Goal | Knob |
|------|------|
| Which views to align | `--views time,freq,tf` (≥2 required) |
| Clusters too diffuse | lower `--latent-dim` (VICReg spreads every dim; 8–16 is good) |
| Don't know k | `--algorithm hdbscan --projection umap` or `--scan-k 8` |
| Stronger/weaker alignment | `--contrast-weight` ↑/↓ |
| Remove propagation effect | `--disentangle --source-dim 8` |
| GPU | `--device cuda` (auto-detected) |

`silhouette` in `summary.txt` is computed on the aligned latent — report that
one in papers; the alignment gap quantifies how well the views agree.
