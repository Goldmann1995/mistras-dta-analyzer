# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

This repo wraps the upstream **MistrasDTA** PyPI package (a binary parser for AEWin
acoustic-emission `.DTA` files) with a full analysis platform built around it:

- `MistrasDTA/` — the original, unmodified-in-spirit binary parser (`read_bin`, `get_waveform_data`). Everything else depends on it.
- `backend/` — FastAPI server exposing the parser plus signal-analysis and clustering endpoints.
- `frontend/` — React + Vite single-page app (AEWin-like UI) that talks to the backend over `/api`.
- `tools/ae_deep_cluster.py` — a standalone, server-free deep-clustering script.
- Root scripts (`main.py`, `inspect_data.py`, `dta_collection_guide.py`) — demo/exploration scripts (Chinese comments) that use the parser directly against sample `.DTA` files in the repo root.

## Commands

Python is managed with **uv** (`uv.lock`, workspace defined in root `pyproject.toml` with `backend` as a member).

```bash
# Backend (auto-reloads): serves on http://localhost:8000, mounts frontend/dist at / if built
python run.py
# or: uvicorn backend.app.main:app --reload

# Backend deps (installs torch, scipy, PyWavelets, EMD-signal, etc.)
pip install -r backend/requirements.txt    # or: uv sync

# Frontend (dev server proxies /api -> :8000)
cd frontend && npm install && npm run dev
cd frontend && npm run build      # tsc -b && vite build -> frontend/dist
cd frontend && npm run lint       # eslint

# Standalone deep clustering (no server)
pip install -r tools/requirements.txt
python tools/ae_deep_cluster.py path/to/data.DTA --feature both --clusters 4
python tools/ae_deep_cluster.py --help
```

There is no test suite in this repo. The `.github/workflows/tests.yml` (pytest/coveralls,
`workflow_dispatch` only) is inherited from upstream MistrasDTA and references a `tests/`
dir that is not present here.

## Architecture

### The parser is the contract (`MistrasDTA/MistrasDTA.py`)
`read_bin(file, skip_wfm=False)` walks a stream of length-prefixed, ID-tagged binary
messages (format = Appendix II of the Mistras manual) and returns two numpy **recarrays**:
- `rec` — one row per AE hit. Fields are added dynamically from the file's `CHID_list`, so **which columns exist depends on the file**. Always guard access with `if 'AMP' in rec.dtype.names`.
- `wfm` — one row per saved waveform, with the samples packed into a `WAVEFORM` byte field. `get_waveform_data(row)` unpacks it to `(t_microseconds, V_volts)`.

Field names are the raw, cryptic recarray names everything downstream must use:
the time field is literally `'SSSSSSSS.mmmuuun'`, plus `'CH'`, `'AMP'`, `'ENER'`,
`'DURATION'`, `'P-FRQ'`, etc. (see `CHID_to_str` in the parser). The backend translates
these to friendly API names via `field_map` / `field_resolve` dicts in
`dta_service.py` — **when you add a feature or column, update those maps in both directions.**

Waveform pretrigger: `TDLY < 0` means pretrigger samples precede the trigger. Most code
trims them by default (`keep_pretrigger=False`); the `_trim_pretrigger` helper is the
canonical implementation.

### Backend (`backend/app/`)
- `main.py` — builds the FastAPI app, wide-open CORS, includes the three routers, loads plugins, and statically mounts `frontend/dist` at `/` when it exists.
- State lives in `dta_service._file_cache`, an **in-memory dict keyed by an 8-char `file_id`**. Uploaded `.DTA` files are parsed once and cached as recarrays; nothing is persisted, so restarting the server drops all loaded files. Uploads land in the OS temp dir (`mistras_uploads/`); exports in `mistras_exports/`.
- Routers: `files` (upload/list/get/delete), `analysis` (the bulk — hits, waveform, FFT, CWT, EMD, filtering, Lamb dispersion, source location, scatter/histogram, exports, and both clustering paths), `plugins`.
- Heavy signal math lives in `services/signal_analysis.py` (pywt, PyEMD, scipy) and `services/deep_clustering.py`. **PyTorch is imported lazily** inside the deep-clustering code so the rest of the backend runs without torch installed.

### Two distinct clustering paths
1. **Parametric** (`compute_clustering`): clusters hits on their scalar features (amplitude, energy, frequency…), then fits a shallow decision tree to produce explainable rules. Fast, interpretable.
2. **Deep latent** (`compute_deep_clustering` and the standalone `tools/ae_deep_cluster.py`): trains a CAE/VAE autoencoder on raw/FFT waveforms, clusters the latent codes (KMeans/GMM/HDBSCAN/DBSCAN), projects to 2D, and emits medoid prototype waveforms. The standalone tool mirrors the backend logic but writes figures + `cluster_labels.csv` to `--out` (default `ae_cluster_out/`) instead of returning JSON.

### Plugin system (`services/plugin_manager.py`)
Drop a `.py` file in `backend/app/plugins/` that subclasses `PluginBase` and exposes a
module-level `register(manager)`. It's auto-discovered at startup and surfaced under
`/api/plugins/...`. `example_neural_operator.py` is a stub showing the interface (intended
for neural-operator / ML models like DeepONet/FNO).

### Frontend (`frontend/src/`)
`App.tsx` is a view switcher driven by `Sidebar`; each analysis area is one component in
`components/` (Dashboard, HitTable, WaveformViewer, WaveletView, ClusterView,
DeepClusterView, SensorView, etc.). All HTTP goes through `services/api.ts`; shared types in
`types/index.ts`. React 19 + Vite, charts via Recharts, 3D via three / @react-three.
In dev, Vite proxies `/api` to `localhost:8000`; in prod the backend serves the built bundle.
