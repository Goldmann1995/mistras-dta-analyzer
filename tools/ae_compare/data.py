"""Data loading, preprocessing, and representations for the AE comparison.

The whole comparison runs on a single :class:`AEDataset` so that every method
(M1..M6) sees *exactly* the same hits, the same denoising, the same length
alignment and the same normalization — only the feature extractor changes.

The dataset exposes four representations, computed lazily:

    waveform     (N, L)         peak-normalized, denoised time-domain trace
    fft          (N, L)         log-magnitude spectrum, resampled to L
    scalogram    (N, S, L)      |CWT| Morlet scalogram (for the 2D / M3 path)
    physical     (N, P)         parametric AE features (amp, energy, RA, AF, ...)

It also carries per-hit ``meta`` (time, channel, sample rate) and the physical
quantities needed for the RA-AF damage-mode plot. A synthetic generator
(:func:`make_synthetic`) produces labelled composite-like AE data so the whole
pipeline can be exercised — and quantitatively validated with ARI/NMI — without
a real ``.DTA`` file.
"""

import os
import sys

import numpy as np

# Make the bundled MistrasDTA package importable regardless of CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# Parametric AE hit features (Mistras recarray field -> readable label).
# Mirrors tools/ae_deep_cluster.py so labels stay consistent across the repo.
FEATURE_FIELDS = [
    ("AMP", "amplitude_dB"), ("ENER", "energy"), ("ABS-ENERGY", "abs_energy"),
    ("RISE", "rise_us"), ("DURATION", "duration_us"), ("COUN", "counts"),
    ("A-FRQ", "avg_freq_kHz"), ("P-FRQ", "peak_freq_kHz"),
    ("FRQ-C", "centroid_freq_kHz"), ("R-FRQ", "rev_freq_kHz"),
    ("I-FRQ", "init_freq_kHz"),
]


# --------------------------------------------------------------------------- #
# Denoising (bandpass + wavelet), copied in spirit from ae_deep_cluster.py
# --------------------------------------------------------------------------- #
def _wavelet_denoise(v, wavelet="db4", level=0, mode="soft"):
    import pywt
    n = len(v)
    if n < 8:
        return v
    v = np.asarray(v, dtype=np.float64)
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(n, w.dec_len)
    lvl = max_level if level <= 0 else min(level, max_level)
    if lvl < 1:
        return v
    coeffs = pywt.wavedec(v, w, mode="periodization", level=lvl)
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail)) / 0.6745 if detail.size else 0.0
    if sigma > 0:
        uthresh = sigma * np.sqrt(2.0 * np.log(n))
        coeffs[1:] = [pywt.threshold(c, uthresh, mode=mode) for c in coeffs[1:]]
    rec = pywt.waverec(coeffs, w, mode="periodization")
    return rec[:n].astype(np.float32)


def _bandpass(v, sr, low_hz, high_hz, order=4):
    from scipy.signal import butter, sosfiltfilt
    n = len(v)
    if n < 3 * (order + 1):
        return v
    nyq = sr / 2.0
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 0.999)
    if low >= high:
        return v
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    pad = min(n - 1, 3 * (sos.shape[0] + 1))
    return sosfiltfilt(sos, v, padlen=pad).astype(np.float32)


def make_denoiser(mode="none", band=(20.0, 400.0), wavelet="db4",
                  level=0, wmode="soft", order=4):
    """Return a ``(v, sr) -> v_clean`` callable, or None for ``mode='none'``."""
    if mode == "none":
        return None
    do_band = "bandpass" in mode
    do_wave = "wavelet" in mode
    low_hz, high_hz = band[0] * 1000.0, band[1] * 1000.0
    if do_wave:
        try:
            import pywt  # noqa: F401
        except ImportError:
            print("  [warn] PyWavelets missing; wavelet denoise disabled")
            do_wave = False
            if not do_band:
                return None

    def denoiser(v, sr):
        if do_band:
            v = _bandpass(v, sr, low_hz, high_hz, order=order)
        if do_wave:
            v = _wavelet_denoise(v, wavelet=wavelet, level=level, mode=wmode)
        return v

    return denoiser


def _round_up(n, m):
    return int(np.ceil(n / m) * m)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class AEDataset:
    """Container for one set of AE hits and their representations.

    Parameters
    ----------
    waves : (N, L) float32      peak-normalized denoised time waveforms
    length : int                L
    sample_rate : float         representative sample rate (Hz)
    meta : list[dict]           per-hit {index, channel, time, sample_rate}
    phys : (N, P) float32 or None   parametric features (NaN-filled if partial)
    phys_names : list[str]
    ra : (N,) / af : (N,)       rise-angle and average-frequency for RA-AF plot
    labels_true : (N,) or None  ground truth (synthetic only)
    """

    def __init__(self, waves, length, sample_rate, meta,
                 phys=None, phys_names=None, ra=None, af=None,
                 labels_true=None):
        self.waves = np.asarray(waves, dtype=np.float32)
        self.length = int(length)
        self.sample_rate = float(sample_rate)
        self.meta = meta
        self.phys = phys
        self.phys_names = phys_names or []
        self.ra = ra
        self.af = af
        self.labels_true = labels_true
        self._fft = None
        self._scalo = None

    def __len__(self):
        return len(self.waves)

    # -- representations ---------------------------------------------------- #
    @property
    def fft(self):
        """Log-magnitude spectra resampled to length L (N, L)."""
        if self._fft is None:
            specs = []
            xq = np.linspace(0, 1, self.length)
            for v in self.waves:
                spec = np.log1p(np.abs(np.fft.rfft(v)))
                xp = np.linspace(0, 1, len(spec))
                rs = np.interp(xq, xp, spec).astype(np.float32)
                m = np.max(np.abs(rs))
                specs.append(rs / m if m > 0 else rs)
            self._fft = np.stack(specs)
        return self._fft

    def scalogram(self, n_scales=32, wavelet="morl"):
        """|CWT| Morlet scalogram (N, n_scales, L), peak-normalized per hit."""
        if self._scalo is not None and self._scalo.shape[1] == n_scales:
            return self._scalo
        import pywt
        scales = np.geomspace(2, self.length / 4, n_scales)
        out = np.empty((len(self.waves), n_scales, self.length), dtype=np.float32)
        for i, v in enumerate(self.waves):
            coef, _ = pywt.cwt(v, scales, wavelet)
            mag = np.abs(coef).astype(np.float32)
            m = mag.max()
            out[i] = mag / m if m > 0 else mag
        self._scalo = out
        return out

    def representation(self, kind, **kw):
        """Return the model input for an encoder.

        ``waveform``/``fft``  -> (N, 1, L)
        ``both``              -> (N, 2, L)   time + spectrum
        ``scalogram``         -> (N, 1, S, L)
        ``physical``          -> (N, P)
        """
        if kind == "waveform":
            return self.waves[:, None, :]
        if kind == "fft":
            return self.fft[:, None, :]
        if kind == "both":
            return np.stack([self.waves, self.fft], axis=1)
        if kind == "scalogram":
            return self.scalogram(**kw)[:, None, :, :]
        if kind == "physical":
            if self.phys is None:
                raise ValueError("no physical features available in this dataset")
            return self.phys
        raise ValueError(f"unknown representation: {kind}")


# --------------------------------------------------------------------------- #
# Real .DTA loading
# --------------------------------------------------------------------------- #
def _phys_row(rec_row):
    if rec_row is None:
        return {}
    names = rec_row.dtype.names
    return {label: float(rec_row[f]) for f, label in FEATURE_FIELDS if f in names}


def _ra_af(feat):
    """RA = rise-time / amplitude  (us / dB);  AF = counts / duration (kHz)."""
    ra = af = np.nan
    rise = feat.get("rise_us")
    amp = feat.get("amplitude_dB")
    cnt = feat.get("counts")
    dur = feat.get("duration_us")
    if rise is not None and amp not in (None, 0):
        ra = rise / amp
    if cnt is not None and dur not in (None, 0):
        af = cnt / dur * 1000.0   # counts/us -> kHz
    return ra, af


def load_dta(path, channel=None, max_hits=5000, fixed_length=0,
             keep_pretrigger=False, denoiser=None):
    """Load AE waveforms + parametric features from a Mistras ``.DTA`` file."""
    from MistrasDTA import read_bin, get_waveform_data

    print(f"[data] reading {path}")
    rec, wfm = read_bin(path)
    if not isinstance(wfm, np.recarray) or len(wfm) == 0:
        raise SystemExit("No waveforms found in this file.")
    print(f"[data] hits={len(rec)}  waveforms={len(wfm)}")

    mask = np.ones(len(wfm), dtype=bool)
    if channel is not None:
        mask &= wfm["CH"] == channel
    idx_all = np.where(mask)[0]
    if len(idx_all) == 0:
        raise SystemExit(f"No waveforms on channel {channel}.")

    if fixed_length <= 0:
        sample = idx_all[np.linspace(0, len(idx_all) - 1,
                                     min(100, len(idx_all))).astype(int)]
        lengths = []
        for i in sample:
            _, V = get_waveform_data(wfm[i])
            if not keep_pretrigger and wfm[i]["TDLY"] < 0:
                V = V[abs(int(wfm[i]["TDLY"])):]
            lengths.append(len(V))
        L = _round_up(max(64, int(np.median(lengths))), 16)
    else:
        L = _round_up(max(64, fixed_length), 16)

    if len(idx_all) > max_hits:
        idx_all = idx_all[np.linspace(0, len(idx_all) - 1, max_hits).astype(int)]

    same_len = isinstance(rec, np.recarray) and len(rec) == len(wfm)
    rec_times = (rec["SSSSSSSS.mmmuuun"]
                 if isinstance(rec, np.recarray) and len(rec)
                 and "SSSSSSSS.mmmuuun" in rec.dtype.names and not same_len
                 else None)

    waves, meta, phys_dicts, ra_list, af_list = [], [], [], [], []
    for i in idx_all:
        row = wfm[i]
        _, V = get_waveform_data(row)
        if not keep_pretrigger and row["TDLY"] < 0:
            V = V[abs(int(row["TDLY"])):]
        if len(V) == 0:
            continue
        if denoiser is not None:
            V = denoiser(V, float(row["SRATE"]))
        v = V[:L] if len(V) >= L else np.pad(V, (0, L - len(V)))
        peak = np.max(np.abs(v))
        if peak > 0:
            v = v / peak
        rec_row = None
        if same_len:
            rec_row = rec[i]
        elif rec_times is not None:
            rec_row = rec[int(np.argmin(np.abs(rec_times - float(row["SSSSSSSS.mmmuuun"]))))]
        feat = _phys_row(rec_row)
        ra, af = _ra_af(feat)
        waves.append(v.astype(np.float32))
        meta.append({"index": int(i), "channel": int(row["CH"]),
                     "time": float(row["SSSSSSSS.mmmuuun"]),
                     "sample_rate": float(row["SRATE"])})
        phys_dicts.append(feat)
        ra_list.append(ra)
        af_list.append(af)

    if len(waves) < 4:
        raise SystemExit("Not enough valid waveforms (need >= 4).")

    waves = np.stack(waves)
    sr = float(np.median([m["sample_rate"] for m in meta]))
    phys, names = _stack_phys(phys_dicts)
    print(f"[data] using {len(waves)} hits, length={L}, "
          f"physical features={len(names)}")
    return AEDataset(waves, L, sr, meta, phys=phys, phys_names=names,
                     ra=np.array(ra_list), af=np.array(af_list))


def _stack_phys(phys_dicts):
    """Build a (N, P) matrix from per-hit feature dicts; mean-fill missing."""
    names = [lbl for _, lbl in FEATURE_FIELDS
             if any(lbl in d for d in phys_dicts)]
    if not names:
        return None, []
    M = np.full((len(phys_dicts), len(names)), np.nan, dtype=np.float32)
    for i, d in enumerate(phys_dicts):
        for j, k in enumerate(names):
            if k in d:
                M[i, j] = d[k]
    col_mean = np.nanmean(M, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    inds = np.where(np.isnan(M))
    M[inds] = np.take(col_mean, inds[1])
    return M, names


# --------------------------------------------------------------------------- #
# Synthetic composite-AE generator (labelled) — backbone of the self-test
# --------------------------------------------------------------------------- #
def make_synthetic(n=600, length=512, sample_rate=2e6, n_classes=4,
                   noise=0.05, seed=0):
    """Generate labelled composite-like AE hits with distinct damage modes.

    Each class is a damped sinusoidal burst with a characteristic centre
    frequency, rise time, amplitude and damping — chosen to mimic the canonical
    CFRP separation (matrix cracking high-freq/short-rise, fiber breakage
    very-high-freq, debonding/delamination low-freq/long-rise). Physical
    parameters (amplitude, energy, rise, duration, counts, peak/centroid freq)
    are computed from the synthesized trace so the RA-AF plot and the M1
    physical-feature path work end to end.
    """
    rng = np.random.default_rng(seed)
    # (centre kHz, rise fraction, rel amplitude, damping)  per damage mode
    modes = [
        (90.0, 0.05, 1.00, 18.0),   # delamination / debonding : low f, long-ish
        (180.0, 0.03, 0.70, 25.0),  # matrix cracking
        (300.0, 0.02, 0.55, 35.0),  # fiber/matrix interface
        (450.0, 0.015, 0.40, 45.0),  # fiber breakage : high f, fast
    ][:n_classes]

    dt = 1.0 / sample_rate
    t = np.arange(length) * dt
    waves, labels, phys_dicts, ra_list, af_list = [], [], [], [], []
    # event times span a "loading history": later classes appear later.
    for i in range(n):
        c = int(rng.integers(n_classes))
        fc_khz, rise_frac, amp_rel, damp = modes[c]
        fc = fc_khz * 1000.0 * rng.normal(1.0, 0.06)
        amp = amp_rel * rng.normal(1.0, 0.12)
        t0 = rise_frac * length * dt * rng.normal(1.0, 0.15)
        env = np.where(t < t0,
                       (t / max(t0, dt)),
                       np.exp(-(t - t0) * damp * fc / fc_khz / 1000.0 * 1000.0))
        env = np.clip(env, 0, None)
        sig = amp * env * np.sin(2 * np.pi * fc * t + rng.uniform(0, 2 * np.pi))
        sig = sig + noise * rng.standard_normal(length) * amp
        v = sig.astype(np.float32)
        peak = np.max(np.abs(v))
        vn = v / peak if peak > 0 else v

        # --- physical params from the trace ---
        amp_db = 20 * np.log10(peak / 1e-3 + 1e-9)  # ref ~1 mV
        thr = 0.1 * peak
        over = np.abs(v) > thr
        counts = int(np.count_nonzero(np.diff((v > thr).astype(int)) > 0))
        idx_over = np.where(over)[0]
        if idx_over.size:
            dur_us = (idx_over[-1] - idx_over[0]) * dt * 1e6
            rise_us = (np.argmax(np.abs(v)) - idx_over[0]) * dt * 1e6
        else:
            dur_us = rise_us = 0.0
        energy = float(np.sum(v ** 2) * dt * 1e6)
        spec = np.abs(np.fft.rfft(v))
        freqs = np.fft.rfftfreq(length, dt) / 1000.0
        peak_f = float(freqs[int(np.argmax(spec))])
        centroid_f = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-9))
        feat = {"amplitude_dB": float(amp_db), "energy": energy,
                "rise_us": float(max(rise_us, 0.01)),
                "duration_us": float(max(dur_us, 0.01)),
                "counts": float(max(counts, 1)),
                "peak_freq_kHz": peak_f, "centroid_freq_kHz": centroid_f,
                "avg_freq_kHz": centroid_f}
        ra, af = _ra_af(feat)
        waves.append(vn)
        labels.append(c)
        phys_dicts.append(feat)
        ra_list.append(ra)
        af_list.append(af)

    waves = np.stack(waves)
    order = np.argsort([rng.random() + lbl for lbl in labels])  # loosely time-ordered by class
    meta = [{"index": int(k), "channel": 1,
             "time": float(rank) / n * 100.0, "sample_rate": sample_rate}
            for rank, k in enumerate(order)]
    # reorder everything by `order` so "time" increases with loading
    waves = waves[order]
    labels = np.array(labels)[order]
    phys_dicts = [phys_dicts[k] for k in order]
    ra_list = np.array(ra_list)[order]
    af_list = np.array(af_list)[order]
    phys, names = _stack_phys(phys_dicts)
    print(f"[data] synthetic: {n} hits, {n_classes} classes, length={length}")
    return AEDataset(waves, length, sample_rate, meta, phys=phys,
                     phys_names=names, ra=ra_list, af=af_list,
                     labels_true=labels)
