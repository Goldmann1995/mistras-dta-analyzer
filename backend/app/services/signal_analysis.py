import numpy as np
import pywt
from PyEMD import EMD, EEMD
from scipy.signal import hilbert, butter, sosfilt, sosfiltfilt
from scipy.fft import fft, fftfreq
from scipy.optimize import brentq, minimize
from scipy.stats import skew, kurtosis
from typing import Optional

from MistrasDTA import get_waveform_data


def apply_filter(
    wfm_row,
    filter_type: str = 'bandpass',
    freq_low: Optional[float] = None,
    freq_high: Optional[float] = None,
    order: int = 4,
    keep_pretrigger: bool = False,
) -> dict:
    t, V = get_waveform_data(wfm_row)
    sr = float(wfm_row['SRATE'])

    if not keep_pretrigger and wfm_row['TDLY'] < 0:
        trim = abs(int(wfm_row['TDLY']))
        t = t[trim:] - t[trim]
        V = V[trim:]

    nyq = sr / 2.0
    V_original = V.copy()

    if filter_type == 'bandpass':
        if freq_low is None or freq_high is None:
            raise ValueError("bandpass requires freq_low and freq_high")
        sos = butter(order, [freq_low / nyq, freq_high / nyq], btype='bandpass', output='sos')
    elif filter_type == 'lowpass':
        if freq_high is None:
            raise ValueError("lowpass requires freq_high")
        sos = butter(order, freq_high / nyq, btype='low', output='sos')
    elif filter_type == 'highpass':
        if freq_low is None:
            raise ValueError("highpass requires freq_low")
        sos = butter(order, freq_low / nyq, btype='high', output='sos')
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    V_filtered = sosfiltfilt(sos, V).astype(np.float64)

    step = max(1, len(t) // 5000)
    t_down = t[::step]
    V_orig_down = V_original[::step]
    V_filt_down = V_filtered[::step]

    N = len(V_filtered)
    yf_orig = fft(V_original)
    yf_filt = fft(V_filtered)
    xf = fftfreq(N, 1.0 / sr)
    pos = xf > 0
    freqs = xf[pos]
    mag_orig = (2.0 / N * np.abs(yf_orig[pos]))
    mag_filt = (2.0 / N * np.abs(yf_filt[pos]))

    f_step = max(1, len(freqs) // 2000)
    freqs_down = freqs[::f_step]
    mag_orig_down = mag_orig[::f_step]
    mag_filt_down = mag_filt[::f_step]

    return {
        'time_array': t_down.tolist(),
        'original': V_orig_down.tolist(),
        'filtered': V_filt_down.tolist(),
        'fft_frequencies': freqs_down.tolist(),
        'fft_original': mag_orig_down.tolist(),
        'fft_filtered': mag_filt_down.tolist(),
        'filter_type': filter_type,
        'freq_low': freq_low,
        'freq_high': freq_high,
        'order': order,
        'channel': int(wfm_row['CH']),
        'sample_rate': sr,
    }


def compute_cwt(
    wfm_row,
    wavelet: str = 'morl',
    freq_min: float = 1000,
    freq_max: Optional[float] = None,
    num_freqs: int = 128,
    keep_pretrigger: bool = False,
) -> dict:
    t, V = get_waveform_data(wfm_row)
    sr = float(wfm_row['SRATE'])

    if not keep_pretrigger and wfm_row['TDLY'] < 0:
        trim = abs(int(wfm_row['TDLY']))
        t = t[trim:] - t[trim]
        V = V[trim:]

    if freq_max is None:
        freq_max = sr / 2.0 * 0.95

    freqs = np.linspace(freq_min, freq_max, num_freqs)
    scales = pywt.central_frequency(wavelet) * sr / freqs

    coeffs, _ = pywt.cwt(V, scales, wavelet, sampling_period=1.0 / sr)
    power = np.abs(coeffs) ** 2

    max_t_points = 500
    max_f_points = 128
    t_step = max(1, len(t) // max_t_points)
    f_step = max(1, len(freqs) // max_f_points)

    t_down = t[::t_step]
    freqs_down = freqs[::f_step]
    power_down = power[::f_step, ::t_step]

    p_max = np.max(power_down)
    if p_max > 0:
        power_norm = power_down / p_max
    else:
        power_norm = power_down

    peak_freq_idx = np.unravel_index(np.argmax(power), power.shape)
    peak_frequency = float(freqs[peak_freq_idx[0]])
    peak_time = float(t[peak_freq_idx[1]])

    return {
        'time_axis': t_down.tolist(),
        'freq_axis': freqs_down.tolist(),
        'power': power_norm.tolist(),
        'peak_frequency': peak_frequency,
        'peak_time': peak_time,
        'wavelet': wavelet,
        'channel': int(wfm_row['CH']),
        'sample_rate': sr,
    }


def compute_group_velocity_dispersion(
    wfm_row,
    wavelet: str = 'morl',
    freq_min: float = 1000,
    freq_max: Optional[float] = None,
    num_freqs: int = 64,
    keep_pretrigger: bool = False,
) -> dict:
    t, V = get_waveform_data(wfm_row)
    sr = float(wfm_row['SRATE'])

    if not keep_pretrigger and wfm_row['TDLY'] < 0:
        trim = abs(int(wfm_row['TDLY']))
        t = t[trim:] - t[trim]
        V = V[trim:]

    if freq_max is None:
        freq_max = sr / 2.0 * 0.95

    freqs = np.linspace(freq_min, freq_max, num_freqs)
    scales = pywt.central_frequency(wavelet) * sr / freqs

    coeffs, _ = pywt.cwt(V, scales, wavelet, sampling_period=1.0 / sr)
    power = np.abs(coeffs) ** 2

    arrival_times = []
    energy_at_freq = []
    for i in range(len(freqs)):
        row_power = power[i, :]
        if np.max(row_power) > 0:
            threshold = np.max(row_power) * 0.1
            above = np.where(row_power > threshold)[0]
            if len(above) > 0:
                arrival_times.append(float(t[above[0]]))
            else:
                arrival_times.append(float(t[np.argmax(row_power)]))
        else:
            arrival_times.append(0.0)
        energy_at_freq.append(float(np.sum(row_power)))

    peak_times = []
    for i in range(len(freqs)):
        peak_times.append(float(t[np.argmax(power[i, :])]))

    return {
        'frequencies': freqs.tolist(),
        'arrival_times': arrival_times,
        'peak_times': peak_times,
        'energy_at_freq': energy_at_freq,
        'channel': int(wfm_row['CH']),
        'sample_rate': sr,
    }


def compute_cross_channel_velocity(
    wfm_data,
    rec_data,
    sensor_distance: float,
    keep_pretrigger: bool = False,
) -> dict:
    channels = sorted(set(int(c) for c in wfm_data['CH']))
    if len(channels) < 2:
        return {'error': 'Need at least 2 channels', 'pairs': []}

    ch_events: dict[int, list] = {ch: [] for ch in channels}
    for i, row in enumerate(wfm_data):
        ch = int(row['CH'])
        ch_events[ch].append(i)

    time_field = 'SSSSSSSS.mmmuuun'
    ch_times: dict[int, list] = {ch: [] for ch in channels}
    for ch in channels:
        mask = rec_data['CH'] == ch
        ch_times[ch] = rec_data[mask][time_field].tolist()

    pairs = []
    for i, ch1 in enumerate(channels):
        for ch2 in channels[i + 1:]:
            times1 = np.array(ch_times[ch1])
            times2 = np.array(ch_times[ch2])

            dt_list = []
            velocity_list = []
            event_times = []

            for t1 in times1:
                diffs = np.abs(times2 - t1)
                min_idx = np.argmin(diffs)
                if diffs[min_idx] < 0.01:
                    dt = abs(t1 - times2[min_idx])
                    if dt > 1e-8:
                        vel = sensor_distance / dt
                        dt_list.append(float(dt))
                        velocity_list.append(float(vel))
                        event_times.append(float(t1))

            if velocity_list:
                pairs.append({
                    'ch1': ch1,
                    'ch2': ch2,
                    'event_count': len(velocity_list),
                    'event_times': event_times,
                    'delta_t': dt_list,
                    'velocities': velocity_list,
                    'avg_velocity': float(np.mean(velocity_list)),
                    'std_velocity': float(np.std(velocity_list)),
                    'median_velocity': float(np.median(velocity_list)),
                })

    return {
        'sensor_distance': sensor_distance,
        'channel_count': len(channels),
        'channels': channels,
        'pairs': pairs,
    }


def compute_emd(
    wfm_row,
    method: str = 'emd',
    max_imfs: int = 8,
    keep_pretrigger: bool = False,
) -> dict:
    t, V = get_waveform_data(wfm_row)
    sr = float(wfm_row['SRATE'])

    if not keep_pretrigger and wfm_row['TDLY'] < 0:
        trim = abs(int(wfm_row['TDLY']))
        t = t[trim:] - t[trim]
        V = V[trim:]

    if method == 'eemd':
        decomposer = EEMD()
        decomposer.eemd(V)
        imfs = decomposer.get_imfs_and_residue()[0]
    else:
        decomposer = EMD()
        imfs = decomposer.emd(V)

    if len(imfs) > max_imfs:
        imfs = imfs[:max_imfs]

    max_points = 2000
    step = max(1, len(t) // max_points)
    t_down = t[::step]

    imf_list = []
    for i, imf in enumerate(imfs):
        imf_down = imf[::step]
        analytic = hilbert(imf)
        inst_amp = np.abs(analytic)
        inst_phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(inst_phase) / (2 * np.pi / sr)
        inst_freq = np.append(inst_freq, inst_freq[-1])
        inst_freq = np.clip(inst_freq, 0, sr / 2)

        freq_spectrum = np.abs(fft(imf))[:len(imf) // 2]
        freq_axis = fftfreq(len(imf), 1.0 / sr)[:len(imf) // 2]
        dominant_freq = float(freq_axis[np.argmax(freq_spectrum)]) if len(freq_spectrum) > 0 else 0.0
        energy = float(np.sum(imf ** 2))

        imf_list.append({
            'index': i,
            'data': imf_down.tolist(),
            'inst_amplitude': inst_amp[::step].tolist(),
            'inst_frequency': inst_freq[::step].tolist(),
            'dominant_frequency': dominant_freq,
            'energy': energy,
            'energy_ratio': 0.0,
        })

    total_energy = sum(m['energy'] for m in imf_list)
    if total_energy > 0:
        for m in imf_list:
            m['energy_ratio'] = m['energy'] / total_energy

    return {
        'time_axis': t_down.tolist(),
        'num_imfs': len(imf_list),
        'imfs': imf_list,
        'method': method,
        'channel': int(wfm_row['CH']),
        'sample_rate': sr,
    }


def _lamb_det_vec(omega, vp_arr, cl, ct, h, mode='S'):
    """Rayleigh-Lamb frequency equation (Giurgiutiu form), element-wise.

    D_S = (2k²-kT²)²·cos(αh)·sin(βh) + 4k²αβ·sin(αh)·cos(βh) = 0
    D_A = 4k²αβ·cos(αh)·sin(βh) + (2k²-kT²)²·sin(αh)·cos(βh) = 0

    Split into three velocity regimes using real arithmetic only,
    extracting the real or imaginary part of the complex determinant
    depending on which is non-trivial in each regime.
    """
    k = omega / vp_arr
    k2 = k ** 2
    kL2 = (omega / cl) ** 2
    kT2 = (omega / ct) ** 2
    a2 = kL2 - k2
    b2 = kT2 - k2
    bracket = (2 * k2 - kT2) ** 2

    result = np.zeros_like(vp_arr, dtype=float)

    r1 = (a2 >= 0) & (b2 >= 0)
    r2 = (a2 < 0) & (b2 >= 0)
    r3 = (a2 < 0) & (b2 < 0)

    # Regime 1: vp > cL, both α,β real → determinant is real
    if np.any(r1):
        a = np.sqrt(a2[r1]); b = np.sqrt(b2[r1])
        ca, sa = np.cos(a * h), np.sin(a * h)
        cb, sb = np.cos(b * h), np.sin(b * h)
        br = bracket[r1]; kk = k2[r1]
        if mode == 'S':
            result[r1] = br * ca * sb + 4 * kk * a * b * sa * cb
        else:
            result[r1] = 4 * kk * a * b * ca * sb + br * sa * cb

    # Regime 2: cT < vp < cL, α imaginary, β real
    #   S determinant is real; A determinant is imaginary (extract imag part)
    if np.any(r2):
        ap = np.sqrt(-a2[r2]); b = np.sqrt(b2[r2])
        cha, sha = np.cosh(ap * h), np.sinh(ap * h)
        cb, sb = np.cos(b * h), np.sin(b * h)
        br = bracket[r2]; kk = k2[r2]
        if mode == 'S':
            result[r2] = br * cha * sb - 4 * kk * ap * b * sha * cb
        else:
            result[r2] = br * sha * cb + 4 * kk * ap * b * cha * sb

    # Regime 3: vp < cT, both α,β imaginary → determinant is imaginary
    if np.any(r3):
        ap = np.sqrt(-a2[r3]); bp = np.sqrt(-b2[r3])
        cha, sha = np.cosh(ap * h), np.sinh(ap * h)
        chb, shb = np.cosh(bp * h), np.sinh(bp * h)
        br = bracket[r3]; kk = k2[r3]
        if mode == 'S':
            result[r3] = br * cha * shb - 4 * kk * ap * bp * sha * chb
        else:
            result[r3] = br * sha * chb - 4 * kk * ap * bp * cha * shb

    return result


def compute_lamb_dispersion(
    thickness: float,
    cl: float,
    ct: float,
    freq_min: float = 1000,
    freq_max: float = 500000,
    num_points: int = 200,
    max_modes: int = 4,
) -> dict:
    h = thickness / 2.0
    freqs = np.linspace(freq_min, freq_max, num_points)
    vp_arr = np.concatenate([
        np.linspace(50.0, ct, 2000),
        np.linspace(ct * 1.001, cl * 3.0, 2000),
    ])

    modes = {'symmetric': [], 'antisymmetric': []}

    for mode_type, label in [('S', 'symmetric'), ('A', 'antisymmetric')]:
        all_mode_freqs: list[list[float]] = [[] for _ in range(max_modes)]
        all_mode_vp: list[list[float]] = [[] for _ in range(max_modes)]

        for freq in freqs:
            if freq <= 0:
                continue
            omega = 2 * np.pi * freq

            det_vals = _lamb_det_vec(omega, vp_arr, cl, ct, h, mode_type)
            raw_sc = np.where(np.diff(np.sign(det_vals)))[0]

            sign_changes = []
            for sc in raw_sc:
                vp_mid = (vp_arr[sc] + vp_arr[sc + 1]) / 2
                if abs(vp_mid - cl) < cl * 0.02 or abs(vp_mid - ct) < ct * 0.02:
                    continue
                if abs(det_vals[sc]) > 1e15 or abs(det_vals[sc + 1]) > 1e15:
                    continue
                sign_changes.append(sc)

            for mode_n in range(min(max_modes, len(sign_changes))):
                sc = sign_changes[mode_n]
                vp_low, vp_high = vp_arr[sc], vp_arr[sc + 1]
                try:
                    vp_root = brentq(
                        lambda vp: float(np.real(
                            _lamb_det_vec(omega, np.array([vp]), cl, ct, h, mode_type)[0]
                        )),
                        float(vp_low), float(vp_high), xtol=0.1
                    )
                    all_mode_freqs[mode_n].append(float(freq))
                    all_mode_vp[mode_n].append(float(vp_root))
                except Exception:
                    pass

        for mode_n in range(max_modes):
            m_freqs = all_mode_freqs[mode_n]
            m_phase_vel = all_mode_vp[mode_n]
            m_group_vel: list[float] = []

            if len(m_freqs) > 2:
                m_omega = np.array(m_freqs) * 2 * np.pi
                m_k = m_omega / np.array(m_phase_vel)
                m_gv = np.gradient(m_omega) / (np.gradient(m_k) + 1e-30)
                m_gv = np.clip(m_gv, 0, cl * 2)
                m_group_vel = m_gv.tolist()

            if m_freqs:
                modes[label].append({
                    'mode': f'{mode_type}{mode_n}',
                    'frequencies': m_freqs,
                    'phase_velocity': m_phase_vel,
                    'group_velocity': m_group_vel,
                })

    return {
        'thickness': thickness,
        'cl': cl,
        'ct': ct,
        'freq_range': [freq_min, freq_max],
        'modes': modes,
    }


def compute_source_locations(
    rec_data,
    sensor_positions: dict[int, list[float]],
    velocity: float = 5400.0,
    time_window: float = 0.001,
) -> dict:
    """Locate AE sources using TDOA with least-squares minimization.

    Groups hits within time_window into events, then for events with >= 3
    channel arrivals, solves for the source position that minimizes the
    residual of (measured_dt - predicted_dt) across all sensor pairs.
    """
    channels = sorted(sensor_positions.keys())
    if len(channels) < 2:
        return {'events': [], 'sensor_positions': sensor_positions}

    time_field = 'SSSSSSSS.mmmuuun'
    ch_hits: dict[int, list] = {ch: [] for ch in channels}
    for i, row in enumerate(rec_data):
        ch = int(row['CH'])
        if ch in ch_hits:
            ch_hits[ch].append({
                'index': i,
                'time': float(row[time_field]),
                'amplitude': float(row['AMP']) if 'AMP' in rec_data.dtype.names else 0,
                'energy': float(row['ENER']) if 'ENER' in rec_data.dtype.names else 0,
            })

    events = []
    used = set()
    all_hits = []
    for ch in channels:
        for h in ch_hits[ch]:
            all_hits.append((h['time'], ch, h))
    all_hits.sort()

    i = 0
    while i < len(all_hits):
        t0 = all_hits[i][0]
        group: dict[int, dict] = {}
        j = i
        while j < len(all_hits) and all_hits[j][0] - t0 <= time_window:
            ch = all_hits[j][1]
            hit = all_hits[j][2]
            hid = (ch, hit['index'])
            if ch not in group and hid not in used:
                group[ch] = hit
                used.add(hid)
            j += 1
        i = j if j > i else i + 1

        if len(group) < 2:
            continue

        event_chs = sorted(group.keys())
        arrivals = {ch: group[ch]['time'] for ch in event_chs}
        first_ch = min(arrivals, key=lambda c: arrivals[c])
        avg_amp = np.mean([group[c]['amplitude'] for c in event_chs])
        avg_energy = np.mean([group[c]['energy'] for c in event_chs])
        event_time = arrivals[first_ch]

        location = None
        if len(event_chs) >= 2:
            pos = np.array([sensor_positions[ch] for ch in event_chs])
            t_arr = np.array([arrivals[ch] for ch in event_chs])
            t_rel = t_arr - t_arr.min()

            x0 = np.mean(pos, axis=0)

            if len(event_chs) == 2:
                dt = t_rel[1] - t_rel[0]
                d = dt * velocity
                mid = (pos[0] + pos[1]) / 2
                direction = pos[1] - pos[0]
                length = np.linalg.norm(direction)
                if length > 0:
                    direction = direction / length
                    location = (mid - direction * d / 2).tolist()
            else:
                def residual(src):
                    dists = np.sqrt(np.sum((pos - src) ** 2, axis=1))
                    predicted_t = dists / velocity
                    predicted_dt = predicted_t - predicted_t.min()
                    return np.sum((predicted_dt - t_rel) ** 2)

                opt = minimize(residual, x0, method='Nelder-Mead',
                               options={'xatol': 1e-5, 'fatol': 1e-12, 'maxiter': 500})
                if opt.success or opt.fun < 1e-6:
                    location = opt.x.tolist()

        events.append({
            'time': event_time,
            'channels': event_chs,
            'arrivals': arrivals,
            'amplitude': float(avg_amp),
            'energy': float(avg_energy),
            'location': location,
            'num_channels': len(event_chs),
        })

    located = [e for e in events if e['location'] is not None]

    return {
        'total_events': len(events),
        'located_events': len(located),
        'events': events,
        'sensor_positions': {str(k): v for k, v in sensor_positions.items()},
        'velocity': velocity,
    }


def compute_waveform_entropy(V: np.ndarray) -> float:
    """Compute Shannon information entropy of a waveform's voltage distribution.

    Uses Scott's rule with skewness/kurtosis correction for optimal bin width.
    """
    n = len(V)
    if n < 2:
        return 0.0

    sigma = np.std(V)
    if sigma == 0:
        return 0.0

    b_n = 3.49 * sigma * n ** (-1.0 / 3.0)

    sk = skew(V)
    kurt_val = kurtosis(V, fisher=True)

    sk2 = sk ** 2
    c_sk = np.sqrt(1.0 + 2.0 * sk2) if sk2 > 0 else 1.0

    c_kur = (1.0 + (kurt_val / 4.0)) ** (-0.2) if kurt_val > -4.0 else 1.0

    b_opt = b_n * c_sk * c_kur
    if b_opt <= 0:
        b_opt = b_n if b_n > 0 else 1.0

    v_min, v_max = np.min(V), np.max(V)
    v_range = v_max - v_min
    if v_range == 0:
        return 0.0

    num_bins = max(1, int(np.ceil(v_range / b_opt)))

    hist, _ = np.histogram(V, bins=num_bins)
    hist = hist[hist > 0]
    P = hist / n

    entropy = -np.sum(P * np.log(P))
    return float(entropy)


def compute_all_entropies(wfm_data, keep_pretrigger: bool = False) -> np.ndarray:
    """Compute waveform entropy for every waveform row."""
    entropies = np.empty(len(wfm_data), dtype=np.float64)
    for i in range(len(wfm_data)):
        row = wfm_data[i]
        t, V = get_waveform_data(row)
        if not keep_pretrigger and row['TDLY'] < 0:
            trim = abs(int(row['TDLY']))
            V = V[trim:]
        entropies[i] = compute_waveform_entropy(V)
    return entropies


def compute_clustering(
    rec_data,
    features: list[str],
    algorithm: str = 'kmeans',
    n_clusters: int = 3,
    eps: float = 0.5,
    min_samples: int = 5,
    max_tree_depth: int = 5,
    channel: Optional[int] = None,
    extra_features: Optional[dict[str, np.ndarray]] = None,
) -> dict:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.mixture import GaussianMixture
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    field_resolve = {
        'time': 'SSSSSSSS.mmmuuun',
        'amplitude': 'AMP', 'energy': 'ENER', 'duration': 'DURATION',
        'rise': 'RISE', 'counts': 'COUN', 'peak_counts': 'PCNTS',
        'rms': 'RMS', 'asl': 'ASL', 'avg_frequency': 'A-FRQ',
        'rev_frequency': 'R-FRQ', 'init_frequency': 'I-FRQ',
        'abs_energy': 'ABS-ENERGY', 'signal_strength': 'SIG STRENGTH',
        'peak_frequency': 'P-FRQ', 'freq_centroid': 'FRQ-C',
    }

    if len(rec_data) == 0:
        raise ValueError("No data to cluster")

    mask = np.ones(len(rec_data), dtype=bool)
    if channel is not None:
        mask &= rec_data['CH'] == channel
    filtered = rec_data[mask]

    if len(filtered) < n_clusters:
        raise ValueError(f"Not enough data points ({len(filtered)}) for {n_clusters} clusters")

    if extra_features is None:
        extra_features = {}

    resolved_features = []
    feature_labels = []
    extra_columns = []
    for f in features:
        if f in extra_features:
            feature_labels.append(f)
            resolved_features.append(None)
            extra_columns.append(extra_features[f][mask].astype(float))
        else:
            rf = field_resolve.get(f, f)
            if rf in filtered.dtype.names:
                resolved_features.append(rf)
                feature_labels.append(f)
                extra_columns.append(None)

    if len(feature_labels) < 2:
        raise ValueError("Need at least 2 valid features for clustering")

    columns = []
    for i, rf in enumerate(resolved_features):
        if rf is not None:
            columns.append(filtered[rf].astype(float))
        else:
            columns.append(extra_columns[i])
    X_raw = np.column_stack(columns)

    valid_mask = np.all(np.isfinite(X_raw), axis=1)
    X_raw = X_raw[valid_mask]
    filtered_indices = np.where(mask)[0][valid_mask]

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    if algorithm == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X)
    elif algorithm == 'dbscan':
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        n_clusters = len(set(labels) - {-1})
    elif algorithm == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=42, n_init=3)
        labels = model.fit_predict(X)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    unique_labels = sorted(set(labels))
    valid_labels = [l for l in unique_labels if l >= 0]

    metrics = {}
    if len(valid_labels) >= 2:
        non_noise = labels >= 0
        if np.sum(non_noise) > len(valid_labels):
            metrics['silhouette'] = float(silhouette_score(X[non_noise], labels[non_noise]))
            metrics['calinski_harabasz'] = float(calinski_harabasz_score(X[non_noise], labels[non_noise]))
            metrics['davies_bouldin'] = float(davies_bouldin_score(X[non_noise], labels[non_noise]))

    tree_rules = []
    tree_feature_importance = []
    tree_accuracy = 0.0

    non_noise_mask = labels >= 0
    if np.sum(non_noise_mask) > 10 and len(valid_labels) >= 2:
        dt = DecisionTreeClassifier(
            max_depth=max_tree_depth,
            min_samples_leaf=max(5, len(X) // 100),
            random_state=42,
        )
        dt.fit(X_raw[non_noise_mask], labels[non_noise_mask])
        tree_accuracy = float(dt.score(X_raw[non_noise_mask], labels[non_noise_mask]))

        importances = dt.feature_importances_
        tree_feature_importance = [
            {'feature': feature_labels[i], 'importance': float(importances[i])}
            for i in range(len(feature_labels))
        ]
        tree_feature_importance.sort(key=lambda x: x['importance'], reverse=True)

        tree_rules = _extract_tree_rules(dt, feature_labels, valid_labels)

    cluster_stats = []
    for label in valid_labels:
        cmask = labels == label
        count = int(np.sum(cmask))
        stats: dict = {'label': int(label), 'count': count, 'percentage': round(100.0 * count / len(labels), 1)}
        for i, fname in enumerate(feature_labels):
            vals = X_raw[cmask, i]
            stats[fname] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'median': float(np.median(vals)),
            }
        cluster_stats.append(stats)

    noise_count = int(np.sum(labels == -1))

    scatter_x_idx = 0
    scatter_y_idx = 1 if len(feature_labels) > 1 else 0
    scatter_data = []
    step = max(1, len(X_raw) // 5000)
    for i in range(0, len(X_raw), step):
        scatter_data.append({
            'x': float(X_raw[i, scatter_x_idx]),
            'y': float(X_raw[i, scatter_y_idx]),
            'cluster': int(labels[i]),
            'index': int(filtered_indices[i]),
        })

    return {
        'algorithm': algorithm,
        'n_clusters': len(valid_labels),
        'total_points': int(len(labels)),
        'noise_points': noise_count,
        'features': feature_labels,
        'scatter_x': feature_labels[scatter_x_idx],
        'scatter_y': feature_labels[scatter_y_idx],
        'scatter_data': scatter_data,
        'cluster_stats': cluster_stats,
        'metrics': metrics,
        'tree_accuracy': tree_accuracy,
        'tree_rules': tree_rules,
        'tree_feature_importance': tree_feature_importance,
    }


def _extract_tree_rules(
    tree,
    feature_names: list[str],
    class_labels: list[int],
) -> list[dict]:
    from sklearn.tree import _tree

    tree_ = tree.tree_
    rules = []

    def recurse(node, path):
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            values = tree_.value[node][0]
            predicted = int(class_labels[int(np.argmax(values))])
            total = int(np.sum(values))
            confidence = float(np.max(values) / total) if total > 0 else 0
            rules.append({
                'conditions': list(path),
                'cluster': predicted,
                'samples': total,
                'confidence': round(confidence, 3),
                'rule_text': _format_rule(path, predicted, confidence),
            })
            return

        fname = feature_names[tree_.feature[node]]
        threshold = float(tree_.threshold[node])

        recurse(tree_.children_left[node],
                path + [{'feature': fname, 'op': '<=', 'value': round(threshold, 4)}])
        recurse(tree_.children_right[node],
                path + [{'feature': fname, 'op': '>', 'value': round(threshold, 4)}])

    recurse(0, [])

    rules.sort(key=lambda r: r['samples'], reverse=True)
    return rules


def _format_rule(conditions: list[dict], cluster: int, confidence: float) -> str:
    if not conditions:
        return f"→ Cluster {cluster} (conf: {confidence:.1%})"
    parts = [f"{c['feature']} {c['op']} {c['value']}" for c in conditions]
    return f"IF {' AND '.join(parts)} → Cluster {cluster} (conf: {confidence:.1%})"
