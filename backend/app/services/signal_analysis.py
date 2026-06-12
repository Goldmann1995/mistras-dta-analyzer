import numpy as np
import pywt
from PyEMD import EMD, EEMD
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
from scipy.optimize import brentq
from typing import Optional

from MistrasDTA import get_waveform_data


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
