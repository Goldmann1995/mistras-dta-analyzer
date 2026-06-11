import numpy as np
import pywt
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
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
