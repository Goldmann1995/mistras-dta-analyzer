import sys
import os
import uuid
import tempfile
import csv
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.io import savemat
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from MistrasDTA import read_bin, get_waveform_data


def _trim_pretrigger(t, V, tdly):
    if tdly >= 0:
        return t, V
    trim_count = abs(int(tdly))
    return t[trim_count:] - t[trim_count], V[trim_count:]


_file_cache: dict[str, dict] = {}


def load_dta_file(filepath: str, filename: str) -> str:
    file_id = str(uuid.uuid4())[:8]
    rec, wfm = read_bin(filepath)

    rec_list = rec if isinstance(rec, np.recarray) else []
    wfm_list = wfm if isinstance(wfm, np.recarray) else []

    _file_cache[file_id] = {
        'filepath': filepath,
        'filename': filename,
        'rec': rec_list,
        'wfm': wfm_list,
    }
    return file_id


def get_file_info(file_id: str) -> dict:
    data = _file_cache[file_id]
    rec = data['rec']
    wfm = data['wfm']

    channels = sorted(set(int(c) for c in rec['CH'])) if len(rec) > 0 else []
    duration = float(rec['SSSSSSSS.mmmuuun'][-1] - rec['SSSSSSSS.mmmuuun'][0]) if len(rec) > 1 else 0.0
    fields = list(rec.dtype.names) if len(rec) > 0 else []

    return {
        'filename': data['filename'],
        'file_id': file_id,
        'hit_count': len(rec),
        'waveform_count': len(wfm),
        'channels': channels,
        'duration': duration,
        'fields': fields,
    }


def get_hits(
    file_id: str,
    channel: Optional[int] = None,
    offset: int = 0,
    limit: int = 100,
    sort_by: Optional[str] = None,
    sort_order: str = 'asc',
    amp_min: Optional[int] = None,
    amp_max: Optional[int] = None,
    time_min: Optional[float] = None,
    time_max: Optional[float] = None,
) -> dict:
    rec = _file_cache[file_id]['rec']
    if len(rec) == 0:
        return {'total': 0, 'hits': []}

    mask = np.ones(len(rec), dtype=bool)
    if channel is not None:
        mask &= rec['CH'] == channel
    if amp_min is not None and 'AMP' in rec.dtype.names:
        mask &= rec['AMP'] >= amp_min
    if amp_max is not None and 'AMP' in rec.dtype.names:
        mask &= rec['AMP'] <= amp_max
    if time_min is not None:
        mask &= rec['SSSSSSSS.mmmuuun'] >= time_min
    if time_max is not None:
        mask &= rec['SSSSSSSS.mmmuuun'] <= time_max

    filtered = rec[mask]
    total = len(filtered)

    if sort_by and sort_by in filtered.dtype.names:
        indices = np.argsort(filtered[sort_by])
        if sort_order == 'desc':
            indices = indices[::-1]
        filtered = filtered[indices]

    page = filtered[offset:offset + limit]

    field_map = {
        'SSSSSSSS.mmmuuun': 'time',
        'CH': 'channel',
        'RISE': 'rise',
        'PCNTS': 'peak_counts',
        'COUN': 'counts',
        'ENER': 'energy',
        'DURATION': 'duration',
        'AMP': 'amplitude',
        'ASL': 'asl',
        'THR': 'threshold',
        'A-FRQ': 'avg_frequency',
        'RMS': 'rms',
        'R-FRQ': 'rev_frequency',
        'I-FRQ': 'init_frequency',
        'SIG STRENGTH': 'signal_strength',
        'ABS-ENERGY': 'abs_energy',
        'FRQ-C': 'freq_centroid',
        'P-FRQ': 'peak_frequency',
        'TIMESTAMP': 'timestamp',
    }

    hits = []
    for i, row in enumerate(page):
        hit = {'index': offset + i}
        for fname in rec.dtype.names:
            key = field_map.get(fname, fname)
            val = row[fname]
            hit[key] = float(val) if isinstance(val, (np.floating, float)) else int(val)
        hits.append(hit)

    return {'total': total, 'hits': hits}


def get_waveform(file_id: str, index: int, keep_pretrigger: bool = False) -> dict:
    wfm = _file_cache[file_id]['wfm']
    if index >= len(wfm):
        raise IndexError(f"Waveform index {index} out of range")

    row = wfm[index]
    t, V = get_waveform_data(row)

    pretrigger_samples = abs(int(row['TDLY'])) if row['TDLY'] < 0 else 0
    if not keep_pretrigger:
        t, V = _trim_pretrigger(t, V, int(row['TDLY']))

    step = max(1, len(t) // 5000)
    t_down = t[::step]
    V_down = V[::step]

    return {
        'index': index,
        'time': float(row['SSSSSSSS.mmmuuun']),
        'channel': int(row['CH']),
        'sample_rate': float(row['SRATE']),
        'pretrigger_samples': pretrigger_samples,
        'trimmed': not keep_pretrigger and pretrigger_samples > 0,
        'time_array': t_down.tolist(),
        'voltage_array': V_down.tolist(),
    }


def get_waveform_fft(file_id: str, index: int, keep_pretrigger: bool = False) -> dict:
    wfm = _file_cache[file_id]['wfm']
    row = wfm[index]
    t, V = get_waveform_data(row)

    if not keep_pretrigger:
        _, V = _trim_pretrigger(t, V, int(row['TDLY']))

    N = len(V)
    sr = float(row['SRATE'])
    yf = fft(V)
    xf = fftfreq(N, 1.0 / sr)

    positive = xf > 0
    freqs = xf[positive]
    mags = 2.0 / N * np.abs(yf[positive])

    step = max(1, len(freqs) // 2000)
    freqs_down = freqs[::step]
    mags_down = mags[::step]

    dominant_idx = np.argmax(mags)

    return {
        'frequencies': freqs_down.tolist(),
        'magnitudes': mags_down.tolist(),
        'dominant_frequency': float(freqs[dominant_idx]),
        'sample_rate': sr,
    }


def get_channel_stats(file_id: str) -> list[dict]:
    rec = _file_cache[file_id]['rec']
    if len(rec) == 0:
        return []

    channels = sorted(set(int(c) for c in rec['CH']))
    stats = []
    for ch in channels:
        mask = rec['CH'] == ch
        ch_data = rec[mask]
        s = {
            'channel': ch,
            'hit_count': len(ch_data),
            'avg_amplitude': float(np.mean(ch_data['AMP'])) if 'AMP' in ch_data.dtype.names else 0,
            'max_amplitude': float(np.max(ch_data['AMP'])) if 'AMP' in ch_data.dtype.names else 0,
            'min_amplitude': float(np.min(ch_data['AMP'])) if 'AMP' in ch_data.dtype.names else 0,
            'avg_energy': float(np.mean(ch_data['ENER'])) if 'ENER' in ch_data.dtype.names else 0,
            'max_energy': float(np.max(ch_data['ENER'])) if 'ENER' in ch_data.dtype.names else 0,
            'avg_duration': float(np.mean(ch_data['DURATION'])) if 'DURATION' in ch_data.dtype.names else 0,
            'max_duration': float(np.max(ch_data['DURATION'])) if 'DURATION' in ch_data.dtype.names else 0,
            'time_span': float(ch_data['SSSSSSSS.mmmuuun'][-1] - ch_data['SSSSSSSS.mmmuuun'][0]) if len(ch_data) > 1 else 0,
        }
        if 'RMS' in ch_data.dtype.names:
            s['avg_rms'] = float(np.mean(ch_data['RMS']))
        stats.append(s)
    return stats


def get_scatter_data(
    file_id: str,
    x_field: str,
    y_field: str,
    color_field: Optional[str] = None,
    channel: Optional[int] = None,
    max_points: int = 5000,
) -> dict:
    rec = _file_cache[file_id]['rec']
    if len(rec) == 0:
        return {'x': [], 'y': [], 'color': None, 'x_field': x_field, 'y_field': y_field}

    mask = np.ones(len(rec), dtype=bool)
    if channel is not None:
        mask &= rec['CH'] == channel

    filtered = rec[mask]

    field_resolve = {
        'time': 'SSSSSSSS.mmmuuun',
        'channel': 'CH', 'amplitude': 'AMP', 'energy': 'ENER',
        'duration': 'DURATION', 'rise': 'RISE', 'counts': 'COUN',
        'peak_counts': 'PCNTS', 'rms': 'RMS', 'asl': 'ASL',
        'avg_frequency': 'A-FRQ', 'abs_energy': 'ABS-ENERGY',
        'signal_strength': 'SIG STRENGTH', 'peak_frequency': 'P-FRQ',
        'freq_centroid': 'FRQ-C', 'timestamp': 'TIMESTAMP',
    }

    xf = field_resolve.get(x_field, x_field)
    yf = field_resolve.get(y_field, y_field)

    step = max(1, len(filtered) // max_points)
    sampled = filtered[::step]

    indices = np.where(mask)[0][::step]

    result = {
        'x': [float(v) for v in sampled[xf]],
        'y': [float(v) for v in sampled[yf]],
        'indices': [int(v) for v in indices],
        'color': None,
        'x_field': x_field,
        'y_field': y_field,
    }

    if color_field:
        cf = field_resolve.get(color_field, color_field)
        if cf in sampled.dtype.names:
            result['color'] = [float(v) for v in sampled[cf]]
            result['color_field'] = color_field

    return result


def get_histogram_data(
    file_id: str,
    field: str,
    bins: int = 50,
    channel: Optional[int] = None,
) -> dict:
    rec = _file_cache[file_id]['rec']
    if len(rec) == 0:
        return {'edges': [], 'counts': [], 'field': field}

    field_resolve = {
        'amplitude': 'AMP', 'energy': 'ENER', 'duration': 'DURATION',
        'rise': 'RISE', 'counts': 'COUN', 'rms': 'RMS',
        'abs_energy': 'ABS-ENERGY', 'signal_strength': 'SIG STRENGTH',
        'peak_frequency': 'P-FRQ', 'freq_centroid': 'FRQ-C',
        'avg_frequency': 'A-FRQ',
    }
    resolved = field_resolve.get(field, field)

    mask = np.ones(len(rec), dtype=bool)
    if channel is not None:
        mask &= rec['CH'] == channel

    values = rec[mask][resolved].astype(float)
    hist_counts, edges = np.histogram(values, bins=bins)

    return {
        'edges': edges.tolist(),
        'counts': hist_counts.tolist(),
        'field': field,
    }


def get_loaded_files() -> list[dict]:
    return [get_file_info(fid) for fid in _file_cache]


def remove_file(file_id: str):
    _file_cache.pop(file_id, None)


def export_npz(
    file_id: str,
    channel: Optional[int] = None,
    keep_pretrigger: bool = False,
    max_waveforms: Optional[int] = None,
    normalize: bool = False,
    fixed_length: Optional[int] = None,
) -> str:
    data = _file_cache[file_id]
    rec = data['rec']
    wfm = data['wfm']

    if len(rec) == 0 or len(wfm) == 0:
        raise ValueError("No data to export")

    mask = np.ones(len(wfm), dtype=bool)
    if channel is not None:
        mask &= wfm['CH'] == channel
    wfm_filtered = wfm[mask]

    rec_mask = np.ones(len(rec), dtype=bool)
    if channel is not None:
        rec_mask &= rec['CH'] == channel
    rec_filtered = rec[rec_mask]

    if max_waveforms and len(wfm_filtered) > max_waveforms:
        wfm_filtered = wfm_filtered[:max_waveforms]

    waveforms = []
    meta_times = []
    meta_channels = []
    meta_amplitudes = []
    meta_energies = []
    meta_durations = []
    meta_sample_rates = []

    for row in wfm_filtered:
        t, V = get_waveform_data(row)
        if not keep_pretrigger:
            t, V = _trim_pretrigger(t, V, int(row['TDLY']))
        waveforms.append(V)
        meta_times.append(float(row['SSSSSSSS.mmmuuun']))
        meta_channels.append(int(row['CH']))
        meta_sample_rates.append(float(row['SRATE']))

    if 'AMP' in rec_filtered.dtype.names:
        meta_amplitudes = rec_filtered['AMP'][:len(waveforms)].astype(float).tolist()
    if 'ENER' in rec_filtered.dtype.names:
        meta_energies = rec_filtered['ENER'][:len(waveforms)].astype(float).tolist()
    if 'DURATION' in rec_filtered.dtype.names:
        meta_durations = rec_filtered['DURATION'][:len(waveforms)].astype(float).tolist()

    if fixed_length:
        padded = []
        for w in waveforms:
            if len(w) >= fixed_length:
                padded.append(w[:fixed_length])
            else:
                padded.append(np.pad(w, (0, fixed_length - len(w))))
        waveform_array = np.array(padded, dtype=np.float32)
    else:
        target_len = max(len(w) for w in waveforms)
        padded = [np.pad(w, (0, target_len - len(w))) if len(w) < target_len else w for w in waveforms]
        waveform_array = np.array(padded, dtype=np.float32)

    if normalize:
        global_max = np.max(np.abs(waveform_array))
        if global_max > 0:
            waveform_array = waveform_array / global_max

    export_dir = os.path.join(tempfile.gettempdir(), "mistras_exports")
    os.makedirs(export_dir, exist_ok=True)
    filename = f"{data['filename'].replace('.DTA', '').replace('.dta', '')}"
    if channel is not None:
        filename += f"_ch{channel}"
    if not keep_pretrigger:
        filename += "_trimmed"
    if normalize:
        filename += "_norm"
    filename += ".npz"
    filepath = os.path.join(export_dir, filename)

    np.savez_compressed(
        filepath,
        waveforms=waveform_array,
        times=np.array(meta_times, dtype=np.float64),
        channels=np.array(meta_channels, dtype=np.int32),
        sample_rates=np.array(meta_sample_rates, dtype=np.float64),
        amplitudes=np.array(meta_amplitudes, dtype=np.float32) if meta_amplitudes else np.array([]),
        energies=np.array(meta_energies, dtype=np.float32) if meta_energies else np.array([]),
        durations=np.array(meta_durations, dtype=np.float32) if meta_durations else np.array([]),
    )

    return filepath


def export_mat(
    file_id: str,
    channel: Optional[int] = None,
    keep_pretrigger: bool = False,
    max_waveforms: Optional[int] = None,
    normalize: bool = False,
    fixed_length: Optional[int] = None,
) -> str:
    data = _file_cache[file_id]
    rec = data['rec']
    wfm = data['wfm']

    if len(rec) == 0 or len(wfm) == 0:
        raise ValueError("No data to export")

    mask = np.ones(len(wfm), dtype=bool)
    if channel is not None:
        mask &= wfm['CH'] == channel
    wfm_filtered = wfm[mask]

    rec_mask = np.ones(len(rec), dtype=bool)
    if channel is not None:
        rec_mask &= rec['CH'] == channel
    rec_filtered = rec[rec_mask]

    if max_waveforms and len(wfm_filtered) > max_waveforms:
        wfm_filtered = wfm_filtered[:max_waveforms]

    waveforms = []
    meta_times = []
    meta_channels = []
    meta_amplitudes = []
    meta_energies = []
    meta_durations = []
    meta_sample_rates = []

    for row in wfm_filtered:
        t, V = get_waveform_data(row)
        if not keep_pretrigger:
            t, V = _trim_pretrigger(t, V, int(row['TDLY']))
        waveforms.append(V)
        meta_times.append(float(row['SSSSSSSS.mmmuuun']))
        meta_channels.append(int(row['CH']))
        meta_sample_rates.append(float(row['SRATE']))

    if 'AMP' in rec_filtered.dtype.names:
        meta_amplitudes = rec_filtered['AMP'][:len(waveforms)].astype(float).tolist()
    if 'ENER' in rec_filtered.dtype.names:
        meta_energies = rec_filtered['ENER'][:len(waveforms)].astype(float).tolist()
    if 'DURATION' in rec_filtered.dtype.names:
        meta_durations = rec_filtered['DURATION'][:len(waveforms)].astype(float).tolist()

    if fixed_length:
        padded = []
        for w in waveforms:
            if len(w) >= fixed_length:
                padded.append(w[:fixed_length])
            else:
                padded.append(np.pad(w, (0, fixed_length - len(w))))
        waveform_array = np.array(padded, dtype=np.float64)
    else:
        target_len = max(len(w) for w in waveforms)
        padded = [np.pad(w, (0, target_len - len(w))) if len(w) < target_len else w for w in waveforms]
        waveform_array = np.array(padded, dtype=np.float64)

    if normalize:
        global_max = np.max(np.abs(waveform_array))
        if global_max > 0:
            waveform_array = waveform_array / global_max

    export_dir = os.path.join(tempfile.gettempdir(), "mistras_exports")
    os.makedirs(export_dir, exist_ok=True)
    filename = f"{data['filename'].replace('.DTA', '').replace('.dta', '')}"
    if channel is not None:
        filename += f"_ch{channel}"
    if not keep_pretrigger:
        filename += "_trimmed"
    if normalize:
        filename += "_norm"
    filename += ".mat"
    filepath = os.path.join(export_dir, filename)

    mat_dict = {
        'waveforms': waveform_array,
        'times': np.array(meta_times, dtype=np.float64),
        'channels': np.array(meta_channels, dtype=np.int32),
        'sample_rates': np.array(meta_sample_rates, dtype=np.float64),
    }
    if meta_amplitudes:
        mat_dict['amplitudes'] = np.array(meta_amplitudes, dtype=np.float64)
    if meta_energies:
        mat_dict['energies'] = np.array(meta_energies, dtype=np.float64)
    if meta_durations:
        mat_dict['durations'] = np.array(meta_durations, dtype=np.float64)

    savemat(filepath, mat_dict, do_compression=True)
    return filepath


def export_csv(
    file_id: str,
    channel: Optional[int] = None,
    include_waveforms: bool = False,
) -> str:
    data = _file_cache[file_id]
    rec = data['rec']

    if len(rec) == 0:
        raise ValueError("No data to export")

    mask = np.ones(len(rec), dtype=bool)
    if channel is not None:
        mask &= rec['CH'] == channel
    filtered = rec[mask]

    field_map = {
        'SSSSSSSS.mmmuuun': 'Time_s',
        'CH': 'Channel',
        'RISE': 'Rise_us',
        'PCNTS': 'Peak_Counts',
        'COUN': 'Counts',
        'ENER': 'Energy',
        'DURATION': 'Duration_us',
        'AMP': 'Amplitude_dB',
        'ASL': 'ASL',
        'THR': 'Threshold',
        'A-FRQ': 'Avg_Frequency_kHz',
        'RMS': 'RMS',
        'R-FRQ': 'Rev_Frequency_kHz',
        'I-FRQ': 'Init_Frequency_kHz',
        'SIG STRENGTH': 'Signal_Strength',
        'ABS-ENERGY': 'Abs_Energy',
        'FRQ-C': 'Freq_Centroid_kHz',
        'P-FRQ': 'Peak_Frequency_kHz',
    }

    export_dir = os.path.join(tempfile.gettempdir(), "mistras_exports")
    os.makedirs(export_dir, exist_ok=True)
    filename = f"{data['filename'].replace('.DTA', '').replace('.dta', '')}"
    if channel is not None:
        filename += f"_ch{channel}"
    filename += "_hits.csv"
    filepath = os.path.join(export_dir, filename)

    headers = []
    field_names = []
    for fname in rec.dtype.names:
        if fname in field_map:
            headers.append(field_map[fname])
            field_names.append(fname)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in filtered:
            writer.writerow([
                float(row[fn]) if isinstance(row[fn], (np.floating, float)) else int(row[fn])
                for fn in field_names
            ])

    return filepath
