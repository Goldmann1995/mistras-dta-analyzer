import { useState, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { getFilteredWaveform } from '../services/api';
import type { FileInfo, FilterResult } from '../types';

interface Props { file: FileInfo; }

export default function FilterView({ file }: Props) {
  const [index, setIndex] = useState(0);
  const [filterType, setFilterType] = useState<'bandpass' | 'highpass' | 'lowpass'>('bandpass');
  const [freqLow, setFreqLow] = useState(10000);
  const [freqHigh, setFreqHigh] = useState(500000);
  const [order, setOrder] = useState(4);
  const [keepPretrigger, setKeepPretrigger] = useState(false);
  const [result, setResult] = useState<FilterResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [showFFT, setShowFFT] = useState(true);

  const compute = useCallback(async () => {
    setLoading(true);
    try {
      const opts: Record<string, unknown> = {
        filter_type: filterType,
        order,
        keep_pretrigger: keepPretrigger,
      };
      if (filterType === 'bandpass' || filterType === 'highpass') opts.freq_low = freqLow;
      if (filterType === 'bandpass' || filterType === 'lowpass') opts.freq_high = freqHigh;
      const r = await getFilteredWaveform(file.file_id, index, opts as never);
      setResult(r);
    } finally {
      setLoading(false);
    }
  }, [file.file_id, index, filterType, freqLow, freqHigh, order, keepPretrigger]);

  const wfData = result ? result.time_array.map((t, i) => ({
    t: Number(t.toFixed(2)),
    orig: result.original[i],
    filt: result.filtered[i],
  })) : [];

  const fftData = result ? result.fft_frequencies.map((f, i) => ({
    f: Number((f / 1000).toFixed(2)),
    orig: result.fft_original[i],
    filt: result.fft_filtered[i],
  })) : [];

  if (file.waveform_count === 0) return <div className="empty-state">No waveform data in this file</div>;

  return (
    <div className="view-waveform">
      <div className="wf-controls">
        <div className="ctrl-group">
          <label>Index</label>
          <input type="number" min={0} max={file.waveform_count - 1} value={index}
            onChange={e => setIndex(Number(e.target.value))} style={{ width: 70 }} />
          <span className="ctrl-hint">/ {file.waveform_count - 1}</span>
        </div>
        <div className="ctrl-group">
          <label>Filter</label>
          <select value={filterType} onChange={e => setFilterType(e.target.value as typeof filterType)}
            style={{ background: 'var(--bg-1)', border: '1px solid var(--border)', color: 'var(--text-1)', padding: '3px 8px', borderRadius: 4, fontSize: 12 }}>
            <option value="bandpass">Bandpass</option>
            <option value="highpass">Highpass</option>
            <option value="lowpass">Lowpass</option>
          </select>
        </div>
        {(filterType === 'bandpass' || filterType === 'highpass') && (
          <div className="ctrl-group">
            <label>Low (Hz)</label>
            <input type="number" min={1} value={freqLow}
              onChange={e => setFreqLow(Number(e.target.value))} style={{ width: 80 }} />
          </div>
        )}
        {(filterType === 'bandpass' || filterType === 'lowpass') && (
          <div className="ctrl-group">
            <label>High (Hz)</label>
            <input type="number" min={1} value={freqHigh}
              onChange={e => setFreqHigh(Number(e.target.value))} style={{ width: 80 }} />
          </div>
        )}
        <div className="ctrl-group">
          <label>Order</label>
          <input type="number" min={1} max={10} value={order}
            onChange={e => setOrder(Number(e.target.value))} style={{ width: 50 }} />
        </div>
        <label className="ctrl-toggle">
          <input type="checkbox" checked={keepPretrigger} onChange={e => setKeepPretrigger(e.target.checked)} />
          <span>Pre-trigger</span>
        </label>
        <button className="ctrl-btn" onClick={compute} disabled={loading}>
          {loading ? 'Filtering...' : 'Apply'}
        </button>
        <button className={`ctrl-btn ${showFFT ? 'active' : ''}`} onClick={() => setShowFFT(!showFFT)}>FFT</button>
      </div>

      {result && (
        <div className="wf-meta">
          <span>CH{result.channel}</span>
          <span>{(result.sample_rate / 1e3).toFixed(0)} kHz</span>
          <span>Filter: {result.filter_type}</span>
          {result.freq_low && <span>Low: {(result.freq_low / 1000).toFixed(1)} kHz</span>}
          {result.freq_high && <span>High: {(result.freq_high / 1000).toFixed(1)} kHz</span>}
          <span>Order: {result.order}</span>
        </div>
      )}

      {result && (
        <>
          <div className="panel">
            <div className="panel-head">
              Waveform Comparison
              <span className="panel-tag">
                <span style={{ color: '#64748b' }}>Original</span> vs <span style={{ color: '#22d3ee' }}>Filtered</span>
              </span>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={wfData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="t" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false}
                  label={{ value: 'Time (us)', fill: '#64748b', fontSize: 11, position: 'insideBottom', offset: -4 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false}
                  label={{ value: 'V', fill: '#64748b', fontSize: 11, angle: -90, position: 'insideLeft' }} />
                <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }} />
                <Line type="monotone" dataKey="orig" stroke="#64748b" dot={false} strokeWidth={1} opacity={0.5} name="Original" />
                <Line type="monotone" dataKey="filt" stroke="#22d3ee" dot={false} strokeWidth={1.5} name="Filtered" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {showFFT && (
            <div className="panel">
              <div className="panel-head">
                FFT Comparison
                <span className="panel-tag">
                  <span style={{ color: '#64748b' }}>Original</span> vs <span style={{ color: '#a78bfa' }}>Filtered</span>
                </span>
              </div>
              <ResponsiveContainer width="100%" height={240}>
                <AreaChart data={fftData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="f" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false}
                    label={{ value: 'Freq (kHz)', fill: '#64748b', fontSize: 11, position: 'insideBottom', offset: -4 }} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} />
                  <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }} />
                  <Area type="monotone" dataKey="orig" stroke="#64748b" fill="#64748b" fillOpacity={0.1} name="Original" />
                  <Area type="monotone" dataKey="filt" stroke="#a78bfa" fill="#a78bfa" fillOpacity={0.2} name="Filtered" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}

      {!result && !loading && (
        <div className="panel" style={{ textAlign: 'center', padding: 40 }}>
          <p style={{ color: 'var(--text-2)', margin: 0 }}>
            Configure filter parameters and click <strong>Apply</strong> to see the filtered waveform
          </p>
        </div>
      )}
    </div>
  );
}
