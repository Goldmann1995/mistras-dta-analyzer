import { useState, useEffect, useCallback } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, ZAxis, LineChart, Line, Cell,
} from 'recharts';
import { getScatterData, getHistogramData, getWaveform, getWaveformFFT } from '../services/api';
import type { FileInfo, WaveformData, FFTResult } from '../types';

interface Props { file: FileInfo; }

const FIELDS = [
  { v: 'time', l: 'Time' }, { v: 'amplitude', l: 'Amplitude' }, { v: 'energy', l: 'Energy' },
  { v: 'duration', l: 'Duration' }, { v: 'rise', l: 'Rise' }, { v: 'counts', l: 'Counts' },
  { v: 'rms', l: 'RMS' }, { v: 'avg_frequency', l: 'Avg Freq' }, { v: 'peak_frequency', l: 'Peak Freq' },
  { v: 'abs_energy', l: 'Abs Energy' }, { v: 'signal_strength', l: 'Sig Strength' }, { v: 'freq_centroid', l: 'Freq Centroid' },
];

export default function ParametricCharts({ file }: Props) {
  const [xF, setXF] = useState('time');
  const [yF, setYF] = useState('amplitude');
  const [cF, setCF] = useState('energy');
  const [hF, setHF] = useState('amplitude');
  const [ch, setCh] = useState<number | undefined>();
  const [scatter, setScatter] = useState<{ x: number; y: number; z: number; idx: number }[]>([]);
  const [hist, setHist] = useState<{ v: string; n: number }[]>([]);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [wf, setWf] = useState<WaveformData | null>(null);
  const [fft, setFft] = useState<FFTResult | null>(null);
  const [wfLoading, setWfLoading] = useState(false);

  useEffect(() => {
    getScatterData(file.file_id, xF, yF, cF, ch).then(d => {
      setScatter(d.x.map((x, i) => ({ x, y: d.y[i], z: d.color?.[i] ?? 1, idx: d.indices[i] })));
    });
  }, [file.file_id, xF, yF, cF, ch]);

  useEffect(() => {
    getHistogramData(file.file_id, hF, 40, ch).then(d => {
      setHist(d.counts.map((c, i) => ({ v: ((d.edges[i] + d.edges[i + 1]) / 2).toFixed(1), n: c })));
    });
  }, [file.file_id, hF, ch]);

  const loadWaveform = useCallback(async (hitIndex: number) => {
    setSelectedIdx(hitIndex);
    setWfLoading(true);
    try {
      const [w, f] = await Promise.all([
        getWaveform(file.file_id, hitIndex),
        getWaveformFFT(file.file_id, hitIndex),
      ]);
      setWf(w);
      setFft(f);
    } finally {
      setWfLoading(false);
    }
  }, [file.file_id]);

  const handleScatterClick = (data: { payload?: { idx?: number } }) => {
    const idx = data?.payload?.idx;
    if (idx !== undefined && file.waveform_count > 0) {
      loadWaveform(idx);
    }
  };

  const fl = (v: string) => FIELDS.find(f => f.v === v)?.l ?? v;

  const wfData = wf ? wf.time_array.map((t, i) => ({ t: Number(t.toFixed(1)), v: wf.voltage_array[i] })) : [];
  const fftData = fft ? fft.frequencies.map((f, i) => ({ f: Number((f / 1000).toFixed(2)), m: fft.magnitudes[i] })).filter(d => d.f > 0) : [];

  return (
    <div className="view-charts">
      <div className="charts-controls">
        <label className="filter-label">Channel</label>
        <select className="filter-select" value={ch ?? ''} onChange={e => setCh(e.target.value ? Number(e.target.value) : undefined)}>
          <option value="">All</option>
          {file.channels.map(c => <option key={c} value={c}>CH{c}</option>)}
        </select>
        {file.waveform_count > 0 && <span className="ctrl-hint" style={{ marginLeft: 8 }}>Click scatter point to view waveform</span>}
      </div>

      <div className="panel-grid-2">
        <div className="panel">
          <div className="panel-head">
            Scatter Plot
            <div className="panel-selectors">
              <select value={xF} onChange={e => setXF(e.target.value)}>{FIELDS.map(f => <option key={f.v} value={f.v}>{f.l}</option>)}</select>
              <span>vs</span>
              <select value={yF} onChange={e => setYF(e.target.value)}>{FIELDS.map(f => <option key={f.v} value={f.v}>{f.l}</option>)}</select>
              <span>color</span>
              <select value={cF} onChange={e => setCF(e.target.value)}>{FIELDS.map(f => <option key={f.v} value={f.v}>{f.l}</option>)}</select>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={340}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="x" name={fl(xF)} tick={{ fill: '#94a3b8', fontSize: 10 }} type="number" axisLine={false} />
              <YAxis dataKey="y" name={fl(yF)} tick={{ fill: '#94a3b8', fontSize: 10 }} type="number" axisLine={false} />
              <ZAxis dataKey="z" range={[15, 150]} />
              <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }} formatter={(v: number) => v.toFixed(3)} />
              <Scatter data={scatter} fillOpacity={0.5} onClick={handleScatterClick} style={{ cursor: 'pointer' }}>
                {scatter.map((entry, i) => (
                  <Cell key={i} fill={entry.idx === selectedIdx ? '#f87171' : '#22d3ee'} fillOpacity={entry.idx === selectedIdx ? 1 : 0.5} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div className="panel">
          <div className="panel-head">
            Histogram
            <div className="panel-selectors">
              <select value={hF} onChange={e => setHF(e.target.value)}>{FIELDS.filter(f => f.v !== 'time').map(f => <option key={f.v} value={f.v}>{f.l}</option>)}</select>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={hist} barSize={6}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="v" tick={{ fill: '#94a3b8', fontSize: 10 }} interval="preserveStartEnd" axisLine={false} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} />
              <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }} />
              <Bar dataKey="n" fill="#a78bfa" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {selectedIdx !== null && (
        <div className="panel-grid-2">
          <div className="panel">
            <div className="panel-head">
              Waveform #{selectedIdx}
              {wf && (
                <span className="panel-tag">CH{wf.channel} · {wf.sample_rate / 1000} kHz SR</span>
              )}
              <button className="ctrl-btn" style={{ marginLeft: 'auto' }} onClick={() => { setSelectedIdx(null); setWf(null); setFft(null); }}>Close</button>
            </div>
            {wfLoading ? (
              <div className="loading-indicator">Loading waveform...</div>
            ) : wf ? (
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={wfData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="t" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false}
                    label={{ value: 'Time (μs)', fill: '#64748b', fontSize: 11, position: 'insideBottom', offset: -4 }} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false}
                    label={{ value: 'Voltage', fill: '#64748b', fontSize: 11, angle: -90, position: 'insideLeft' }} />
                  <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }} />
                  <Line type="monotone" dataKey="v" stroke="#22d3ee" dot={false} strokeWidth={1} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="loading-indicator">No waveform data</div>
            )}
          </div>

          <div className="panel">
            <div className="panel-head">
              FFT
              {fft && <span className="panel-tag">Peak: {(fft.dominant_frequency / 1000).toFixed(1)} kHz</span>}
            </div>
            {wfLoading ? (
              <div className="loading-indicator">Loading FFT...</div>
            ) : fft ? (
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={fftData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="f" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false}
                    label={{ value: 'Frequency (kHz)', fill: '#64748b', fontSize: 11, position: 'insideBottom', offset: -4 }} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false}
                    label={{ value: 'Magnitude', fill: '#64748b', fontSize: 11, angle: -90, position: 'insideLeft' }} />
                  <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }} />
                  <Line type="monotone" dataKey="m" stroke="#a78bfa" dot={false} strokeWidth={1} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="loading-indicator">No FFT data</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
