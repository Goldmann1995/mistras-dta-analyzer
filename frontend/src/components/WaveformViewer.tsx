import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { getWaveform, getWaveformFFT } from '../services/api';
import type { FileInfo, WaveformData, FFTResult } from '../types';

interface Props {
  file: FileInfo;
  initialIndex?: number;
}

const WAVE_COLORS = ['#22d3ee', '#a78bfa', '#34d399', '#fb923c', '#f472b6', '#facc15'];

export default function WaveformViewer({ file, initialIndex }: Props) {
  const [wf, setWf] = useState<WaveformData | null>(null);
  const [fft, setFft] = useState<FFTResult | null>(null);
  const [index, setIndex] = useState(initialIndex ?? 0);
  const [showFFT, setShowFFT] = useState(false);
  const [keepPretrigger, setKeepPretrigger] = useState(false);
  const [overlayN, setOverlayN] = useState(1);
  const [overlays, setOverlays] = useState<WaveformData[]>([]);

  useEffect(() => {
    if (file.waveform_count === 0) return;
    getWaveform(file.file_id, index, keepPretrigger).then(setWf);
  }, [file.file_id, file.waveform_count, index, keepPretrigger]);

  useEffect(() => {
    if (!showFFT || file.waveform_count === 0) return;
    getWaveformFFT(file.file_id, index, keepPretrigger).then(setFft);
  }, [file.file_id, file.waveform_count, index, showFFT, keepPretrigger]);

  useEffect(() => {
    if (overlayN <= 1 || file.waveform_count === 0) { setOverlays([]); return; }
    const n = Math.min(overlayN, file.waveform_count, 20);
    Promise.all(Array.from({ length: n }, (_, i) => getWaveform(file.file_id, i, keepPretrigger))).then(setOverlays);
  }, [file.file_id, file.waveform_count, overlayN, keepPretrigger]);

  if (file.waveform_count === 0) return <div className="empty-state">No waveform data in this file</div>;

  const wfData = wf ? wf.time_array.map((t, i) => ({ t: Number(t.toFixed(2)), v: wf.voltage_array[i] })) : [];
  const fftData = fft ? fft.frequencies.map((f, i) => ({ f: Number((f / 1000).toFixed(2)), m: fft.magnitudes[i] })) : [];
  const overlayData = overlays.length > 0
    ? overlays[0].time_array.map((t, ti) => {
        const pt: Record<string, number> = { t: Number(t.toFixed(2)) };
        overlays.forEach((w, wi) => { pt[`v${wi}`] = w.voltage_array[ti] ?? 0; });
        return pt;
      }) : [];

  return (
    <div className="view-waveform">
      <div className="wf-controls">
        <div className="ctrl-group">
          <label>Index</label>
          <input type="number" min={0} max={file.waveform_count - 1} value={index} onChange={e => setIndex(Number(e.target.value))} />
          <span className="ctrl-hint">/ {file.waveform_count - 1}</span>
        </div>
        <div className="ctrl-group">
          <label>Overlay</label>
          <input type="number" min={1} max={Math.min(20, file.waveform_count)} value={overlayN} onChange={e => setOverlayN(Number(e.target.value))} />
        </div>
        <label className="ctrl-toggle">
          <input type="checkbox" checked={keepPretrigger} onChange={e => setKeepPretrigger(e.target.checked)} />
          <span>Pre-trigger</span>
        </label>
        <button className={`ctrl-btn ${showFFT ? 'active' : ''}`} onClick={() => setShowFFT(!showFFT)}>FFT</button>
      </div>

      {wf && (
        <div className="wf-meta">
          <span>CH{wf.channel}</span>
          <span>{(wf.sample_rate / 1e3).toFixed(0)} kHz</span>
          <span>t = {wf.time.toFixed(6)} s</span>
          <span>{wf.time_array.length} pts</span>
          {wf.trimmed && <span className="tag-trim">trimmed</span>}
          {!wf.trimmed && wf.pretrigger_samples > 0 && <span className="tag-pre">pre-trigger: {wf.pretrigger_samples} pts</span>}
        </div>
      )}

      <div className="panel">
        <div className="panel-head">{overlayN > 1 ? `Overlay (first ${overlayN})` : `Waveform #${index}`}</div>
        <ResponsiveContainer width="100%" height={300}>
          {overlayN > 1 && overlayData.length > 0 ? (
            <LineChart data={overlayData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="t" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} label={{ value: 'Time (us)', fill: '#64748b', fontSize: 11, position: 'insideBottom', offset: -4 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} label={{ value: 'V', fill: '#64748b', fontSize: 11, angle: -90, position: 'insideLeft' }} />
              <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }} />
              {overlays.map((_, i) => <Line key={i} type="monotone" dataKey={`v${i}`} stroke={WAVE_COLORS[i % WAVE_COLORS.length]} dot={false} strokeWidth={1} opacity={0.6} />)}
            </LineChart>
          ) : (
            <LineChart data={wfData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="t" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} label={{ value: 'Time (us)', fill: '#64748b', fontSize: 11, position: 'insideBottom', offset: -4 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} label={{ value: 'V', fill: '#64748b', fontSize: 11, angle: -90, position: 'insideLeft' }} />
              <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }} />
              <Line type="monotone" dataKey="v" stroke="#22d3ee" dot={false} strokeWidth={1.5} />
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>

      {showFFT && fft && (
        <div className="panel">
          <div className="panel-head">
            FFT Spectrum
            <span className="panel-tag">Dominant: {(fft.dominant_frequency / 1000).toFixed(1)} kHz</span>
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <AreaChart data={fftData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="f" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} label={{ value: 'Freq (kHz)', fill: '#64748b', fontSize: 11, position: 'insideBottom', offset: -4 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} />
              <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }} />
              <Area type="monotone" dataKey="m" stroke="#a78bfa" fill="#a78bfa" fillOpacity={0.2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
