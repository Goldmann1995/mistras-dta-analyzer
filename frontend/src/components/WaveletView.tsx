import { useState, useEffect, useRef, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ScatterChart, Scatter, BarChart, Bar, Cell, ZAxis,
} from 'recharts';
import { getCWT, getDispersion, getGroupVelocity, getWaveform } from '../services/api';
import type { FileInfo, CWTResult, DispersionResult, GroupVelocityResult, WaveformData } from '../types';

interface Props { file: FileInfo; }

const COLORMAPS = [
  [0, '#060a13'], [0.15, '#0c1a4a'], [0.3, '#1a237e'],
  [0.45, '#4a148c'], [0.6, '#b71c1c'], [0.75, '#e65100'],
  [0.9, '#f9a825'], [1.0, '#ffffff'],
];

function interpolateColor(value: number): string {
  const v = Math.max(0, Math.min(1, value));
  let i = 0;
  while (i < COLORMAPS.length - 1 && COLORMAPS[i + 1][0] <= v) i++;
  if (i >= COLORMAPS.length - 1) return COLORMAPS[COLORMAPS.length - 1][1] as string;

  const [t0, c0] = COLORMAPS[i];
  const [t1, c1] = COLORMAPS[i + 1];
  const frac = ((v as number) - (t0 as number)) / ((t1 as number) - (t0 as number));

  const parse = (hex: string) => [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
  const rgb0 = parse(c0 as string);
  const rgb1 = parse(c1 as string);
  const r = Math.round(rgb0[0] + (rgb1[0] - rgb0[0]) * frac);
  const g = Math.round(rgb0[1] + (rgb1[1] - rgb0[1]) * frac);
  const b = Math.round(rgb0[2] + (rgb1[2] - rgb0[2]) * frac);
  return `rgb(${r},${g},${b})`;
}

function SpectrogramCanvas({ cwt }: { cwt: CWTResult }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !cwt) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const nFreqs = cwt.power.length;
    const nTimes = cwt.power[0]?.length ?? 0;
    if (nFreqs === 0 || nTimes === 0) return;

    const w = canvas.width;
    const h = canvas.height;
    const margin = { top: 10, right: 10, bottom: 40, left: 60 };
    const plotW = w - margin.left - margin.right;
    const plotH = h - margin.top - margin.bottom;

    ctx.fillStyle = '#060a13';
    ctx.fillRect(0, 0, w, h);

    const imgData = ctx.createImageData(nTimes, nFreqs);
    for (let fi = 0; fi < nFreqs; fi++) {
      for (let ti = 0; ti < nTimes; ti++) {
        const val = cwt.power[nFreqs - 1 - fi][ti];
        const color = interpolateColor(val);
        const match = color.match(/\d+/g);
        if (match) {
          const idx = (fi * nTimes + ti) * 4;
          imgData.data[idx] = parseInt(match[0]);
          imgData.data[idx + 1] = parseInt(match[1]);
          imgData.data[idx + 2] = parseInt(match[2]);
          imgData.data[idx + 3] = 255;
        }
      }
    }

    const offscreen = new OffscreenCanvas(nTimes, nFreqs);
    const offCtx = offscreen.getContext('2d');
    if (offCtx) {
      offCtx.putImageData(imgData, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(offscreen, margin.left, margin.top, plotW, plotH);
    }

    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.strokeRect(margin.left, margin.top, plotW, plotH);

    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px "IBM Plex Mono", monospace';
    ctx.textAlign = 'center';
    const tAxis = cwt.time_axis;
    for (let i = 0; i <= 5; i++) {
      const ti = Math.floor(i / 5 * (tAxis.length - 1));
      const x = margin.left + (i / 5) * plotW;
      ctx.fillText(`${tAxis[ti].toFixed(0)}`, x, h - margin.bottom + 14);
    }
    ctx.fillText('Time (μs)', margin.left + plotW / 2, h - 5);

    ctx.textAlign = 'right';
    const fAxis = cwt.freq_axis;
    for (let i = 0; i <= 5; i++) {
      const fi = Math.floor(i / 5 * (fAxis.length - 1));
      const y = margin.top + plotH - (i / 5) * plotH;
      ctx.fillText(`${(fAxis[fi] / 1000).toFixed(1)}`, margin.left - 6, y + 3);
    }
    ctx.save();
    ctx.translate(12, margin.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Freq (kHz)', 0, 0);
    ctx.restore();

    const pkX = margin.left + ((cwt.peak_time - tAxis[0]) / (tAxis[tAxis.length - 1] - tAxis[0])) * plotW;
    const pkY = margin.top + plotH - ((cwt.peak_frequency - fAxis[0]) / (fAxis[fAxis.length - 1] - fAxis[0])) * plotH;
    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(pkX, pkY, 6, 0, Math.PI * 2);
    ctx.stroke();

  }, [cwt]);

  return <canvas ref={canvasRef} width={1200} height={500} style={{ width: '100%', height: 500, borderRadius: 4 }} />;
}

export default function WaveletView({ file }: Props) {
  const [index, setIndex] = useState(0);
  const [wavelet, setWavelet] = useState('morl');
  const [numFreqs, setNumFreqs] = useState(128);
  const [keepPre, setKeepPre] = useState(false);
  const [cwt, setCwt] = useState<CWTResult | null>(null);
  const [disp, setDisp] = useState<DispersionResult | null>(null);
  const [wf, setWf] = useState<WaveformData | null>(null);
  const [gv, setGv] = useState<GroupVelocityResult | null>(null);
  const [sensorDist, setSensorDist] = useState(0.3);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState<'cwt' | 'velocity'>('cwt');

  const loadCWT = useCallback(async () => {
    if (file.waveform_count === 0) return;
    setLoading(true);
    try {
      const [c, d, w] = await Promise.all([
        getCWT(file.file_id, index, { wavelet, num_freqs: numFreqs, keep_pretrigger: keepPre }),
        getDispersion(file.file_id, index, { wavelet, num_freqs: 48, keep_pretrigger: keepPre }),
        getWaveform(file.file_id, index, keepPre),
      ]);
      setCwt(c);
      setDisp(d);
      setWf(w);
    } finally {
      setLoading(false);
    }
  }, [file.file_id, file.waveform_count, index, wavelet, numFreqs, keepPre]);

  useEffect(() => { loadCWT(); }, [loadCWT]);

  const loadGV = useCallback(async () => {
    setLoading(true);
    try {
      const r = await getGroupVelocity(file.file_id, sensorDist, keepPre);
      setGv(r);
    } finally {
      setLoading(false);
    }
  }, [file.file_id, sensorDist, keepPre]);

  if (file.waveform_count === 0) return <div className="empty-state">No waveform data</div>;

  const dispData = disp
    ? disp.frequencies.map((f, i) => ({
        freq: Number((f / 1000).toFixed(2)),
        arrival: disp.arrival_times[i],
        peak: disp.peak_times[i],
        energy: disp.energy_at_freq[i],
      }))
    : [];

  const wfData = wf ? wf.time_array.map((t, i) => ({ t: Number(t.toFixed(1)), v: wf.voltage_array[i] })) : [];

  return (
    <div className="view-wavelet">
      <div className="wavelet-tabs">
        <button className={`ctrl-btn ${tab === 'cwt' ? 'active' : ''}`} onClick={() => setTab('cwt')}>Wavelet Transform</button>
        <button className={`ctrl-btn ${tab === 'velocity' ? 'active' : ''}`} onClick={() => { setTab('velocity'); if (!gv) loadGV(); }}>Group Velocity</button>
      </div>

      {tab === 'cwt' && (
        <>
          <div className="wf-controls">
            <div className="ctrl-group">
              <label>Index</label>
              <input type="number" min={0} max={file.waveform_count - 1} value={index} onChange={e => setIndex(Number(e.target.value))} />
              <span className="ctrl-hint">/ {file.waveform_count - 1}</span>
            </div>
            <div className="ctrl-group">
              <label>Wavelet</label>
              <select className="filter-select" value={wavelet} onChange={e => setWavelet(e.target.value)}>
                <option value="morl">Morlet</option>
                <option value="cmor1.5-1.0">Complex Morlet</option>
                <option value="cgau4">Complex Gaussian</option>
                <option value="mexh">Mexican Hat</option>
              </select>
            </div>
            <div className="ctrl-group">
              <label>Freq bins</label>
              <input type="number" min={32} max={256} value={numFreqs} onChange={e => setNumFreqs(Number(e.target.value))} />
            </div>
            <label className="ctrl-toggle">
              <input type="checkbox" checked={keepPre} onChange={e => setKeepPre(e.target.checked)} />
              <span>Pre-trigger</span>
            </label>
          </div>

          {loading && <div className="loading-indicator">Computing...</div>}

          {cwt && (
            <div className="wf-meta">
              <span>CH{cwt.channel}</span>
              <span>Wavelet: {cwt.wavelet}</span>
              <span>Peak: {(cwt.peak_frequency / 1000).toFixed(1)} kHz @ {cwt.peak_time.toFixed(0)} μs</span>
              <span>{cwt.sample_rate / 1000} kHz SR</span>
            </div>
          )}

          {wf && (
            <div className="panel">
              <div className="panel-head">Waveform #{index}</div>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={wfData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="t" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} label={{ value: 'Time (μs)', fill: '#64748b', fontSize: 11, position: 'insideBottom', offset: -4 }} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} label={{ value: 'Voltage', fill: '#64748b', fontSize: 11, angle: -90, position: 'insideLeft' }} />
                  <Line type="monotone" dataKey="v" stroke="#22d3ee" dot={false} strokeWidth={1} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {cwt && (
            <div className="panel">
              <div className="panel-head">
                CWT Spectrogram
                <span className="panel-tag">Peak: {(cwt.peak_frequency / 1000).toFixed(1)} kHz</span>
              </div>
              <SpectrogramCanvas cwt={cwt} />
            </div>
          )}

          {disp && (
            <div className="panel-grid-2">
              <div className="panel">
                <div className="panel-head">Arrival Time vs Frequency</div>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={dispData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="freq" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} label={{ value: 'Freq (kHz)', fill: '#64748b', fontSize: 11, position: 'insideBottom', offset: -4 }} />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} label={{ value: 'Time (μs)', fill: '#64748b', fontSize: 11, angle: -90, position: 'insideLeft' }} />
                    <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }} />
                    <Line type="monotone" dataKey="arrival" stroke="#34d399" dot={{ r: 2 }} strokeWidth={1.5} name="Arrival" />
                    <Line type="monotone" dataKey="peak" stroke="#fb923c" dot={{ r: 2 }} strokeWidth={1.5} name="Peak" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="panel">
                <div className="panel-head">Energy at Frequency</div>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={dispData} barSize={4}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="freq" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} label={{ value: 'Freq (kHz)', fill: '#64748b', fontSize: 11, position: 'insideBottom', offset: -4 }} />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} />
                    <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }} />
                    <Bar dataKey="energy" fill="#a78bfa" radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </>
      )}

      {tab === 'velocity' && (
        <>
          <div className="wf-controls">
            <div className="ctrl-group">
              <label>Sensor distance (m)</label>
              <input type="number" min={0.01} step={0.01} value={sensorDist} onChange={e => setSensorDist(Number(e.target.value))} style={{ width: 90 }} />
            </div>
            <label className="ctrl-toggle">
              <input type="checkbox" checked={keepPre} onChange={e => setKeepPre(e.target.checked)} />
              <span>Pre-trigger</span>
            </label>
            <button className="ctrl-btn" onClick={loadGV}>Compute</button>
          </div>

          {loading && <div className="loading-indicator">Computing cross-channel analysis...</div>}

          {gv && gv.pairs.length > 0 && (
            <>
              <div className="metrics-row compact">
                <div className="metric"><span className="metric-val">{gv.channel_count}</span><span className="metric-key">Channels</span></div>
                <div className="metric"><span className="metric-val">{gv.sensor_distance}<small>m</small></span><span className="metric-key">Distance</span></div>
                {gv.pairs.map(p => (
                  <div className="metric" key={`${p.ch1}-${p.ch2}`}>
                    <span className="metric-val">{p.avg_velocity.toFixed(0)}<small>m/s</small></span>
                    <span className="metric-key">CH{p.ch1}-CH{p.ch2} Avg</span>
                  </div>
                ))}
              </div>

              {gv.pairs.map(p => (
                <div key={`${p.ch1}-${p.ch2}`} className="panel">
                  <div className="panel-head">
                    CH{p.ch1} → CH{p.ch2} Group Velocity
                    <span className="panel-tag">{p.event_count} matched events · median {p.median_velocity.toFixed(0)} m/s</span>
                  </div>
                  <div className="panel-grid-2" style={{ marginBottom: 0 }}>
                    <ResponsiveContainer width="100%" height={240}>
                      <ScatterChart>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                        <XAxis dataKey="x" name="Time" tick={{ fill: '#94a3b8', fontSize: 10 }} type="number" axisLine={false} label={{ value: 'Event Time (s)', fill: '#64748b', fontSize: 11, position: 'insideBottom', offset: -4 }} />
                        <YAxis dataKey="y" name="Velocity" tick={{ fill: '#94a3b8', fontSize: 10 }} type="number" axisLine={false} label={{ value: 'Velocity (m/s)', fill: '#64748b', fontSize: 11, angle: -90, position: 'insideLeft' }} />
                        <ZAxis range={[15, 15]} />
                        <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }} />
                        <Scatter
                          data={p.event_times.map((t, i) => ({ x: Number(t.toFixed(3)), y: p.velocities[i] }))}
                          fill="#22d3ee" fillOpacity={0.5}
                        />
                      </ScatterChart>
                    </ResponsiveContainer>
                    <ResponsiveContainer width="100%" height={240}>
                      <ScatterChart>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                        <XAxis dataKey="x" name="Time" tick={{ fill: '#94a3b8', fontSize: 10 }} type="number" axisLine={false} label={{ value: 'Event Time (s)', fill: '#64748b', fontSize: 11, position: 'insideBottom', offset: -4 }} />
                        <YAxis dataKey="y" name="ΔT" tick={{ fill: '#94a3b8', fontSize: 10 }} type="number" axisLine={false} label={{ value: 'ΔT (s)', fill: '#64748b', fontSize: 11, angle: -90, position: 'insideLeft' }} />
                        <ZAxis range={[15, 15]} />
                        <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }} />
                        <Scatter
                          data={p.event_times.map((t, i) => ({ x: Number(t.toFixed(3)), y: p.delta_t[i] }))}
                          fill="#a78bfa" fillOpacity={0.5}
                        />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              ))}
            </>
          )}

          {gv && gv.pairs.length === 0 && (
            <div className="empty-state">
              <span>No matched events between channels</span>
              <small>Events are matched when arrival times differ by less than 10ms</small>
            </div>
          )}
        </>
      )}
    </div>
  );
}
