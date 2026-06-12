import { useState, useEffect, useRef, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ScatterChart, Scatter, BarChart, Bar, Cell, ZAxis,
} from 'recharts';
import { getCWT, getDispersion, getGroupVelocity, getWaveform, getEMD, getLambDispersion } from '../services/api';
import type { FileInfo, CWTResult, DispersionResult, GroupVelocityResult, WaveformData, EMDResult, LambDispersionResult } from '../types';

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

    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth;
    const cssH = 500;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const nFreqs = cwt.power.length;
    const nTimes = cwt.power[0]?.length ?? 0;
    if (nFreqs === 0 || nTimes === 0) return;

    const margin = { top: 16, right: 20, bottom: 56, left: 80 };
    const plotW = cssW - margin.left - margin.right;
    const plotH = cssH - margin.top - margin.bottom;

    ctx.fillStyle = '#060a13';
    ctx.fillRect(0, 0, cssW, cssH);

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

    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 1;
    ctx.strokeRect(margin.left, margin.top, plotW, plotH);

    // X-axis tick labels
    ctx.fillStyle = '#e2e8f0';
    ctx.font = '13px "IBM Plex Mono", monospace';
    ctx.textAlign = 'center';
    const tAxis = cwt.time_axis;
    for (let i = 0; i <= 6; i++) {
      const ti = Math.floor(i / 6 * (tAxis.length - 1));
      const x = margin.left + (i / 6) * plotW;
      ctx.fillText(`${tAxis[ti].toFixed(0)}`, x, cssH - margin.bottom + 18);
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + plotH); ctx.stroke();
    }
    // X-axis label
    ctx.fillStyle = '#22d3ee';
    ctx.font = 'bold 14px "Inter", sans-serif';
    ctx.fillText('Time (μs)', margin.left + plotW / 2, cssH - 6);

    // Y-axis tick labels
    ctx.fillStyle = '#e2e8f0';
    ctx.font = '13px "IBM Plex Mono", monospace';
    ctx.textAlign = 'right';
    const fAxis = cwt.freq_axis;
    for (let i = 0; i <= 6; i++) {
      const fi = Math.floor(i / 6 * (fAxis.length - 1));
      const y = margin.top + plotH - (i / 6) * plotH;
      ctx.fillText(`${(fAxis[fi] / 1000).toFixed(0)}`, margin.left - 8, y + 4);
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(margin.left + plotW, y); ctx.stroke();
    }
    // Y-axis label
    ctx.save();
    ctx.fillStyle = '#22d3ee';
    ctx.font = 'bold 14px "Inter", sans-serif';
    ctx.translate(16, margin.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Frequency (kHz)', 0, 0);
    ctx.restore();

    // Peak marker
    const pkX = margin.left + ((cwt.peak_time - tAxis[0]) / (tAxis[tAxis.length - 1] - tAxis[0])) * plotW;
    const pkY = margin.top + plotH - ((cwt.peak_frequency - fAxis[0]) / (fAxis[fAxis.length - 1] - fAxis[0])) * plotH;
    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(pkX, pkY, 8, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = '#22d3ee';
    ctx.font = 'bold 12px "IBM Plex Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`${(cwt.peak_frequency / 1000).toFixed(1)} kHz`, pkX + 12, pkY + 4);

  }, [cwt]);

  return <canvas ref={canvasRef} style={{ width: '100%', height: 500, borderRadius: 6 }} />;
}

const IMF_COLORS = ['#22d3ee', '#a78bfa', '#34d399', '#fb923c', '#f87171', '#fbbf24', '#818cf8', '#f472b6'];

export default function WaveletView({ file }: Props) {
  const [index, setIndex] = useState(0);
  const [wavelet, setWavelet] = useState('morl');
  const [numFreqs, setNumFreqs] = useState(128);
  const [keepPre, setKeepPre] = useState(false);
  const [cwt, setCwt] = useState<CWTResult | null>(null);
  const [disp, setDisp] = useState<DispersionResult | null>(null);
  const [wf, setWf] = useState<WaveformData | null>(null);
  const [gv, setGv] = useState<GroupVelocityResult | null>(null);
  const [emd, setEmd] = useState<EMDResult | null>(null);
  const [emdMethod, setEmdMethod] = useState('emd');
  const [sensorDist, setSensorDist] = useState(0.3);
  const [loading, setLoading] = useState(false);
  const [lamb, setLamb] = useState<LambDispersionResult | null>(null);
  const [plateThickness, setPlateThickness] = useState(0.002);
  const [plateCl, setPlateCl] = useState(6320);
  const [plateCt, setPlateCt] = useState(3130);
  const [tab, setTab] = useState<'cwt' | 'emd' | 'lamb' | 'velocity'>('cwt');

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
      setGv(await getGroupVelocity(file.file_id, sensorDist, keepPre));
    } finally {
      setLoading(false);
    }
  }, [file.file_id, sensorDist, keepPre]);

  const loadLamb = useCallback(async () => {
    setLoading(true);
    try {
      setLamb(await getLambDispersion({
        thickness: plateThickness, cl: plateCl, ct: plateCt,
        freq_max: (cwt?.sample_rate ?? 1000000) / 2,
      }));
    } finally {
      setLoading(false);
    }
  }, [plateThickness, plateCl, plateCt, cwt?.sample_rate]);

  const loadEMD = useCallback(async () => {
    setLoading(true);
    try {
      setEmd(await getEMD(file.file_id, index, { method: emdMethod, keep_pretrigger: keepPre }));
    } finally {
      setLoading(false);
    }
  }, [file.file_id, index, emdMethod, keepPre]);

  useEffect(() => { if (tab === 'emd') loadEMD(); }, [tab, loadEMD]);

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
        <button className={`ctrl-btn ${tab === 'emd' ? 'active' : ''}`} onClick={() => setTab('emd')}>EMD</button>
        <button className={`ctrl-btn ${tab === 'lamb' ? 'active' : ''}`} onClick={() => { setTab('lamb'); if (!lamb) loadLamb(); }}>Lamb Wave</button>
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

      {tab === 'emd' && (
        <>
          <div className="wf-controls">
            <div className="ctrl-group">
              <label>Index</label>
              <input type="number" min={0} max={file.waveform_count - 1} value={index} onChange={e => setIndex(Number(e.target.value))} />
              <span className="ctrl-hint">/ {file.waveform_count - 1}</span>
            </div>
            <div className="ctrl-group">
              <label>Method</label>
              <select className="filter-select" value={emdMethod} onChange={e => setEmdMethod(e.target.value)}>
                <option value="emd">EMD</option>
                <option value="eemd">EEMD</option>
              </select>
            </div>
            <label className="ctrl-toggle">
              <input type="checkbox" checked={keepPre} onChange={e => setKeepPre(e.target.checked)} />
              <span>Pre-trigger</span>
            </label>
            <button className="ctrl-btn" onClick={loadEMD}>Compute</button>
          </div>

          {loading && <div className="loading-indicator">Computing EMD decomposition...</div>}

          {emd && (
            <>
              <div className="wf-meta">
                <span>CH{emd.channel}</span>
                <span>Method: {emd.method.toUpperCase()}</span>
                <span>{emd.num_imfs} IMFs</span>
                <span>{emd.sample_rate / 1000} kHz SR</span>
              </div>

              <div className="panel">
                <div className="panel-head">IMF Energy Distribution</div>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={emd.imfs.map(m => ({
                    name: `IMF ${m.index + 1}`,
                    energy: Number((m.energy_ratio * 100).toFixed(1)),
                    freq: Number((m.dominant_frequency / 1000).toFixed(1)),
                  }))} barSize={24}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="name" tick={{ fill: '#e2e8f0', fontSize: 11 }} axisLine={false} />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} label={{ value: 'Energy %', fill: '#64748b', fontSize: 11, angle: -90, position: 'insideLeft' }} />
                    <Tooltip
                      contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }}
                      formatter={(v: number, name: string) => name === 'energy' ? [`${v}%`, 'Energy'] : [`${v} kHz`, 'Dom. Freq']}
                    />
                    <Bar dataKey="energy" radius={[4, 4, 0, 0]}>
                      {emd.imfs.map((_, i) => <Cell key={i} fill={IMF_COLORS[i % IMF_COLORS.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {emd.imfs.map((imf, idx) => {
                const imfData = imf.data.map((v, i) => ({
                  t: Number(emd.time_axis[i]?.toFixed(1) ?? 0),
                  v,
                  freq: imf.inst_frequency[i] ? Number((imf.inst_frequency[i] / 1000).toFixed(1)) : 0,
                }));
                const color = IMF_COLORS[idx % IMF_COLORS.length];
                return (
                  <div key={idx} className="panel">
                    <div className="panel-head">
                      IMF {imf.index + 1}
                      <span className="panel-tag">
                        {(imf.dominant_frequency / 1000).toFixed(1)} kHz · {(imf.energy_ratio * 100).toFixed(1)}% energy
                      </span>
                    </div>
                    <div className="panel-grid-2" style={{ marginBottom: 0 }}>
                      <ResponsiveContainer width="100%" height={160}>
                        <LineChart data={imfData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                          <XAxis dataKey="t" tick={{ fill: '#94a3b8', fontSize: 9 }} axisLine={false} label={{ value: 'Time (μs)', fill: '#64748b', fontSize: 10, position: 'insideBottom', offset: -4 }} />
                          <YAxis tick={{ fill: '#94a3b8', fontSize: 9 }} axisLine={false} />
                          <Line type="monotone" dataKey="v" stroke={color} dot={false} strokeWidth={1} />
                        </LineChart>
                      </ResponsiveContainer>
                      <ResponsiveContainer width="100%" height={160}>
                        <LineChart data={imfData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                          <XAxis dataKey="t" tick={{ fill: '#94a3b8', fontSize: 9 }} axisLine={false} label={{ value: 'Time (μs)', fill: '#64748b', fontSize: 10, position: 'insideBottom', offset: -4 }} />
                          <YAxis tick={{ fill: '#94a3b8', fontSize: 9 }} axisLine={false} label={{ value: 'Freq (kHz)', fill: '#64748b', fontSize: 10, angle: -90, position: 'insideLeft' }} />
                          <Line type="monotone" dataKey="freq" stroke={color} dot={false} strokeWidth={1} strokeOpacity={0.6} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                );
              })}
            </>
          )}
        </>
      )}

      {tab === 'lamb' && (
        <>
          <div className="wf-controls">
            <div className="ctrl-group">
              <label>Thickness (mm)</label>
              <input type="number" min={0.1} step={0.1} value={plateThickness * 1000}
                onChange={e => setPlateThickness(Number(e.target.value) / 1000)} style={{ width: 70 }} />
            </div>
            <div className="ctrl-group">
              <label>c<sub>L</sub> (m/s)</label>
              <input type="number" min={1000} step={100} value={plateCl}
                onChange={e => setPlateCl(Number(e.target.value))} style={{ width: 80 }} />
            </div>
            <div className="ctrl-group">
              <label>c<sub>T</sub> (m/s)</label>
              <input type="number" min={500} step={100} value={plateCt}
                onChange={e => setPlateCt(Number(e.target.value))} style={{ width: 80 }} />
            </div>
            <div className="ctrl-group">
              <label>Material</label>
              <select className="filter-select" onChange={e => {
                const presets: Record<string, [number, number, number]> = {
                  aluminum: [0.002, 6320, 3130],
                  steel: [0.002, 5960, 3260],
                  copper: [0.002, 4760, 2325],
                  glass: [0.002, 5640, 3280],
                };
                const p = presets[e.target.value];
                if (p) { setPlateThickness(p[0]); setPlateCl(p[1]); setPlateCt(p[2]); }
              }}>
                <option value="">Custom</option>
                <option value="aluminum">Aluminum</option>
                <option value="steel">Steel</option>
                <option value="copper">Copper</option>
                <option value="glass">Glass</option>
              </select>
            </div>
            <button className="ctrl-btn" onClick={loadLamb}>Compute</button>
          </div>

          {loading && <div className="loading-indicator">Computing Lamb wave dispersion...</div>}

          {lamb && (
            <>
              <div className="wf-meta">
                <span>Thickness: {(lamb.thickness * 1000).toFixed(1)} mm</span>
                <span>c<sub>L</sub>: {lamb.cl} m/s</span>
                <span>c<sub>T</sub>: {lamb.ct} m/s</span>
                <span>S modes: {lamb.modes.symmetric.length}</span>
                <span>A modes: {lamb.modes.antisymmetric.length}</span>
              </div>

              <div className="panel-grid-2">
                <div className="panel">
                  <div className="panel-head">Phase Velocity Dispersion</div>
                  <ResponsiveContainer width="100%" height={360}>
                    <LineChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                      <XAxis
                        dataKey="freq" type="number"
                        domain={[lamb.freq_range[0] / 1000, lamb.freq_range[1] / 1000]}
                        tick={{ fill: '#e2e8f0', fontSize: 11 }} axisLine={false}
                        label={{ value: 'Frequency (kHz)', fill: '#22d3ee', fontSize: 12, fontWeight: 600, position: 'insideBottom', offset: -4 }}
                      />
                      <YAxis
                        tick={{ fill: '#e2e8f0', fontSize: 11 }} axisLine={false}
                        label={{ value: 'Phase Velocity (m/s)', fill: '#22d3ee', fontSize: 12, fontWeight: 600, angle: -90, position: 'insideLeft' }}
                      />
                      <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }} />
                      <Legend />
                      {lamb.modes.symmetric.map((m, i) => (
                        <Line key={m.mode} data={m.frequencies.map((f, j) => ({ freq: f / 1000, vp: m.phase_velocity[j] }))}
                          dataKey="vp" name={m.mode} stroke={['#22d3ee', '#34d399', '#a78bfa', '#fbbf24'][i % 4]}
                          dot={false} strokeWidth={2} strokeDasharray={undefined} />
                      ))}
                      {lamb.modes.antisymmetric.map((m, i) => (
                        <Line key={m.mode} data={m.frequencies.map((f, j) => ({ freq: f / 1000, vp: m.phase_velocity[j] }))}
                          dataKey="vp" name={m.mode} stroke={['#fb923c', '#f87171', '#f472b6', '#818cf8'][i % 4]}
                          dot={false} strokeWidth={2} strokeDasharray="6 3" />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div className="panel">
                  <div className="panel-head">Group Velocity Dispersion</div>
                  <ResponsiveContainer width="100%" height={360}>
                    <LineChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                      <XAxis
                        dataKey="freq" type="number"
                        domain={[lamb.freq_range[0] / 1000, lamb.freq_range[1] / 1000]}
                        tick={{ fill: '#e2e8f0', fontSize: 11 }} axisLine={false}
                        label={{ value: 'Frequency (kHz)', fill: '#22d3ee', fontSize: 12, fontWeight: 600, position: 'insideBottom', offset: -4 }}
                      />
                      <YAxis
                        tick={{ fill: '#e2e8f0', fontSize: 11 }} axisLine={false}
                        label={{ value: 'Group Velocity (m/s)', fill: '#22d3ee', fontSize: 12, fontWeight: 600, angle: -90, position: 'insideLeft' }}
                      />
                      <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }} />
                      <Legend />
                      {lamb.modes.symmetric.map((m, i) => m.group_velocity.length > 0 && (
                        <Line key={m.mode} data={m.frequencies.map((f, j) => ({ freq: f / 1000, vg: m.group_velocity[j] }))}
                          dataKey="vg" name={m.mode} stroke={['#22d3ee', '#34d399', '#a78bfa', '#fbbf24'][i % 4]}
                          dot={false} strokeWidth={2} />
                      ))}
                      {lamb.modes.antisymmetric.map((m, i) => m.group_velocity.length > 0 && (
                        <Line key={m.mode} data={m.frequencies.map((f, j) => ({ freq: f / 1000, vg: m.group_velocity[j] }))}
                          dataKey="vg" name={m.mode} stroke={['#fb923c', '#f87171', '#f472b6', '#818cf8'][i % 4]}
                          dot={false} strokeWidth={2} strokeDasharray="6 3" />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="panel">
                <div className="panel-head">Mode Summary</div>
                <table className="hit-table">
                  <thead><tr>
                    <th>Mode</th><th>Type</th><th>Freq Range (kHz)</th>
                    <th>Phase Vel (m/s)</th><th>Group Vel (m/s)</th>
                  </tr></thead>
                  <tbody>
                    {[...lamb.modes.symmetric, ...lamb.modes.antisymmetric].map(m => {
                      const gv = m.group_velocity.filter(v => v > 0);
                      return (
                        <tr key={m.mode}>
                          <td style={{ fontWeight: 600, color: m.mode.startsWith('S') ? '#22d3ee' : '#fb923c' }}>{m.mode}</td>
                          <td>{m.mode.startsWith('S') ? 'Symmetric' : 'Antisymmetric'}</td>
                          <td>{(m.frequencies[0]/1000).toFixed(0)} - {(m.frequencies[m.frequencies.length-1]/1000).toFixed(0)}</td>
                          <td>{Math.min(...m.phase_velocity).toFixed(0)} - {Math.max(...m.phase_velocity).toFixed(0)}</td>
                          <td>{gv.length ? `${Math.min(...gv).toFixed(0)} - ${Math.max(...gv).toFixed(0)}` : '-'}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </>
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
