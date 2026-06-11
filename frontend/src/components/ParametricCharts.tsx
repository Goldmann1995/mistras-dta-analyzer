import { useState, useEffect } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, ZAxis } from 'recharts';
import { getScatterData, getHistogramData } from '../services/api';
import type { FileInfo } from '../types';

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
  const [scatter, setScatter] = useState<{ x: number; y: number; z: number }[]>([]);
  const [hist, setHist] = useState<{ v: string; n: number }[]>([]);

  useEffect(() => {
    getScatterData(file.file_id, xF, yF, cF, ch).then(d => {
      setScatter(d.x.map((x, i) => ({ x, y: d.y[i], z: d.color?.[i] ?? 1 })));
    });
  }, [file.file_id, xF, yF, cF, ch]);

  useEffect(() => {
    getHistogramData(file.file_id, hF, 40, ch).then(d => {
      setHist(d.counts.map((c, i) => ({ v: ((d.edges[i] + d.edges[i + 1]) / 2).toFixed(1), n: c })));
    });
  }, [file.file_id, hF, ch]);

  const fl = (v: string) => FIELDS.find(f => f.v === v)?.l ?? v;

  return (
    <div className="view-charts">
      <div className="charts-controls">
        <label className="filter-label">Channel</label>
        <select className="filter-select" value={ch ?? ''} onChange={e => setCh(e.target.value ? Number(e.target.value) : undefined)}>
          <option value="">All</option>
          {file.channels.map(c => <option key={c} value={c}>CH{c}</option>)}
        </select>
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
              <Scatter data={scatter} fill="#22d3ee" fillOpacity={0.5} />
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
    </div>
  );
}
