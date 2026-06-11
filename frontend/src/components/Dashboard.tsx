import { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { getChannelStats, getHistogramData, getScatterData } from '../services/api';
import type { FileInfo, ChannelStats } from '../types';

interface Props { file: FileInfo; }

const CH_COLORS = ['#22d3ee', '#a78bfa', '#34d399', '#fb923c', '#f472b6', '#facc15', '#38bdf8', '#c084fc'];

export default function Dashboard({ file }: Props) {
  const [stats, setStats] = useState<ChannelStats[]>([]);
  const [ampHist, setAmpHist] = useState<{ v: string; n: number }[]>([]);
  const [ampTime, setAmpTime] = useState<{ x: number; y: number }[]>([]);

  useEffect(() => {
    getChannelStats(file.file_id).then(setStats);
    getHistogramData(file.file_id, 'amplitude', 30).then((d) => {
      setAmpHist(d.counts.map((c, i) => ({ v: `${d.edges[i].toFixed(0)}`, n: c })));
    });
    getScatterData(file.file_id, 'time', 'amplitude').then((d) => {
      setAmpTime(d.x.map((x, i) => ({ x: Number(x.toFixed(2)), y: d.y[i] })));
    });
  }, [file.file_id]);

  const chData = stats.map((s, i) => ({ name: `CH${s.channel}`, hits: s.hit_count, color: CH_COLORS[i % CH_COLORS.length] }));
  const totalHits = stats.reduce((a, s) => a + s.hit_count, 0);

  return (
    <div className="view-dashboard">
      <div className="metrics-row">
        <div className="metric">
          <span className="metric-val">{file.hit_count.toLocaleString()}</span>
          <span className="metric-key">Total Events</span>
        </div>
        <div className="metric">
          <span className="metric-val">{file.waveform_count.toLocaleString()}</span>
          <span className="metric-key">Waveforms</span>
        </div>
        <div className="metric">
          <span className="metric-val">{file.channels.length}</span>
          <span className="metric-key">Channels</span>
        </div>
        <div className="metric">
          <span className="metric-val">{file.duration.toFixed(1)}<small>s</small></span>
          <span className="metric-key">Duration</span>
        </div>
        <div className="metric">
          <span className="metric-val">{stats.length > 0 ? Math.max(...stats.map(s => s.max_amplitude)) : '-'}</span>
          <span className="metric-key">Peak Amplitude</span>
        </div>
      </div>

      <div className="panel-grid-2">
        <div className="panel">
          <div className="panel-head">Channel Distribution</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chData} barSize={28}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} />
              <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 12 }} />
              <Bar dataKey="hits" radius={[3, 3, 0, 0]}>
                {chData.map((d, i) => <Cell key={i} fill={d.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="channel-legend">
            {stats.map((s, i) => (
              <span key={s.channel} className="ch-tag" style={{ borderColor: CH_COLORS[i % CH_COLORS.length] }}>
                CH{s.channel}: {s.hit_count} <small>({(s.hit_count / totalHits * 100).toFixed(0)}%)</small>
              </span>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">Amplitude Distribution</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={ampHist} barSize={6}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="v" tick={{ fill: '#94a3b8', fontSize: 10 }} interval="preserveStartEnd" axisLine={false} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} />
              <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 12 }} />
              <Bar dataKey="n" fill="#22d3ee" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="panel full">
        <div className="panel-head">Amplitude vs Time</div>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={ampTime} barSize={1.5}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="x" tick={{ fill: '#94a3b8', fontSize: 10 }} interval="preserveStartEnd" axisLine={false} label={{ value: 'Time (s)', fill: '#64748b', fontSize: 11, position: 'insideBottom', offset: -4 }} />
            <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} label={{ value: 'Amp (dB)', fill: '#64748b', fontSize: 11, angle: -90, position: 'insideLeft' }} />
            <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 12 }} />
            <Bar dataKey="y" fill="#a78bfa" radius={[1, 1, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="panel">
        <div className="panel-head">Channel Statistics</div>
        <table className="data-table">
          <thead>
            <tr>
              <th>CH</th><th>Events</th><th>Avg Amp</th><th>Max Amp</th>
              <th>Avg Energy</th><th>Avg Duration</th><th>Span (s)</th>
            </tr>
          </thead>
          <tbody>
            {stats.map((s, i) => (
              <tr key={s.channel}>
                <td><span className="ch-dot" style={{ background: CH_COLORS[i % CH_COLORS.length] }} />CH{s.channel}</td>
                <td>{s.hit_count}</td>
                <td>{s.avg_amplitude.toFixed(1)}</td>
                <td>{s.max_amplitude}</td>
                <td>{s.avg_energy.toFixed(1)}</td>
                <td>{s.avg_duration.toFixed(0)} us</td>
                <td>{s.time_span.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
