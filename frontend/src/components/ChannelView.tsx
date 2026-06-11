import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { getScatterData, getChannelStats, getHistogramData } from '../services/api';
import type { FileInfo, ChannelStats } from '../types';

interface Props { file: FileInfo; }

const COLORS = ['#22d3ee', '#a78bfa', '#34d399', '#fb923c', '#f472b6', '#facc15', '#38bdf8', '#c084fc'];

export default function ChannelView({ file }: Props) {
  const [stats, setStats] = useState<ChannelStats[]>([]);
  const [selCh, setSelCh] = useState<number | null>(null);
  const [ampTime, setAmpTime] = useState<{ x: number; y: number }[]>([]);
  const [eHist, setEHist] = useState<{ v: string; n: number }[]>([]);

  useEffect(() => {
    getChannelStats(file.file_id).then(s => { setStats(s); if (s.length > 0 && selCh === null) setSelCh(s[0].channel); });
  }, [file.file_id]);

  useEffect(() => {
    if (selCh === null) return;
    getScatterData(file.file_id, 'time', 'amplitude', undefined, selCh).then(d => {
      setAmpTime(d.x.map((x, i) => ({ x: Number(x.toFixed(3)), y: d.y[i] })));
    });
    getHistogramData(file.file_id, 'energy', 30, selCh).then(d => {
      setEHist(d.counts.map((c, i) => ({ v: ((d.edges[i] + d.edges[i + 1]) / 2).toFixed(0), n: c })));
    });
  }, [file.file_id, selCh]);

  const chStat = stats.find(s => s.channel === selCh);
  const chIdx = file.channels.indexOf(selCh ?? -1);
  const color = COLORS[chIdx >= 0 ? chIdx % COLORS.length : 0];

  return (
    <div className="view-channels">
      <div className="ch-tabs">
        {file.channels.map((ch, i) => {
          const s = stats.find(st => st.channel === ch);
          return (
            <button key={ch} className={`ch-tab ${selCh === ch ? 'active' : ''}`} style={{ '--ch-color': COLORS[i % COLORS.length] } as React.CSSProperties} onClick={() => setSelCh(ch)}>
              <span className="ch-tab-name">CH{ch}</span>
              <span className="ch-tab-count">{s?.hit_count ?? 0}</span>
            </button>
          );
        })}
      </div>

      {chStat && (
        <>
          <div className="metrics-row compact">
            <div className="metric"><span className="metric-val">{chStat.hit_count}</span><span className="metric-key">Events</span></div>
            <div className="metric"><span className="metric-val">{chStat.avg_amplitude.toFixed(1)}</span><span className="metric-key">Avg Amp</span></div>
            <div className="metric"><span className="metric-val">{chStat.max_amplitude}</span><span className="metric-key">Max Amp</span></div>
            <div className="metric"><span className="metric-val">{chStat.avg_energy.toFixed(1)}</span><span className="metric-key">Avg Energy</span></div>
            <div className="metric"><span className="metric-val">{chStat.avg_duration.toFixed(0)}<small>us</small></span><span className="metric-key">Avg Duration</span></div>
            <div className="metric"><span className="metric-val">{chStat.time_span.toFixed(1)}<small>s</small></span><span className="metric-key">Time Span</span></div>
          </div>

          <div className="panel-grid-2">
            <div className="panel">
              <div className="panel-head">CH{selCh} Amplitude vs Time</div>
              <ResponsiveContainer width="100%" height={270}>
                <LineChart data={ampTime}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="x" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} label={{ value: 'Time (s)', fill: '#64748b', fontSize: 11, position: 'insideBottom', offset: -4 }} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} />
                  <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }} />
                  <Line type="monotone" dataKey="y" stroke={color} dot={{ r: 1.5, fill: color }} strokeWidth={0} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="panel">
              <div className="panel-head">CH{selCh} Energy Distribution</div>
              <ResponsiveContainer width="100%" height={270}>
                <BarChart data={eHist} barSize={8}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="v" tick={{ fill: '#94a3b8', fontSize: 10 }} interval="preserveStartEnd" axisLine={false} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} />
                  <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }} />
                  <Bar dataKey="n" fill={color} radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
