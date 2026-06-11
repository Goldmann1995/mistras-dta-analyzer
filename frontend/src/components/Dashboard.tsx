import { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { getChannelStats, getHistogramData } from '../services/api';
import type { FileInfo, ChannelStats } from '../types';

interface Props {
  file: FileInfo;
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];

export default function Dashboard({ file }: Props) {
  const [stats, setStats] = useState<ChannelStats[]>([]);
  const [ampHist, setAmpHist] = useState<{ range: string; count: number }[]>([]);

  useEffect(() => {
    getChannelStats(file.file_id).then(setStats);
    getHistogramData(file.file_id, 'amplitude', 30).then((data) => {
      const bars = data.counts.map((c, i) => ({
        range: `${data.edges[i].toFixed(0)}`,
        count: c,
      }));
      setAmpHist(bars);
    });
  }, [file.file_id]);

  const channelHits = stats.map((s) => ({
    name: `CH${s.channel}`,
    hits: s.hit_count,
    avgAmp: s.avg_amplitude,
  }));

  return (
    <div className="dashboard">
      <div className="stats-cards">
        <div className="stat-card">
          <div className="stat-value">{file.hit_count.toLocaleString()}</div>
          <div className="stat-label">总事件数</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{file.waveform_count.toLocaleString()}</div>
          <div className="stat-label">波形数</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{file.channels.length}</div>
          <div className="stat-label">通道数</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{file.duration.toFixed(1)}s</div>
          <div className="stat-label">持续时间</div>
        </div>
      </div>

      <div className="dashboard-charts">
        <div className="chart-panel">
          <h3>各通道事件数</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={channelHits}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="name" stroke="#aaa" />
              <YAxis stroke="#aaa" />
              <Tooltip contentStyle={{ background: '#1e1e2e', border: '1px solid #333' }} />
              <Bar dataKey="hits" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-panel">
          <h3>通道分布</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={channelHits}
                dataKey="hits"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={90}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {channelHits.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ background: '#1e1e2e', border: '1px solid #333' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-panel full-width">
          <h3>振幅分布</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={ampHist}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="range" stroke="#aaa" interval="preserveStartEnd" />
              <YAxis stroke="#aaa" />
              <Tooltip contentStyle={{ background: '#1e1e2e', border: '1px solid #333' }} />
              <Bar dataKey="count" fill="#10b981" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="channel-stats-table">
        <h3>通道统计</h3>
        <table>
          <thead>
            <tr>
              <th>通道</th>
              <th>事件数</th>
              <th>平均振幅</th>
              <th>最大振幅</th>
              <th>平均能量</th>
              <th>最大能量</th>
              <th>平均持续时间</th>
              <th>时间跨度(s)</th>
            </tr>
          </thead>
          <tbody>
            {stats.map((s) => (
              <tr key={s.channel}>
                <td>CH{s.channel}</td>
                <td>{s.hit_count}</td>
                <td>{s.avg_amplitude.toFixed(1)}</td>
                <td>{s.max_amplitude}</td>
                <td>{s.avg_energy.toFixed(1)}</td>
                <td>{s.max_energy}</td>
                <td>{s.avg_duration.toFixed(0)} μs</td>
                <td>{s.time_span.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
