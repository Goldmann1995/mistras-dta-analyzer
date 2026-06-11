import { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar,
} from 'recharts';
import { getScatterData, getChannelStats, getHistogramData } from '../services/api';
import type { FileInfo, ChannelStats } from '../types';

interface Props {
  file: FileInfo;
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];

export default function ChannelView({ file }: Props) {
  const [stats, setStats] = useState<ChannelStats[]>([]);
  const [selectedChannel, setSelectedChannel] = useState<number | null>(null);
  const [ampVsTime, setAmpVsTime] = useState<{ x: number; y: number }[]>([]);
  const [energyHist, setEnergyHist] = useState<{ range: string; count: number }[]>([]);

  useEffect(() => {
    getChannelStats(file.file_id).then((s) => {
      setStats(s);
      if (s.length > 0 && selectedChannel === null) {
        setSelectedChannel(s[0].channel);
      }
    });
  }, [file.file_id]);

  useEffect(() => {
    if (selectedChannel === null) return;
    getScatterData(file.file_id, 'time', 'amplitude', undefined, selectedChannel).then((data) => {
      setAmpVsTime(data.x.map((x, i) => ({ x: Number(x.toFixed(4)), y: data.y[i] })));
    });
    getHistogramData(file.file_id, 'energy', 30, selectedChannel).then((data) => {
      setEnergyHist(data.counts.map((c, i) => ({
        range: ((data.edges[i] + data.edges[i + 1]) / 2).toFixed(0),
        count: c,
      })));
    });
  }, [file.file_id, selectedChannel]);

  const chStat = stats.find((s) => s.channel === selectedChannel);

  return (
    <div className="channel-view">
      <div className="channel-tabs">
        {file.channels.map((ch, i) => (
          <button
            key={ch}
            className={`channel-tab ${selectedChannel === ch ? 'active' : ''}`}
            style={{ borderColor: COLORS[i % COLORS.length] }}
            onClick={() => setSelectedChannel(ch)}
          >
            CH{ch}
            <span className="tab-count">{stats.find((s) => s.channel === ch)?.hit_count ?? 0}</span>
          </button>
        ))}
      </div>

      {chStat && (
        <div className="channel-detail">
          <div className="stats-cards compact">
            <div className="stat-card"><div className="stat-value">{chStat.hit_count}</div><div className="stat-label">事件数</div></div>
            <div className="stat-card"><div className="stat-value">{chStat.avg_amplitude.toFixed(1)}</div><div className="stat-label">平均振幅</div></div>
            <div className="stat-card"><div className="stat-value">{chStat.max_amplitude}</div><div className="stat-label">最大振幅</div></div>
            <div className="stat-card"><div className="stat-value">{chStat.avg_energy.toFixed(1)}</div><div className="stat-label">平均能量</div></div>
            <div className="stat-card"><div className="stat-value">{chStat.avg_duration.toFixed(0)}</div><div className="stat-label">平均持续(μs)</div></div>
            <div className="stat-card"><div className="stat-value">{chStat.time_span.toFixed(2)}s</div><div className="stat-label">时间跨度</div></div>
          </div>

          <div className="charts-grid">
            <div className="chart-panel">
              <h3>CH{selectedChannel} 振幅 vs 时间</h3>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={ampVsTime}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis dataKey="x" stroke="#aaa" label={{ value: '时间 (s)', position: 'insideBottom', offset: -5, fill: '#aaa' }} />
                  <YAxis stroke="#aaa" label={{ value: '振幅', angle: -90, position: 'insideLeft', fill: '#aaa' }} />
                  <Tooltip contentStyle={{ background: '#1e1e2e', border: '1px solid #333' }} />
                  <Line type="monotone" dataKey="y" stroke={COLORS[(selectedChannel - 1) % COLORS.length]} dot={{ r: 2 }} strokeWidth={0} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-panel">
              <h3>CH{selectedChannel} 能量分布</h3>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={energyHist}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis dataKey="range" stroke="#aaa" interval="preserveStartEnd" />
                  <YAxis stroke="#aaa" />
                  <Tooltip contentStyle={{ background: '#1e1e2e', border: '1px solid #333' }} />
                  <Bar dataKey="count" fill={COLORS[(selectedChannel - 1) % COLORS.length]} radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
