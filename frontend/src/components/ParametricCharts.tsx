import { useState, useEffect } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, ZAxis,
} from 'recharts';
import { getScatterData, getHistogramData } from '../services/api';
import type { FileInfo, ScatterData, HistogramData } from '../types';

interface Props {
  file: FileInfo;
}

const FIELD_OPTIONS = [
  { value: 'time', label: '时间' },
  { value: 'amplitude', label: '振幅' },
  { value: 'energy', label: '能量' },
  { value: 'duration', label: '持续时间' },
  { value: 'rise', label: '上升时间' },
  { value: 'counts', label: '计数' },
  { value: 'rms', label: 'RMS' },
  { value: 'avg_frequency', label: '平均频率' },
  { value: 'peak_frequency', label: '峰值频率' },
  { value: 'abs_energy', label: '绝对能量' },
  { value: 'signal_strength', label: '信号强度' },
  { value: 'freq_centroid', label: '频率质心' },
];

export default function ParametricCharts({ file }: Props) {
  const [xField, setXField] = useState('time');
  const [yField, setYField] = useState('amplitude');
  const [colorField, setColorField] = useState('energy');
  const [histField, setHistField] = useState('amplitude');
  const [channel, setChannel] = useState<number | undefined>();
  const [scatter, setScatter] = useState<ScatterData | null>(null);
  const [histogram, setHistogram] = useState<HistogramData | null>(null);

  useEffect(() => {
    getScatterData(file.file_id, xField, yField, colorField, channel).then(setScatter);
  }, [file.file_id, xField, yField, colorField, channel]);

  useEffect(() => {
    getHistogramData(file.file_id, histField, 40, channel).then(setHistogram);
  }, [file.file_id, histField, channel]);

  const scatterData = scatter
    ? scatter.x.map((x, i) => ({ x, y: scatter.y[i], z: scatter.color?.[i] ?? 1 }))
    : [];

  const histData = histogram
    ? histogram.counts.map((c, i) => ({
        range: ((histogram.edges[i] + histogram.edges[i + 1]) / 2).toFixed(1),
        count: c,
      }))
    : [];

  const fieldLabel = (v: string) => FIELD_OPTIONS.find((f) => f.value === v)?.label ?? v;

  return (
    <div className="parametric-charts">
      <div className="chart-controls">
        <div className="control-group">
          <label>通道:</label>
          <select value={channel ?? ''} onChange={(e) => setChannel(e.target.value ? Number(e.target.value) : undefined)}>
            <option value="">全部</option>
            {file.channels.map((ch) => <option key={ch} value={ch}>CH{ch}</option>)}
          </select>
        </div>
      </div>

      <div className="charts-grid">
        <div className="chart-panel">
          <div className="chart-header">
            <h3>散点图</h3>
            <div className="chart-selectors">
              <select value={xField} onChange={(e) => setXField(e.target.value)}>
                {FIELD_OPTIONS.map((f) => <option key={f.value} value={f.value}>{f.label}</option>)}
              </select>
              <span>vs</span>
              <select value={yField} onChange={(e) => setYField(e.target.value)}>
                {FIELD_OPTIONS.map((f) => <option key={f.value} value={f.value}>{f.label}</option>)}
              </select>
              <span>颜色:</span>
              <select value={colorField} onChange={(e) => setColorField(e.target.value)}>
                {FIELD_OPTIONS.map((f) => <option key={f.value} value={f.value}>{f.label}</option>)}
              </select>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={350}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="x" name={fieldLabel(xField)} stroke="#aaa" type="number" />
              <YAxis dataKey="y" name={fieldLabel(yField)} stroke="#aaa" type="number" />
              <ZAxis dataKey="z" range={[20, 200]} />
              <Tooltip
                contentStyle={{ background: '#1e1e2e', border: '1px solid #333' }}
                formatter={(value: number) => value.toFixed(4)}
              />
              <Scatter data={scatterData} fill="#3b82f6" fillOpacity={0.6} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-panel">
          <div className="chart-header">
            <h3>直方图</h3>
            <div className="chart-selectors">
              <select value={histField} onChange={(e) => setHistField(e.target.value)}>
                {FIELD_OPTIONS.filter((f) => f.value !== 'time').map((f) => (
                  <option key={f.value} value={f.value}>{f.label}</option>
                ))}
              </select>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={histData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="range" stroke="#aaa" interval="preserveStartEnd" />
              <YAxis stroke="#aaa" />
              <Tooltip contentStyle={{ background: '#1e1e2e', border: '1px solid #333' }} />
              <Bar dataKey="count" fill="#10b981" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
