import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { getWaveform, getWaveformFFT } from '../services/api';
import type { FileInfo, WaveformData, FFTResult } from '../types';

interface Props {
  file: FileInfo;
  initialIndex?: number;
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

export default function WaveformViewer({ file, initialIndex }: Props) {
  const [waveform, setWaveform] = useState<WaveformData | null>(null);
  const [fft, setFft] = useState<FFTResult | null>(null);
  const [index, setIndex] = useState(initialIndex ?? 0);
  const [showFFT, setShowFFT] = useState(false);
  const [overlayCount, setOverlayCount] = useState(1);
  const [overlayData, setOverlayData] = useState<WaveformData[]>([]);

  useEffect(() => {
    if (file.waveform_count === 0) return;
    getWaveform(file.file_id, index).then(setWaveform);
  }, [file.file_id, file.waveform_count, index]);

  useEffect(() => {
    if (!showFFT || file.waveform_count === 0) return;
    getWaveformFFT(file.file_id, index).then(setFft);
  }, [file.file_id, file.waveform_count, index, showFFT]);

  useEffect(() => {
    if (overlayCount <= 1 || file.waveform_count === 0) {
      setOverlayData([]);
      return;
    }
    const maxIdx = Math.min(overlayCount, file.waveform_count);
    Promise.all(
      Array.from({ length: maxIdx }, (_, i) => getWaveform(file.file_id, i))
    ).then(setOverlayData);
  }, [file.file_id, file.waveform_count, overlayCount]);

  if (file.waveform_count === 0) {
    return <div className="empty-state">该文件没有波形数据</div>;
  }

  const waveChartData = waveform
    ? waveform.time_array.map((t, i) => ({ t: Number(t.toFixed(2)), v: waveform.voltage_array[i] }))
    : [];

  const fftChartData = fft
    ? fft.frequencies.map((f, i) => ({ freq: Number((f / 1000).toFixed(2)), mag: fft.magnitudes[i] }))
    : [];

  const overlayChartData = overlayData.length > 0
    ? overlayData[0].time_array.map((t, ti) => {
        const point: Record<string, number> = { t: Number(t.toFixed(2)) };
        overlayData.forEach((w, wi) => {
          point[`v${wi}`] = w.voltage_array[ti] ?? 0;
        });
        return point;
      })
    : [];

  return (
    <div className="waveform-viewer">
      <div className="waveform-controls">
        <div className="control-group">
          <label>波形索引:</label>
          <input
            type="number"
            min={0}
            max={file.waveform_count - 1}
            value={index}
            onChange={(e) => setIndex(Number(e.target.value))}
          />
          <span className="control-hint">/ {file.waveform_count - 1}</span>
        </div>
        <div className="control-group">
          <label>叠加数量:</label>
          <input
            type="number"
            min={1}
            max={Math.min(20, file.waveform_count)}
            value={overlayCount}
            onChange={(e) => setOverlayCount(Number(e.target.value))}
          />
        </div>
        <button
          className={`toggle-btn ${showFFT ? 'active' : ''}`}
          onClick={() => setShowFFT(!showFFT)}
        >
          FFT 频谱
        </button>
      </div>

      {waveform && (
        <div className="waveform-info">
          <span>通道: CH{waveform.channel}</span>
          <span>采样率: {(waveform.sample_rate / 1e6).toFixed(1)} MHz</span>
          <span>时间: {waveform.time.toFixed(6)} s</span>
          <span>采样点: {waveform.time_array.length}</span>
        </div>
      )}

      <div className="chart-panel">
        <h3>{overlayCount > 1 ? `波形叠加 (前${overlayCount}个)` : `波形 #${index}`}</h3>
        <ResponsiveContainer width="100%" height={300}>
          {overlayCount > 1 && overlayChartData.length > 0 ? (
            <LineChart data={overlayChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="t" stroke="#aaa" label={{ value: '时间 (μs)', position: 'insideBottom', offset: -5, fill: '#aaa' }} />
              <YAxis stroke="#aaa" label={{ value: '电压 (V)', angle: -90, position: 'insideLeft', fill: '#aaa' }} />
              <Tooltip contentStyle={{ background: '#1e1e2e', border: '1px solid #333' }} />
              {overlayData.map((_, i) => (
                <Line key={i} type="monotone" dataKey={`v${i}`} stroke={COLORS[i % COLORS.length]} dot={false} strokeWidth={1} opacity={0.6} />
              ))}
            </LineChart>
          ) : (
            <LineChart data={waveChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="t" stroke="#aaa" label={{ value: '时间 (μs)', position: 'insideBottom', offset: -5, fill: '#aaa' }} />
              <YAxis stroke="#aaa" label={{ value: '电压 (V)', angle: -90, position: 'insideLeft', fill: '#aaa' }} />
              <Tooltip contentStyle={{ background: '#1e1e2e', border: '1px solid #333' }} />
              <Line type="monotone" dataKey="v" stroke="#3b82f6" dot={false} strokeWidth={1.5} />
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>

      {showFFT && fft && (
        <div className="chart-panel">
          <h3>FFT 频谱分析 — 主频: {(fft.dominant_frequency / 1000).toFixed(1)} kHz</h3>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={fftChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="freq" stroke="#aaa" label={{ value: '频率 (kHz)', position: 'insideBottom', offset: -5, fill: '#aaa' }} />
              <YAxis stroke="#aaa" label={{ value: '幅值', angle: -90, position: 'insideLeft', fill: '#aaa' }} />
              <Tooltip contentStyle={{ background: '#1e1e2e', border: '1px solid #333' }} />
              <Area type="monotone" dataKey="mag" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
