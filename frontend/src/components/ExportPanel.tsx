import { useState } from 'react';
import { getExportUrl } from '../services/api';
import type { FileInfo, ExportOptions } from '../types';

interface Props { file: FileInfo; }

export default function ExportPanel({ file }: Props) {
  const [opts, setOpts] = useState<ExportOptions>({
    keep_pretrigger: false,
    normalize: false,
    format: 'npz',
  });

  const handleExport = () => {
    const url = getExportUrl(file.file_id, opts);
    const a = document.createElement('a');
    a.href = url;
    a.download = '';
    a.click();
  };

  const isCsv = opts.format === 'csv';

  return (
    <div className="view-export">
      <div className="export-header">
        <h2>Data Export</h2>
        <p>Export waveform and hit data for analysis and neural network training</p>
      </div>

      <div className="export-form">
        <div className="export-section">
          <h3>Export Format</h3>
          <div className="export-format-btns">
            {([
              { fmt: 'npz' as const, label: 'NumPy (.npz)', desc: 'Compressed NumPy arrays for Python/ML' },
              { fmt: 'mat' as const, label: 'MATLAB (.mat)', desc: 'HDF5-compatible for MATLAB workspace' },
              { fmt: 'csv' as const, label: 'CSV (.csv)', desc: 'Hit records as spreadsheet data' },
            ]).map(({ fmt, label, desc }) => (
              <button
                key={fmt}
                className={`format-btn ${opts.format === fmt ? 'active' : ''}`}
                onClick={() => setOpts({ ...opts, format: fmt })}
              >
                <span className="format-label">{label}</span>
                <span className="format-desc">{desc}</span>
              </button>
            ))}
          </div>
        </div>

        {!isCsv && (
          <div className="export-section">
            <h3>Output Structure</h3>
            <div className="export-preview">
              <code>
                waveforms: (N, L) float{opts.format === 'mat' ? '64' : '32'} — voltage arrays{'\n'}
                times: (N,) float64 — event timestamps{'\n'}
                channels: (N,) int32 — channel IDs{'\n'}
                sample_rates: (N,) float64 — sampling rates{'\n'}
                amplitudes: (N,) float{opts.format === 'mat' ? '64' : '32'} — hit amplitudes{'\n'}
                energies: (N,) float{opts.format === 'mat' ? '64' : '32'} — hit energies{'\n'}
                durations: (N,) float{opts.format === 'mat' ? '64' : '32'} — event durations
              </code>
            </div>
          </div>
        )}

        {isCsv && (
          <div className="export-section">
            <h3>Output Columns</h3>
            <div className="export-preview">
              <code>
                Time_s, Channel, Rise_us, Counts, Energy,{'\n'}
                Duration_us, Amplitude_dB, ASL, Threshold,{'\n'}
                Avg_Frequency_kHz, RMS, Signal_Strength,{'\n'}
                Abs_Energy, Freq_Centroid_kHz, Peak_Frequency_kHz
              </code>
            </div>
          </div>
        )}

        <div className="export-section">
          <h3>Configuration</h3>

          <div className="export-field">
            <label>Channel</label>
            <select className="filter-select" value={opts.channel ?? ''} onChange={e => setOpts({ ...opts, channel: e.target.value ? Number(e.target.value) : undefined })}>
              <option value="">All channels</option>
              {file.channels.map(ch => <option key={ch} value={ch}>CH{ch}</option>)}
            </select>
          </div>

          {!isCsv && (
            <>
              <div className="export-field">
                <label>Max waveforms</label>
                <input type="number" className="filter-input" placeholder="All" min={1} value={opts.max_waveforms ?? ''} onChange={e => setOpts({ ...opts, max_waveforms: e.target.value ? Number(e.target.value) : undefined })} />
              </div>

              <div className="export-field">
                <label>Fixed length (samples)</label>
                <input type="number" className="filter-input" placeholder="Auto (pad to max)" min={1} value={opts.fixed_length ?? ''} onChange={e => setOpts({ ...opts, fixed_length: e.target.value ? Number(e.target.value) : undefined })} />
              </div>

              <div className="export-toggles">
                <label className="ctrl-toggle">
                  <input type="checkbox" checked={opts.normalize} onChange={e => setOpts({ ...opts, normalize: e.target.checked })} />
                  <span>Normalize to [-1, 1]</span>
                </label>
                <label className="ctrl-toggle">
                  <input type="checkbox" checked={opts.keep_pretrigger} onChange={e => setOpts({ ...opts, keep_pretrigger: e.target.checked })} />
                  <span>Keep pre-trigger data</span>
                </label>
              </div>
            </>
          )}
        </div>

        <div className="export-section">
          <h3>Summary</h3>
          <div className="export-summary">
            <span>Source: {file.filename}</span>
            <span>Format: {opts.format === 'npz' ? 'NumPy (.npz)' : opts.format === 'mat' ? 'MATLAB (.mat)' : 'CSV (.csv)'}</span>
            {isCsv ? (
              <span>Records: {file.hit_count.toLocaleString()} hits</span>
            ) : (
              <span>Available: {file.waveform_count.toLocaleString()} waveforms</span>
            )}
            <span>Channels: {opts.channel ? `CH${opts.channel}` : `All (${file.channels.map(c => `CH${c}`).join(', ')})`}</span>
            {!isCsv && <span>Pre-trigger: {opts.keep_pretrigger ? 'included' : 'trimmed'}</span>}
            {!isCsv && <span>Normalize: {opts.normalize ? 'yes' : 'no'}</span>}
            {!isCsv && opts.fixed_length && <span>Fixed length: {opts.fixed_length} samples</span>}
          </div>
        </div>

        <button className="export-btn" onClick={handleExport}>
          Download .{opts.format || 'npz'}
        </button>
      </div>
    </div>
  );
}
