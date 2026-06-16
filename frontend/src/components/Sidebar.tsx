import type { FileInfo } from '../types';

export type ViewType = 'dashboard' | 'hits' | 'waveform' | 'filter' | 'wavelet' | 'charts' | 'cluster' | 'deepcluster' | 'channels' | 'sensor' | 'export' | 'plugins';

interface Props {
  activeFile: FileInfo | null;
  activeView: ViewType;
  onViewChange: (view: ViewType) => void;
}

const navItems: { view: ViewType; label: string; code: string }[] = [
  { view: 'dashboard', label: 'Overview', code: 'OVR' },
  { view: 'hits', label: 'Event Log', code: 'EVT' },
  { view: 'waveform', label: 'Waveform', code: 'WFM' },
  { view: 'filter', label: 'Filter', code: 'FLT' },
  { view: 'wavelet', label: 'Wavelet', code: 'CWT' },
  { view: 'charts', label: 'Analysis', code: 'ANL' },
  { view: 'cluster', label: 'Clustering', code: 'CLU' },
  { view: 'deepcluster', label: 'Deep Cluster', code: 'DCL' },
  { view: 'channels', label: 'Channels', code: 'CHN' },
  { view: 'sensor', label: 'Sensor 3D', code: 'LOC' },
  { view: 'export', label: 'Export', code: 'EXP' },
  { view: 'plugins', label: 'Plugins', code: 'PLG' },
];

export default function Sidebar({ activeFile, activeView, onViewChange }: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="2,12 6,8 10,16 14,4 18,14 22,10" />
          </svg>
        </div>
        <div className="brand-text">
          <span className="brand-name">MISTRAS</span>
          <span className="brand-sub">DTA Analyzer</span>
        </div>
      </div>

      {activeFile && (
        <div className="sidebar-file">
          <div className="file-indicator" />
          <div className="file-details">
            <span className="file-label">{activeFile.filename}</span>
            <span className="file-stats">
              {activeFile.hit_count.toLocaleString()} events · {activeFile.channels.length} ch · {activeFile.duration.toFixed(0)}s
            </span>
          </div>
        </div>
      )}

      <nav className="sidebar-nav">
        {navItems.map(({ view, label, code }) => (
          <button
            key={view}
            className={`nav-btn ${activeView === view ? 'active' : ''}`}
            onClick={() => onViewChange(view)}
            disabled={!activeFile && view !== 'dashboard'}
          >
            <span className="nav-code">{code}</span>
            <span className="nav-label">{label}</span>
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <span>v1.0.0</span>
      </div>
    </aside>
  );
}
