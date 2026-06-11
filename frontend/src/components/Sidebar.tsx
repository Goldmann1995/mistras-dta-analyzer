import { FileText, BarChart3, Activity, Table2, Layers, Plug } from 'lucide-react';
import type { FileInfo } from '../types';

export type ViewType = 'dashboard' | 'hits' | 'waveform' | 'charts' | 'channels' | 'plugins';

interface Props {
  activeFile: FileInfo | null;
  activeView: ViewType;
  onViewChange: (view: ViewType) => void;
}

const navItems: { view: ViewType; label: string; icon: React.ReactNode }[] = [
  { view: 'dashboard', label: '总览', icon: <BarChart3 size={18} /> },
  { view: 'hits', label: '事件表', icon: <Table2 size={18} /> },
  { view: 'waveform', label: '波形', icon: <Activity size={18} /> },
  { view: 'charts', label: '参数图', icon: <Layers size={18} /> },
  { view: 'channels', label: '通道', icon: <BarChart3 size={18} /> },
  { view: 'plugins', label: '插件', icon: <Plug size={18} /> },
];

export default function Sidebar({ activeFile, activeView, onViewChange }: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <Activity size={22} />
        <span>Mistras DTA</span>
      </div>

      {activeFile && (
        <div className="sidebar-file-info">
          <FileText size={14} />
          <div>
            <div className="file-name">{activeFile.filename}</div>
            <div className="file-meta">
              {activeFile.hit_count} hits · {activeFile.channels.length} ch
            </div>
          </div>
        </div>
      )}

      <nav className="sidebar-nav">
        {navItems.map(({ view, label, icon }) => (
          <button
            key={view}
            className={`nav-item ${activeView === view ? 'active' : ''}`}
            onClick={() => onViewChange(view)}
            disabled={!activeFile && view !== 'dashboard'}
          >
            {icon}
            <span>{label}</span>
          </button>
        ))}
      </nav>
    </aside>
  );
}
