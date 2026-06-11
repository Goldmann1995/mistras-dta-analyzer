import { useState } from 'react';
import Sidebar from './components/Sidebar';
import type { ViewType } from './components/Sidebar';
import FileUpload from './components/FileUpload';
import Dashboard from './components/Dashboard';
import HitTable from './components/HitTable';
import WaveformViewer from './components/WaveformViewer';
import ParametricCharts from './components/ParametricCharts';
import ChannelView from './components/ChannelView';
import ExportPanel from './components/ExportPanel';
import PluginPanel from './components/PluginPanel';
import WaveletView from './components/WaveletView';
import type { FileInfo } from './types';
import './App.css';

function App() {
  const [file, setFile] = useState<FileInfo | null>(null);
  const [view, setView] = useState<ViewType>('dashboard');
  const [wfIdx, setWfIdx] = useState<number | undefined>();

  const onHitSelect = (i: number) => { setWfIdx(i); setView('waveform'); };

  const renderView = () => {
    if (!file) {
      return (
        <div className="welcome">
          <div className="welcome-inner">
            <svg className="welcome-wave" width="64" height="40" viewBox="0 0 64 40" fill="none" stroke="currentColor" strokeWidth="2" opacity={0.3}>
              <polyline points="0,20 8,10 16,30 24,5 32,35 40,15 48,25 56,8 64,20" />
            </svg>
            <h1>Mistras DTA Analyzer</h1>
            <p>Acoustic Emission Data Analysis Platform</p>
            <FileUpload onFileLoaded={(f) => { setFile(f); setView('dashboard'); }} />
          </div>
        </div>
      );
    }

    switch (view) {
      case 'dashboard': return <Dashboard file={file} />;
      case 'hits': return <HitTable file={file} onSelectHit={onHitSelect} />;
      case 'waveform': return <WaveformViewer file={file} initialIndex={wfIdx} />;
      case 'wavelet': return <WaveletView file={file} />;
      case 'charts': return <ParametricCharts file={file} />;
      case 'channels': return <ChannelView file={file} />;
      case 'export': return <ExportPanel file={file} />;
      case 'plugins': return <PluginPanel />;
    }
  };

  return (
    <div className="app">
      <Sidebar activeFile={file} activeView={view} onViewChange={setView} />
      <main className="main">
        {file && (
          <header className="topbar">
            <FileUpload compact onFileLoaded={(f) => { setFile(f); setView('dashboard'); }} />
          </header>
        )}
        <div className="content">{renderView()}</div>
      </main>
    </div>
  );
}

export default App;
