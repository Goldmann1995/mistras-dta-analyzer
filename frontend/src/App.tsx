import { useState } from 'react';
import Sidebar from './components/Sidebar';
import type { ViewType } from './components/Sidebar';
import FileUpload from './components/FileUpload';
import Dashboard from './components/Dashboard';
import HitTable from './components/HitTable';
import WaveformViewer from './components/WaveformViewer';
import ParametricCharts from './components/ParametricCharts';
import ChannelView from './components/ChannelView';
import PluginPanel from './components/PluginPanel';
import type { FileInfo } from './types';
import './App.css';

function App() {
  const [activeFile, setActiveFile] = useState<FileInfo | null>(null);
  const [activeView, setActiveView] = useState<ViewType>('dashboard');
  const [waveformIndex, setWaveformIndex] = useState<number | undefined>();

  const handleSelectHit = (index: number) => {
    setWaveformIndex(index);
    setActiveView('waveform');
  };

  const renderView = () => {
    if (!activeFile) {
      return (
        <div className="welcome">
          <h1>Mistras DTA Analyzer</h1>
          <p>声发射数据分析平台</p>
          <FileUpload onFileLoaded={(f) => { setActiveFile(f); setActiveView('dashboard'); }} />
        </div>
      );
    }

    switch (activeView) {
      case 'dashboard':
        return <Dashboard file={activeFile} />;
      case 'hits':
        return <HitTable file={activeFile} onSelectHit={handleSelectHit} />;
      case 'waveform':
        return <WaveformViewer file={activeFile} initialIndex={waveformIndex} />;
      case 'charts':
        return <ParametricCharts file={activeFile} />;
      case 'channels':
        return <ChannelView file={activeFile} />;
      case 'plugins':
        return <PluginPanel />;
      default:
        return null;
    }
  };

  return (
    <div className="app">
      <Sidebar activeFile={activeFile} activeView={activeView} onViewChange={setActiveView} />
      <main className="main-content">
        <header className="top-bar">
          <div className="top-bar-left">
            {activeFile && (
              <FileUpload onFileLoaded={(f) => { setActiveFile(f); setActiveView('dashboard'); }} />
            )}
          </div>
          <div className="top-bar-right">
            <span className="app-version">v1.0.0</span>
          </div>
        </header>
        <div className="content-area">
          {renderView()}
        </div>
      </main>
    </div>
  );
}

export default App;
