import { useState, useEffect } from 'react';
import { getPlugins } from '../services/api';
import type { PluginInfo } from '../types';

export default function PluginPanel() {
  const [plugins, setPlugins] = useState<PluginInfo[]>([]);
  useEffect(() => { getPlugins().then(setPlugins).catch(() => {}); }, []);

  return (
    <div className="view-plugins">
      <h2>Analysis Plugins</h2>
      <p className="view-desc">Extend analysis with neural operators (DeepONet, FNO), signal processing, and custom models.</p>

      {plugins.length === 0 ? (
        <div className="empty-state">
          <span>No plugins registered</span>
          <small>Add plugin files to backend/app/plugins/</small>
        </div>
      ) : (
        <div className="plugin-list">
          {plugins.map(p => (
            <div key={p.name} className="plugin-card">
              <div className="plugin-card-head">
                <h3>{p.name}</h3>
                <span className="plugin-ver">v{p.version}</span>
              </div>
              <p>{p.description}</p>
              <div className="plugin-eps">
                {p.endpoints.map((ep, i) => <code key={i}>{ep}</code>)}
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="panel" style={{ marginTop: 24 }}>
        <div className="panel-head">Plugin Development</div>
        <pre className="code-block">{`from backend.app.services.plugin_manager import PluginBase

class MyPlugin(PluginBase):
    name = "my_model"
    version = "1.0.0"
    description = "Custom analysis plugin"

    def infer(self, file_id, params):
        # Load waveform data, run inference
        return {"prediction": ...}

def register(manager):
    manager.register(MyPlugin())`}</pre>
      </div>
    </div>
  );
}
