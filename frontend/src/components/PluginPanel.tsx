import { useState, useEffect } from 'react';
import { Plug, ExternalLink } from 'lucide-react';
import { getPlugins } from '../services/api';
import type { PluginInfo } from '../types';

export default function PluginPanel() {
  const [plugins, setPlugins] = useState<PluginInfo[]>([]);

  useEffect(() => {
    getPlugins().then(setPlugins).catch(() => {});
  }, []);

  return (
    <div className="plugin-panel">
      <h2>分析插件</h2>
      <p className="plugin-desc">
        通过插件扩展分析能力。支持神经算子 (DeepONet, FNO)、信号处理算法等。
      </p>

      {plugins.length === 0 ? (
        <div className="empty-state">
          <Plug size={48} />
          <p>暂无已注册插件</p>
          <p className="hint">在 backend/app/plugins/ 目录下添加插件文件</p>
        </div>
      ) : (
        <div className="plugin-list">
          {plugins.map((p) => (
            <div key={p.name} className="plugin-card">
              <div className="plugin-header">
                <Plug size={18} />
                <h3>{p.name}</h3>
                <span className="plugin-version">v{p.version}</span>
              </div>
              <p>{p.description}</p>
              <div className="plugin-endpoints">
                <h4>接口:</h4>
                {p.endpoints.map((ep, i) => (
                  <code key={i}>{ep}</code>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="plugin-guide">
        <h3>开发插件</h3>
        <pre>{`from backend.app.services.plugin_manager import PluginBase

class MyPlugin(PluginBase):
    name = "my_plugin"
    version = "1.0.0"
    description = "自定义分析插件"

    def infer(self, file_id, params):
        # 实现推理逻辑
        return {"result": ...}

def register(manager):
    manager.register(MyPlugin())`}</pre>
      </div>
    </div>
  );
}
