import { useState, useCallback } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, Legend,
} from 'recharts';
import { runClustering } from '../services/api';
import type { FileInfo, ClusterResult, ClusterFeatureStats } from '../types';

interface Props { file: FileInfo; }

const CLUSTER_COLORS = [
  '#22d3ee', '#a78bfa', '#34d399', '#fb923c', '#f472b6',
  '#facc15', '#38bdf8', '#c084fc', '#4ade80', '#f87171',
];
const NOISE_COLOR = '#475569';

const AVAILABLE_FEATURES = [
  { key: 'amplitude', label: 'Amplitude (dB)' },
  { key: 'energy', label: 'Energy' },
  { key: 'duration', label: 'Duration (μs)' },
  { key: 'rise', label: 'Rise Time (μs)' },
  { key: 'counts', label: 'Counts' },
  { key: 'peak_counts', label: 'Peak Counts' },
  { key: 'avg_frequency', label: 'Avg Frequency (kHz)' },
  { key: 'peak_frequency', label: 'Peak Frequency (kHz)' },
  { key: 'freq_centroid', label: 'Freq Centroid (kHz)' },
  { key: 'rms', label: 'RMS' },
  { key: 'abs_energy', label: 'Abs Energy' },
  { key: 'signal_strength', label: 'Signal Strength' },
  { key: 'init_frequency', label: 'Init Frequency (kHz)' },
  { key: 'rev_frequency', label: 'Rev Frequency (kHz)' },
  { key: 'entropy', label: 'Waveform Entropy (nats)' },
];

export default function ClusterView({ file }: Props) {
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([
    'amplitude', 'energy', 'duration', 'avg_frequency', 'peak_frequency',
  ]);
  const [algorithm, setAlgorithm] = useState<'kmeans' | 'dbscan' | 'gmm'>('kmeans');
  const [nClusters, setNClusters] = useState(3);
  const [eps, setEps] = useState(0.5);
  const [minSamples, setMinSamples] = useState(5);
  const [maxTreeDepth, setMaxTreeDepth] = useState(5);
  const [channel, setChannel] = useState<number | undefined>();
  const [result, setResult] = useState<ClusterResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [expandedRules, setExpandedRules] = useState(false);
  const [scatterX, setScatterX] = useState('');
  const [scatterY, setScatterY] = useState('');

  const toggleFeature = (f: string) => {
    setSelectedFeatures(prev =>
      prev.includes(f) ? prev.filter(x => x !== f) : [...prev, f]
    );
  };

  const compute = useCallback(async () => {
    if (selectedFeatures.length < 2) return;
    setLoading(true);
    try {
      const r = await runClustering(file.file_id, {
        features: selectedFeatures,
        algorithm,
        n_clusters: nClusters,
        eps,
        min_samples: minSamples,
        max_tree_depth: maxTreeDepth,
        channel,
      });
      setResult(r);
      setScatterX(r.scatter_x);
      setScatterY(r.scatter_y);
    } finally {
      setLoading(false);
    }
  }, [file.file_id, selectedFeatures, algorithm, nClusters, eps, minSamples, maxTreeDepth, channel]);

  const scatterByCluster = result ? (() => {
    const groups: Record<number, { x: number; y: number; cluster: number }[]> = {};
    for (const pt of result.scatter_data) {
      if (!groups[pt.cluster]) groups[pt.cluster] = [];
      groups[pt.cluster].push(pt);
    }
    return groups;
  })() : {};

  const importanceData = result?.tree_feature_importance.map(f => ({
    name: f.feature,
    importance: Number((f.importance * 100).toFixed(1)),
  })) ?? [];

  const clusterDistData = result?.cluster_stats.map(s => ({
    name: `C${s.label}`,
    count: s.count,
    percentage: s.percentage,
    label: s.label,
  })) ?? [];

  return (
    <div className="view-wavelet">
      {/* Config Panel */}
      <div className="panel-grid-2">
        <div className="panel" style={{ maxHeight: 420, overflowY: 'auto' }}>
          <div className="panel-head">Feature Selection</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, padding: '8px 0' }}>
            {AVAILABLE_FEATURES.map(({ key, label }) => (
              <button
                key={key}
                className={`format-btn ${selectedFeatures.includes(key) ? 'active' : ''}`}
                onClick={() => toggleFeature(key)}
                style={{ minWidth: 'auto', padding: '4px 10px', flex: 'none' }}
              >
                <span className="format-label" style={{ fontSize: 11 }}>{label}</span>
              </button>
            ))}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 4 }}>
            {selectedFeatures.length} features selected (min 2)
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">Algorithm Settings</div>
          <div className="wf-controls" style={{ flexDirection: 'column', gap: 8 }}>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
              <div className="ctrl-group">
                <label>Algorithm</label>
                <select value={algorithm} onChange={e => setAlgorithm(e.target.value as typeof algorithm)}
                  style={{ background: 'var(--bg-1)', border: '1px solid var(--border)', color: 'var(--text-1)', padding: '3px 8px', borderRadius: 4, fontSize: 12 }}>
                  <option value="kmeans">K-Means</option>
                  <option value="gmm">GMM</option>
                  <option value="dbscan">DBSCAN</option>
                </select>
              </div>
              {algorithm !== 'dbscan' && (
                <div className="ctrl-group">
                  <label>Clusters</label>
                  <input type="number" min={2} max={10} value={nClusters}
                    onChange={e => setNClusters(Number(e.target.value))} style={{ width: 50 }} />
                </div>
              )}
              {algorithm === 'dbscan' && (
                <>
                  <div className="ctrl-group">
                    <label>Eps</label>
                    <input type="number" min={0.1} step={0.1} value={eps}
                      onChange={e => setEps(Number(e.target.value))} style={{ width: 60 }} />
                  </div>
                  <div className="ctrl-group">
                    <label>Min Pts</label>
                    <input type="number" min={2} value={minSamples}
                      onChange={e => setMinSamples(Number(e.target.value))} style={{ width: 50 }} />
                  </div>
                </>
              )}
              <div className="ctrl-group">
                <label>Tree Depth</label>
                <input type="number" min={2} max={10} value={maxTreeDepth}
                  onChange={e => setMaxTreeDepth(Number(e.target.value))} style={{ width: 50 }} />
              </div>
              <div className="ctrl-group">
                <label>Channel</label>
                <select value={channel ?? ''} onChange={e => setChannel(e.target.value ? Number(e.target.value) : undefined)}
                  style={{ background: 'var(--bg-1)', border: '1px solid var(--border)', color: 'var(--text-1)', padding: '3px 8px', borderRadius: 4, fontSize: 12 }}>
                  <option value="">All</option>
                  {file.channels.map(ch => <option key={ch} value={ch}>CH{ch}</option>)}
                </select>
              </div>
            </div>
            <button className="ctrl-btn" onClick={compute} disabled={loading || selectedFeatures.length < 2}
              style={{ alignSelf: 'flex-start' }}>
              {loading ? 'Computing...' : 'Run Clustering'}
            </button>
          </div>
        </div>
      </div>

      {loading && <div className="loading-indicator">Running clustering analysis...</div>}

      {result && (
        <>
          {/* Metrics Row */}
          <div className="metrics-row compact">
            <div className="metric">
              <span className="metric-val">{result.n_clusters}</span>
              <span className="metric-key">Clusters</span>
            </div>
            <div className="metric">
              <span className="metric-val">{result.total_points.toLocaleString()}</span>
              <span className="metric-key">Points</span>
            </div>
            {result.noise_points > 0 && (
              <div className="metric">
                <span className="metric-val">{result.noise_points}</span>
                <span className="metric-key">Noise</span>
              </div>
            )}
            {result.metrics.silhouette !== undefined && (
              <div className="metric">
                <span className="metric-val">{result.metrics.silhouette.toFixed(3)}</span>
                <span className="metric-key">Silhouette</span>
              </div>
            )}
            {result.metrics.davies_bouldin !== undefined && (
              <div className="metric">
                <span className="metric-val">{result.metrics.davies_bouldin.toFixed(3)}</span>
                <span className="metric-key">Davies-Bouldin</span>
              </div>
            )}
            <div className="metric">
              <span className="metric-val">{(result.tree_accuracy * 100).toFixed(1)}%</span>
              <span className="metric-key">Tree Accuracy</span>
            </div>
          </div>

          {/* Scatter + Distribution */}
          <div className="panel-grid-2">
            <div className="panel">
              <div className="panel-head">
                Cluster Scatter
                <span className="panel-tag">
                  <select value={scatterX} onChange={e => setScatterX(e.target.value)} style={{ background: 'transparent', border: 'none', color: 'var(--text-2)', fontSize: 11, fontFamily: 'var(--font-data)' }}>
                    {result.features.map(f => <option key={f} value={f}>{f}</option>)}
                  </select>
                  {' vs '}
                  <select value={scatterY} onChange={e => setScatterY(e.target.value)} style={{ background: 'transparent', border: 'none', color: 'var(--text-2)', fontSize: 11, fontFamily: 'var(--font-data)' }}>
                    {result.features.map(f => <option key={f} value={f}>{f}</option>)}
                  </select>
                </span>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="x" type="number" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false}
                    label={{ value: scatterX, fill: '#64748b', fontSize: 11, position: 'insideBottom', offset: -4 }} />
                  <YAxis dataKey="y" type="number" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false}
                    label={{ value: scatterY, fill: '#64748b', fontSize: 11, angle: -90, position: 'insideLeft' }} />
                  <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }}
                    formatter={(value: number) => value.toFixed(2)} />
                  <Legend />
                  {Object.entries(scatterByCluster).map(([clusterStr, points]) => {
                    const cluster = Number(clusterStr);
                    return (
                      <Scatter
                        key={cluster}
                        name={cluster === -1 ? 'Noise' : `Cluster ${cluster}`}
                        data={points}
                        fill={cluster === -1 ? NOISE_COLOR : CLUSTER_COLORS[cluster % CLUSTER_COLORS.length]}
                        opacity={cluster === -1 ? 0.3 : 0.7}
                        r={cluster === -1 ? 2 : 3}
                      />
                    );
                  })}
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            <div className="panel">
              <div className="panel-head">Cluster Distribution</div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={clusterDistData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} />
                  <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }}
                    formatter={(v: number, _n: string, entry: { payload: { percentage: number } }) => [`${v} (${entry.payload.percentage}%)`, 'Count']} />
                  <Bar dataKey="count">
                    {clusterDistData.map((d, i) => (
                      <Cell key={i} fill={CLUSTER_COLORS[d.label % CLUSTER_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>

              {/* Feature Importance */}
              {importanceData.length > 0 && (
                <>
                  <div className="panel-head" style={{ marginTop: 8 }}>Feature Importance (Decision Tree)</div>
                  <ResponsiveContainer width="100%" height={Math.max(120, importanceData.length * 28)}>
                    <BarChart data={importanceData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                      <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false}
                        label={{ value: '%', fill: '#64748b', fontSize: 11, position: 'insideRight' }} />
                      <YAxis dataKey="name" type="category" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} width={100} />
                      <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }} />
                      <Bar dataKey="importance" fill="#a78bfa" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </>
              )}
            </div>
          </div>

          {/* Decision Tree Rules */}
          {result.tree_rules.length > 0 && (
            <div className="panel">
              <div className="panel-head">
                Decision Tree Rules
                <span className="panel-tag">
                  {result.tree_rules.length} rules · Accuracy: {(result.tree_accuracy * 100).toFixed(1)}%
                </span>
                <button className="ctrl-btn" style={{ marginLeft: 8, padding: '2px 8px', fontSize: 10 }}
                  onClick={() => setExpandedRules(!expandedRules)}>
                  {expandedRules ? 'Show Top 10' : 'Show All'}
                </button>
              </div>
              <div style={{ maxHeight: 400, overflowY: 'auto' }}>
                <table className="hit-table" style={{ fontSize: 11 }}>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Rule</th>
                      <th>Cluster</th>
                      <th>Samples</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(expandedRules ? result.tree_rules : result.tree_rules.slice(0, 10)).map((rule, i) => (
                      <tr key={i}>
                        <td>{i + 1}</td>
                        <td style={{ fontFamily: 'var(--font-data)', fontSize: 10, whiteSpace: 'pre-wrap', maxWidth: 500 }}>
                          {rule.conditions.map((c, ci) => (
                            <span key={ci}>
                              {ci > 0 && <span style={{ color: 'var(--text-3)' }}> AND </span>}
                              <span style={{ color: '#a78bfa' }}>{c.feature}</span>
                              <span style={{ color: 'var(--text-3)' }}> {c.op} </span>
                              <span style={{ color: '#22d3ee' }}>{c.value}</span>
                            </span>
                          ))}
                        </td>
                        <td>
                          <span style={{
                            display: 'inline-block', padding: '1px 8px', borderRadius: 10,
                            background: CLUSTER_COLORS[rule.cluster % CLUSTER_COLORS.length] + '30',
                            color: CLUSTER_COLORS[rule.cluster % CLUSTER_COLORS.length],
                            fontWeight: 600, fontSize: 11,
                          }}>
                            C{rule.cluster}
                          </span>
                        </td>
                        <td>{rule.samples}</td>
                        <td>{(rule.confidence * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Cluster Statistics */}
          <div className="panel">
            <div className="panel-head">
              Cluster Feature Statistics
              <span className="panel-tag">Mean ± Std</span>
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="hit-table" style={{ fontSize: 11 }}>
                <thead>
                  <tr>
                    <th>Cluster</th>
                    <th>Count</th>
                    <th>%</th>
                    {result.features.map(f => (
                      <th key={f}>{f}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.cluster_stats.map((stat) => (
                    <tr key={stat.label}>
                      <td>
                        <span style={{
                          display: 'inline-block', padding: '1px 8px', borderRadius: 10,
                          background: CLUSTER_COLORS[stat.label % CLUSTER_COLORS.length] + '30',
                          color: CLUSTER_COLORS[stat.label % CLUSTER_COLORS.length],
                          fontWeight: 600,
                        }}>
                          C{stat.label}
                        </span>
                      </td>
                      <td>{stat.count.toLocaleString()}</td>
                      <td>{stat.percentage}%</td>
                      {result.features.map(f => {
                        const fstat = stat[f] as ClusterFeatureStats | undefined;
                        return (
                          <td key={f} style={{ fontFamily: 'var(--font-data)' }}>
                            {fstat ? `${fstat.mean.toFixed(1)} ± ${fstat.std.toFixed(1)}` : '-'}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
