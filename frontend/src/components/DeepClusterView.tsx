import { useState, useCallback } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, Legend,
} from 'recharts';
import { runDeepClustering } from '../services/api';
import type { FileInfo, DeepClusterResult } from '../types';

interface Props { file: FileInfo; }

const CLUSTER_COLORS = [
  '#22d3ee', '#a78bfa', '#34d399', '#fb923c', '#f472b6',
  '#facc15', '#38bdf8', '#c084fc', '#4ade80', '#f87171',
];
const NOISE_COLOR = '#475569';

export default function DeepClusterView({ file }: Props) {
  const [model, setModel] = useState<'cae' | 'vae'>('cae');
  const [latentDim, setLatentDim] = useState(16);
  const [epochs, setEpochs] = useState(40);
  const [fixedLength, setFixedLength] = useState(1024);
  const [maxWaveforms, setMaxWaveforms] = useState(2000);
  const [algorithm, setAlgorithm] = useState<'kmeans' | 'gmm' | 'hdbscan' | 'dbscan'>('kmeans');
  const [nClusters, setNClusters] = useState(4);
  const [minSamples, setMinSamples] = useState(10);
  const [eps, setEps] = useState(0.8);
  const [projection, setProjection] = useState<'pca' | 'tsne'>('pca');
  const [channel, setChannel] = useState<number | undefined>();
  const [result, setResult] = useState<DeepClusterResult | null>(null);
  const [loading, setLoading] = useState(false);

  const compute = useCallback(async () => {
    setLoading(true);
    try {
      const r = await runDeepClustering(file.file_id, {
        model, latent_dim: latentDim, epochs,
        fixed_length: fixedLength, max_waveforms: maxWaveforms,
        algorithm, n_clusters: nClusters, min_samples: minSamples, eps,
        projection, channel,
      });
      setResult(r);
    } finally {
      setLoading(false);
    }
  }, [file.file_id, model, latentDim, epochs, fixedLength, maxWaveforms, algorithm, nClusters, minSamples, eps, projection, channel]);

  const scatterByCluster = result ? (() => {
    const groups: Record<number, { x: number; y: number }[]> = {};
    for (const pt of result.scatter_data) {
      if (!groups[pt.cluster]) groups[pt.cluster] = [];
      groups[pt.cluster].push(pt);
    }
    return groups;
  })() : {};

  const lossData = result?.loss_curve.map((l, i) => ({ epoch: i + 1, loss: l })) ?? [];

  if (file.waveform_count === 0) return <div className="empty-state">No waveform data in this file</div>;

  return (
    <div className="view-wavelet">
      <div className="panel">
        <div className="panel-head">
          Deep Latent-Space Clustering
          <span className="panel-tag">Autoencoder → latent space → clustering</span>
        </div>
        <div className="wf-controls" style={{ flexWrap: 'wrap', gap: 10 }}>
          <div className="ctrl-group">
            <label>Model</label>
            <select value={model} onChange={e => setModel(e.target.value as typeof model)}
              style={{ background: 'var(--bg-1)', border: '1px solid var(--border)', color: 'var(--text-1)', padding: '3px 8px', borderRadius: 4, fontSize: 12 }}>
              <option value="cae">CAE (Conv Autoencoder)</option>
              <option value="vae">VAE (Variational)</option>
            </select>
          </div>
          <div className="ctrl-group">
            <label>Latent Dim</label>
            <input type="number" min={2} max={128} value={latentDim}
              onChange={e => setLatentDim(Number(e.target.value))} style={{ width: 55 }} />
          </div>
          <div className="ctrl-group">
            <label>Epochs</label>
            <input type="number" min={5} max={300} value={epochs}
              onChange={e => setEpochs(Number(e.target.value))} style={{ width: 55 }} />
          </div>
          <div className="ctrl-group">
            <label>Wave Length</label>
            <input type="number" min={64} step={64} value={fixedLength}
              onChange={e => setFixedLength(Number(e.target.value))} style={{ width: 65 }} />
          </div>
          <div className="ctrl-group">
            <label>Max Waves</label>
            <input type="number" min={50} step={100} value={maxWaveforms}
              onChange={e => setMaxWaveforms(Number(e.target.value))} style={{ width: 65 }} />
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
        <div className="wf-controls" style={{ flexWrap: 'wrap', gap: 10, marginTop: 8 }}>
          <div className="ctrl-group">
            <label>Cluster</label>
            <select value={algorithm} onChange={e => setAlgorithm(e.target.value as typeof algorithm)}
              style={{ background: 'var(--bg-1)', border: '1px solid var(--border)', color: 'var(--text-1)', padding: '3px 8px', borderRadius: 4, fontSize: 12 }}>
              <option value="kmeans">K-Means</option>
              <option value="gmm">GMM</option>
              <option value="hdbscan">HDBSCAN</option>
              <option value="dbscan">DBSCAN</option>
            </select>
          </div>
          {(algorithm === 'kmeans' || algorithm === 'gmm') && (
            <div className="ctrl-group">
              <label>Clusters</label>
              <input type="number" min={2} max={10} value={nClusters}
                onChange={e => setNClusters(Number(e.target.value))} style={{ width: 50 }} />
            </div>
          )}
          {(algorithm === 'hdbscan' || algorithm === 'dbscan') && (
            <div className="ctrl-group">
              <label>Min Size</label>
              <input type="number" min={2} value={minSamples}
                onChange={e => setMinSamples(Number(e.target.value))} style={{ width: 55 }} />
            </div>
          )}
          {algorithm === 'dbscan' && (
            <div className="ctrl-group">
              <label>Eps</label>
              <input type="number" min={0.1} step={0.1} value={eps}
                onChange={e => setEps(Number(e.target.value))} style={{ width: 55 }} />
            </div>
          )}
          <div className="ctrl-group">
            <label>2D Projection</label>
            <select value={projection} onChange={e => setProjection(e.target.value as typeof projection)}
              style={{ background: 'var(--bg-1)', border: '1px solid var(--border)', color: 'var(--text-1)', padding: '3px 8px', borderRadius: 4, fontSize: 12 }}>
              <option value="pca">PCA (fast)</option>
              <option value="tsne">t-SNE (slow)</option>
            </select>
          </div>
          <button className="ctrl-btn" onClick={compute} disabled={loading} style={{ alignSelf: 'center' }}>
            {loading ? 'Training & clustering...' : 'Run Deep Clustering'}
          </button>
        </div>
      </div>

      {loading && (
        <div className="loading-indicator">
          Training {model.toUpperCase()} for {epochs} epochs on up to {maxWaveforms} waveforms, then clustering latent space... (may take 30–90s on CPU)
        </div>
      )}

      {result && (
        <>
          <div className="metrics-row compact">
            <div className="metric">
              <span className="metric-val">{result.n_clusters}</span>
              <span className="metric-key">Clusters</span>
            </div>
            <div className="metric">
              <span className="metric-val">{result.n_waveforms.toLocaleString()}</span>
              <span className="metric-key">Waveforms</span>
            </div>
            <div className="metric">
              <span className="metric-val">{result.latent_dim}</span>
              <span className="metric-key">Latent Dim</span>
            </div>
            <div className="metric">
              <span className="metric-val">{result.final_loss.toExponential(2)}</span>
              <span className="metric-key">Final Loss</span>
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
          </div>

          <div className="panel-grid-2">
            <div className="panel">
              <div className="panel-head">
                Latent Space ({result.projection})
                <span className="panel-tag">{result.model.toUpperCase()} embedding</span>
              </div>
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="x" type="number" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false}
                    label={{ value: `${result.projection}-1`, fill: '#64748b', fontSize: 11, position: 'insideBottom', offset: -4 }} />
                  <YAxis dataKey="y" type="number" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false}
                    label={{ value: `${result.projection}-2`, fill: '#64748b', fontSize: 11, angle: -90, position: 'insideLeft' }} />
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
              <div className="panel-head">Training Loss</div>
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={lossData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="epoch" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false}
                    label={{ value: 'Epoch', fill: '#64748b', fontSize: 11, position: 'insideBottom', offset: -4 }} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false}
                    label={{ value: 'Loss', fill: '#64748b', fontSize: 11, angle: -90, position: 'insideLeft' }} />
                  <Tooltip contentStyle={{ background: '#0c1222', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, fontSize: 11 }}
                    formatter={(v: number) => v.toExponential(3)} />
                  <Line type="monotone" dataKey="loss" stroke="#22d3ee" dot={false} strokeWidth={1.5} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Prototype (medoid) waveforms */}
          {result.prototypes.length > 0 && (
            <div className="panel">
              <div className="panel-head">
                Cluster Prototype Waveforms (Medoid)
                <span className="panel-tag">Most representative waveform per cluster</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
                {result.prototypes.map((proto) => {
                  const color = CLUSTER_COLORS[proto.cluster % CLUSTER_COLORS.length];
                  const data = proto.time.map((t, i) => ({ t: Number(t.toFixed(2)), v: proto.waveform[i] }));
                  const stat = result.cluster_stats.find(s => s.label === proto.cluster);
                  return (
                    <div key={proto.cluster} style={{ border: `1px solid ${color}40`, borderRadius: 6, padding: 8 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                        <span style={{ color, fontWeight: 600, fontSize: 12, fontFamily: 'var(--font-data)' }}>
                          Cluster {proto.cluster}
                        </span>
                        <span style={{ fontSize: 10, color: 'var(--text-3)' }}>
                          {stat ? `${stat.count} (${stat.percentage}%)` : ''} · CH{proto.channel} · #{proto.index}
                        </span>
                      </div>
                      <ResponsiveContainer width="100%" height={120}>
                        <LineChart data={data}>
                          <XAxis dataKey="t" tick={{ fill: '#64748b', fontSize: 9 }} axisLine={false} tickLine={false} />
                          <YAxis tick={{ fill: '#64748b', fontSize: 9 }} axisLine={false} tickLine={false} width={30} />
                          <Line type="monotone" dataKey="v" stroke={color} dot={false} strokeWidth={1} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </>
      )}

      {!result && !loading && (
        <div className="panel" style={{ padding: 24 }}>
          <p style={{ color: 'var(--text-2)', margin: '0 0 8px', fontSize: 13 }}>
            This module learns an unsupervised <strong>latent representation</strong> of raw AE
            waveforms with a convolutional/variational autoencoder, then clusters the events
            in that latent space — the deep-learning analogue of image-based latent clustering.
          </p>
          <p style={{ color: 'var(--text-3)', margin: 0, fontSize: 12 }}>
            Pipeline: <code>waveform → {model.toUpperCase()} encoder → latent code → {algorithm} → 2D ({projection}) + medoid prototypes</code>
          </p>
        </div>
      )}
    </div>
  );
}
