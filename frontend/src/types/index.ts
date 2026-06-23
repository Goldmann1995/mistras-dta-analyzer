export interface FileInfo {
  filename: string;
  file_id: string;
  hit_count: number;
  waveform_count: number;
  channels: number[];
  duration: number;
  fields: string[];
}

export interface HitRecord {
  index: number;
  time: number;
  channel: number;
  rise?: number;
  counts?: number;
  peak_counts?: number;
  energy?: number;
  duration?: number;
  amplitude?: number;
  asl?: number;
  threshold?: number;
  avg_frequency?: number;
  rms?: number;
  rev_frequency?: number;
  init_frequency?: number;
  signal_strength?: number;
  abs_energy?: number;
  freq_centroid?: number;
  peak_frequency?: number;
  timestamp?: number;
  entropy?: number;
}

export interface HitsResponse {
  total: number;
  hits: HitRecord[];
}

export interface WaveformData {
  index: number;
  time: number;
  channel: number;
  sample_rate: number;
  pretrigger_samples: number;
  trimmed: boolean;
  time_array: number[];
  voltage_array: number[];
}

export interface FFTResult {
  frequencies: number[];
  magnitudes: number[];
  dominant_frequency: number;
  sample_rate: number;
}

export interface ChannelStats {
  channel: number;
  hit_count: number;
  avg_amplitude: number;
  max_amplitude: number;
  min_amplitude: number;
  avg_energy: number;
  max_energy: number;
  avg_duration: number;
  max_duration: number;
  avg_rms?: number;
  time_span: number;
}

export interface ScatterData {
  x: number[];
  y: number[];
  indices: number[];
  color?: number[];
  x_field: string;
  y_field: string;
  color_field?: string;
}

export interface HistogramData {
  edges: number[];
  counts: number[];
  field: string;
}

export interface PluginInfo {
  name: string;
  version: string;
  description: string;
  endpoints: string[];
}

export interface CWTResult {
  time_axis: number[];
  freq_axis: number[];
  power: number[][];
  peak_frequency: number;
  peak_time: number;
  wavelet: string;
  channel: number;
  sample_rate: number;
}

export interface DispersionResult {
  frequencies: number[];
  arrival_times: number[];
  peak_times: number[];
  energy_at_freq: number[];
  channel: number;
  sample_rate: number;
}

export interface VelocityPair {
  ch1: number;
  ch2: number;
  event_count: number;
  event_times: number[];
  delta_t: number[];
  velocities: number[];
  avg_velocity: number;
  std_velocity: number;
  median_velocity: number;
}

export interface GroupVelocityResult {
  sensor_distance: number;
  channel_count: number;
  channels: number[];
  pairs: VelocityPair[];
}

export interface IMFData {
  index: number;
  data: number[];
  inst_amplitude: number[];
  inst_frequency: number[];
  dominant_frequency: number;
  energy: number;
  energy_ratio: number;
}

export interface EMDResult {
  time_axis: number[];
  num_imfs: number;
  imfs: IMFData[];
  method: string;
  channel: number;
  sample_rate: number;
}

export interface LambMode {
  mode: string;
  frequencies: number[];
  phase_velocity: number[];
  group_velocity: number[];
}

export interface LambDispersionResult {
  thickness: number;
  cl: number;
  ct: number;
  freq_range: [number, number];
  modes: {
    symmetric: LambMode[];
    antisymmetric: LambMode[];
  };
}

export interface SourceEvent {
  time: number;
  channels: number[];
  arrivals: Record<number, number>;
  amplitude: number;
  energy: number;
  location: number[] | null;
  num_channels: number;
}

export interface SourceLocationResult {
  total_events: number;
  located_events: number;
  events: SourceEvent[];
  sensor_positions: Record<string, number[]>;
  velocity: number;
}

export interface ExportOptions {
  channel?: number;
  keep_pretrigger: boolean;
  normalize: boolean;
  fixed_length?: number;
  max_waveforms?: number;
  format?: 'npz' | 'mat' | 'csv';
}

export interface FilterResult {
  time_array: number[];
  original: number[];
  filtered: number[];
  fft_frequencies: number[];
  fft_original: number[];
  fft_filtered: number[];
  filter_type: string;
  freq_low: number | null;
  freq_high: number | null;
  order: number;
  channel: number;
  sample_rate: number;
}

export interface ClusterScatterPoint {
  x: number;
  y: number;
  cluster: number;
  index: number;
}

export interface ClusterFeatureStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  median: number;
}

export interface ClusterStat {
  label: number;
  count: number;
  percentage: number;
  [feature: string]: number | ClusterFeatureStats;
}

export interface TreeRule {
  conditions: { feature: string; op: string; value: number }[];
  cluster: number;
  samples: number;
  confidence: number;
  rule_text: string;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface ClusterResult {
  algorithm: string;
  n_clusters: number;
  total_points: number;
  noise_points: number;
  features: string[];
  scatter_x: string;
  scatter_y: string;
  scatter_data: ClusterScatterPoint[];
  cluster_stats: ClusterStat[];
  metrics: {
    silhouette?: number;
    calinski_harabasz?: number;
    davies_bouldin?: number;
  };
  tree_accuracy: number;
  tree_rules: TreeRule[];
  tree_feature_importance: FeatureImportance[];
}

export interface DeepClusterScatterPoint {
  x: number;
  y: number;
  cluster: number;
  index: number;
}

export interface DeepClusterStat {
  label: number;
  count: number;
  percentage: number;
  medoid_index: number;
}

export interface DeepClusterPrototype {
  cluster: number;
  index: number;
  channel: number;
  sample_rate: number;
  time: number[];
  waveform: number[];
}

export interface DeepClusterResult {
  model: string;
  latent_dim: number;
  n_waveforms: number;
  waveform_length: number;
  epochs: number;
  final_loss: number;
  loss_curve: number[];
  algorithm: string;
  n_clusters: number;
  noise_points: number;
  projection: string;
  scatter_data: DeepClusterScatterPoint[];
  cluster_stats: DeepClusterStat[];
  prototypes: DeepClusterPrototype[];
  metrics: {
    silhouette?: number;
    calinski_harabasz?: number;
    davies_bouldin?: number;
  };
}
