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

export interface ExportOptions {
  channel?: number;
  keep_pretrigger: boolean;
  normalize: boolean;
  fixed_length?: number;
  max_waveforms?: number;
}
