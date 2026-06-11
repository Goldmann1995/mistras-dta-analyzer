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

export interface ExportOptions {
  channel?: number;
  keep_pretrigger: boolean;
  normalize: boolean;
  fixed_length?: number;
  max_waveforms?: number;
}
