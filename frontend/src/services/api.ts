import axios from 'axios';
import type {
  FileInfo, HitsResponse, WaveformData, FFTResult,
  ChannelStats, ScatterData, HistogramData, PluginInfo, ExportOptions,
  CWTResult, DispersionResult, GroupVelocityResult, EMDResult,
} from '../types';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
});

export async function uploadFile(file: File): Promise<FileInfo> {
  const form = new FormData();
  form.append('file', file);
  const { data } = await api.post<FileInfo>('/api/files/upload', form);
  return data;
}

export async function listFiles(): Promise<FileInfo[]> {
  const { data } = await api.get<FileInfo[]>('/api/files/');
  return data;
}

export async function getHits(
  fileId: string,
  params: {
    channel?: number; offset?: number; limit?: number;
    sort_by?: string; sort_order?: string;
    amp_min?: number; amp_max?: number;
    time_min?: number; time_max?: number;
  } = {},
): Promise<HitsResponse> {
  const { data } = await api.get<HitsResponse>(`/api/analysis/${fileId}/hits`, { params });
  return data;
}

export async function getWaveform(
  fileId: string, index: number, keepPretrigger: boolean = false,
): Promise<WaveformData> {
  const { data } = await api.get<WaveformData>(
    `/api/analysis/${fileId}/waveform/${index}`, { params: { keep_pretrigger: keepPretrigger } },
  );
  return data;
}

export async function getWaveformFFT(
  fileId: string, index: number, keepPretrigger: boolean = false,
): Promise<FFTResult> {
  const { data } = await api.get<FFTResult>(
    `/api/analysis/${fileId}/waveform/${index}/fft`, { params: { keep_pretrigger: keepPretrigger } },
  );
  return data;
}

export async function getChannelStats(fileId: string): Promise<ChannelStats[]> {
  const { data } = await api.get<ChannelStats[]>(`/api/analysis/${fileId}/channels`);
  return data;
}

export async function getScatterData(
  fileId: string, x: string, y: string, color?: string, channel?: number,
): Promise<ScatterData> {
  const { data } = await api.get<ScatterData>(`/api/analysis/${fileId}/scatter`, {
    params: { x, y, color, channel },
  });
  return data;
}

export async function getHistogramData(
  fileId: string, field: string, bins?: number, channel?: number,
): Promise<HistogramData> {
  const { data } = await api.get<HistogramData>(`/api/analysis/${fileId}/histogram`, {
    params: { field, bins, channel },
  });
  return data;
}

export async function getCWT(
  fileId: string, index: number,
  opts: { wavelet?: string; freq_min?: number; freq_max?: number; num_freqs?: number; keep_pretrigger?: boolean } = {},
): Promise<CWTResult> {
  const { data } = await api.get<CWTResult>(`/api/analysis/${fileId}/waveform/${index}/cwt`, { params: opts });
  return data;
}

export async function getDispersion(
  fileId: string, index: number,
  opts: { wavelet?: string; freq_min?: number; freq_max?: number; num_freqs?: number; keep_pretrigger?: boolean } = {},
): Promise<DispersionResult> {
  const { data } = await api.get<DispersionResult>(`/api/analysis/${fileId}/waveform/${index}/dispersion`, { params: opts });
  return data;
}

export async function getEMD(
  fileId: string, index: number,
  opts: { method?: string; max_imfs?: number; keep_pretrigger?: boolean } = {},
): Promise<EMDResult> {
  const { data } = await api.get<EMDResult>(`/api/analysis/${fileId}/waveform/${index}/emd`, { params: opts });
  return data;
}

export async function getGroupVelocity(
  fileId: string, sensorDistance: number, keepPretrigger: boolean = false,
): Promise<GroupVelocityResult> {
  const { data } = await api.get<GroupVelocityResult>(`/api/analysis/${fileId}/group-velocity`, {
    params: { sensor_distance: sensorDistance, keep_pretrigger: keepPretrigger },
  });
  return data;
}

export async function getPlugins(): Promise<PluginInfo[]> {
  const { data } = await api.get<PluginInfo[]>('/api/plugins/');
  return data;
}

export function getExportUrl(fileId: string, opts: ExportOptions): string {
  const base = api.defaults.baseURL || '';
  const params = new URLSearchParams();
  if (opts.channel !== undefined) params.set('channel', String(opts.channel));
  params.set('keep_pretrigger', String(opts.keep_pretrigger));
  params.set('normalize', String(opts.normalize));
  if (opts.fixed_length) params.set('fixed_length', String(opts.fixed_length));
  if (opts.max_waveforms) params.set('max_waveforms', String(opts.max_waveforms));
  return `${base}/api/analysis/${fileId}/export/npz?${params.toString()}`;
}
