import axios from 'axios';
import type {
  FileInfo, HitsResponse, WaveformData, FFTResult,
  ChannelStats, ScatterData, HistogramData, PluginInfo,
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

export async function getFileInfo(fileId: string): Promise<FileInfo> {
  const { data } = await api.get<FileInfo>(`/api/files/${fileId}`);
  return data;
}

export async function getHits(
  fileId: string,
  params: {
    channel?: number;
    offset?: number;
    limit?: number;
    sort_by?: string;
    sort_order?: string;
    amp_min?: number;
    amp_max?: number;
    time_min?: number;
    time_max?: number;
  } = {},
): Promise<HitsResponse> {
  const { data } = await api.get<HitsResponse>(`/api/analysis/${fileId}/hits`, { params });
  return data;
}

export async function getWaveform(fileId: string, index: number): Promise<WaveformData> {
  const { data } = await api.get<WaveformData>(`/api/analysis/${fileId}/waveform/${index}`);
  return data;
}

export async function getWaveformFFT(fileId: string, index: number): Promise<FFTResult> {
  const { data } = await api.get<FFTResult>(`/api/analysis/${fileId}/waveform/${index}/fft`);
  return data;
}

export async function getChannelStats(fileId: string): Promise<ChannelStats[]> {
  const { data } = await api.get<ChannelStats[]>(`/api/analysis/${fileId}/channels`);
  return data;
}

export async function getScatterData(
  fileId: string,
  x: string, y: string,
  color?: string, channel?: number,
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

export async function getPlugins(): Promise<PluginInfo[]> {
  const { data } = await api.get<PluginInfo[]>('/api/plugins/');
  return data;
}
