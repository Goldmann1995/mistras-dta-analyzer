from pydantic import BaseModel
from typing import Optional


class HitRecord(BaseModel):
    index: int
    time: float
    channel: int
    rise: Optional[int] = None
    counts: Optional[int] = None
    peak_counts: Optional[int] = None
    energy: Optional[int] = None
    duration: Optional[int] = None
    amplitude: Optional[int] = None
    asl: Optional[int] = None
    threshold: Optional[int] = None
    avg_frequency: Optional[int] = None
    rms: Optional[float] = None
    rev_frequency: Optional[int] = None
    init_frequency: Optional[int] = None
    signal_strength: Optional[float] = None
    abs_energy: Optional[float] = None
    freq_centroid: Optional[int] = None
    peak_frequency: Optional[int] = None
    timestamp: Optional[float] = None


class WaveformData(BaseModel):
    index: int
    time: float
    channel: int
    sample_rate: float
    time_array: list[float]
    voltage_array: list[float]


class ChannelStats(BaseModel):
    channel: int
    hit_count: int
    avg_amplitude: float
    max_amplitude: float
    min_amplitude: float
    avg_energy: float
    max_energy: float
    avg_duration: float
    max_duration: float
    avg_rms: Optional[float] = None
    time_span: float


class FileInfo(BaseModel):
    filename: str
    file_id: str
    hit_count: int
    waveform_count: int
    channels: list[int]
    duration: float
    fields: list[str]


class FFTResult(BaseModel):
    frequencies: list[float]
    magnitudes: list[float]
    dominant_frequency: float
    sample_rate: float


class PluginInfo(BaseModel):
    name: str
    version: str
    description: str
    endpoints: list[str]


class PluginInferenceRequest(BaseModel):
    plugin_name: str
    file_id: str
    params: dict = {}


class PluginInferenceResponse(BaseModel):
    plugin_name: str
    result: dict
