import os
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from ..services import dta_service
from ..services.signal_analysis import (
    compute_cwt, compute_group_velocity_dispersion, compute_cross_channel_velocity,
)

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


@router.get("/{file_id}/hits")
async def get_hits(
    file_id: str,
    channel: Optional[int] = None,
    offset: int = 0,
    limit: int = Query(default=100, le=5000),
    sort_by: Optional[str] = None,
    sort_order: str = "asc",
    amp_min: Optional[int] = None,
    amp_max: Optional[int] = None,
    time_min: Optional[float] = None,
    time_max: Optional[float] = None,
):
    try:
        return dta_service.get_hits(
            file_id, channel=channel, offset=offset, limit=limit,
            sort_by=sort_by, sort_order=sort_order,
            amp_min=amp_min, amp_max=amp_max,
            time_min=time_min, time_max=time_max,
        )
    except KeyError:
        raise HTTPException(404, "File not found")


@router.get("/{file_id}/waveform/{index}")
async def get_waveform(file_id: str, index: int, keep_pretrigger: bool = False):
    try:
        return dta_service.get_waveform(file_id, index, keep_pretrigger=keep_pretrigger)
    except KeyError:
        raise HTTPException(404, "File not found")
    except IndexError as e:
        raise HTTPException(404, str(e))


@router.get("/{file_id}/waveform/{index}/fft")
async def get_waveform_fft(file_id: str, index: int, keep_pretrigger: bool = False):
    try:
        return dta_service.get_waveform_fft(file_id, index, keep_pretrigger=keep_pretrigger)
    except KeyError:
        raise HTTPException(404, "File not found")
    except IndexError as e:
        raise HTTPException(404, str(e))


@router.get("/{file_id}/waveform/{index}/cwt")
async def get_cwt(
    file_id: str,
    index: int,
    wavelet: str = "morl",
    freq_min: float = 1000,
    freq_max: Optional[float] = None,
    num_freqs: int = 128,
    keep_pretrigger: bool = False,
):
    try:
        cache = dta_service._file_cache[file_id]
        wfm = cache['wfm']
        if index >= len(wfm):
            raise IndexError(f"Waveform index {index} out of range")
        return compute_cwt(
            wfm[index], wavelet=wavelet,
            freq_min=freq_min, freq_max=freq_max,
            num_freqs=num_freqs, keep_pretrigger=keep_pretrigger,
        )
    except KeyError:
        raise HTTPException(404, "File not found")
    except IndexError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(400, str(e))


@router.get("/{file_id}/waveform/{index}/dispersion")
async def get_dispersion(
    file_id: str,
    index: int,
    wavelet: str = "morl",
    freq_min: float = 1000,
    freq_max: Optional[float] = None,
    num_freqs: int = 64,
    keep_pretrigger: bool = False,
):
    try:
        cache = dta_service._file_cache[file_id]
        wfm = cache['wfm']
        if index >= len(wfm):
            raise IndexError(f"Waveform index {index} out of range")
        return compute_group_velocity_dispersion(
            wfm[index], wavelet=wavelet,
            freq_min=freq_min, freq_max=freq_max,
            num_freqs=num_freqs, keep_pretrigger=keep_pretrigger,
        )
    except KeyError:
        raise HTTPException(404, "File not found")
    except IndexError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(400, str(e))


@router.get("/{file_id}/group-velocity")
async def get_group_velocity(
    file_id: str,
    sensor_distance: float = Query(..., description="Distance between sensors in meters"),
    keep_pretrigger: bool = False,
):
    try:
        cache = dta_service._file_cache[file_id]
        return compute_cross_channel_velocity(
            cache['wfm'], cache['rec'],
            sensor_distance=sensor_distance,
            keep_pretrigger=keep_pretrigger,
        )
    except KeyError:
        raise HTTPException(404, "File not found")
    except Exception as e:
        raise HTTPException(400, str(e))


@router.get("/{file_id}/channels")
async def get_channel_stats(file_id: str):
    try:
        return dta_service.get_channel_stats(file_id)
    except KeyError:
        raise HTTPException(404, "File not found")


@router.get("/{file_id}/scatter")
async def get_scatter(
    file_id: str,
    x: str = "time",
    y: str = "amplitude",
    color: Optional[str] = None,
    channel: Optional[int] = None,
    max_points: int = 5000,
):
    try:
        return dta_service.get_scatter_data(file_id, x, y, color, channel, max_points)
    except KeyError:
        raise HTTPException(404, "File not found")
    except Exception as e:
        raise HTTPException(400, str(e))


@router.get("/{file_id}/histogram")
async def get_histogram(
    file_id: str,
    field: str = "amplitude",
    bins: int = 50,
    channel: Optional[int] = None,
):
    try:
        return dta_service.get_histogram_data(file_id, field, bins, channel)
    except KeyError:
        raise HTTPException(404, "File not found")
    except Exception as e:
        raise HTTPException(400, str(e))


@router.get("/{file_id}/export/npz")
async def export_npz(
    file_id: str,
    channel: Optional[int] = None,
    keep_pretrigger: bool = False,
    max_waveforms: Optional[int] = None,
    normalize: bool = False,
    fixed_length: Optional[int] = None,
):
    try:
        filepath = dta_service.export_npz(
            file_id, channel=channel, keep_pretrigger=keep_pretrigger,
            max_waveforms=max_waveforms, normalize=normalize, fixed_length=fixed_length,
        )
        return FileResponse(filepath, media_type="application/octet-stream", filename=os.path.basename(filepath))
    except KeyError:
        raise HTTPException(404, "File not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
