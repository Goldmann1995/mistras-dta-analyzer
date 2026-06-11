from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from ..services import dta_service

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
async def get_waveform(file_id: str, index: int):
    try:
        return dta_service.get_waveform(file_id, index)
    except KeyError:
        raise HTTPException(404, "File not found")
    except IndexError as e:
        raise HTTPException(404, str(e))


@router.get("/{file_id}/waveform/{index}/fft")
async def get_waveform_fft(file_id: str, index: int):
    try:
        return dta_service.get_waveform_fft(file_id, index)
    except KeyError:
        raise HTTPException(404, "File not found")
    except IndexError as e:
        raise HTTPException(404, str(e))


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
