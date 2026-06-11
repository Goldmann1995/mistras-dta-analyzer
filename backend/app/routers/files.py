import os
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException

from ..services import dta_service

router = APIRouter(prefix="/api/files", tags=["files"])

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "mistras_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.dta'):
        raise HTTPException(400, "Only .DTA files are supported")

    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        file_id = dta_service.load_dta_file(filepath, file.filename)
    except Exception as e:
        os.remove(filepath)
        raise HTTPException(400, f"Failed to parse DTA file: {str(e)}")

    return dta_service.get_file_info(file_id)


@router.get("/")
async def list_files():
    return dta_service.get_loaded_files()


@router.get("/{file_id}")
async def get_file(file_id: str):
    try:
        return dta_service.get_file_info(file_id)
    except KeyError:
        raise HTTPException(404, "File not found")


@router.delete("/{file_id}")
async def delete_file(file_id: str):
    dta_service.remove_file(file_id)
    return {"status": "ok"}
