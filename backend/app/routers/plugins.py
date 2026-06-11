from fastapi import APIRouter, HTTPException
from ..models.schemas import PluginInferenceRequest
from ..services.plugin_manager import plugin_manager

router = APIRouter(prefix="/api/plugins", tags=["plugins"])


@router.get("/")
async def list_plugins():
    return plugin_manager.list_plugins()


@router.get("/{plugin_name}")
async def get_plugin_info(plugin_name: str):
    p = plugin_manager.get_plugin(plugin_name)
    if not p:
        raise HTTPException(404, f"Plugin '{plugin_name}' not found")
    return p.get_info()


@router.post("/{plugin_name}/infer")
async def plugin_infer(plugin_name: str, request: PluginInferenceRequest):
    p = plugin_manager.get_plugin(plugin_name)
    if not p:
        raise HTTPException(404, f"Plugin '{plugin_name}' not found")
    try:
        result = p.infer(request.file_id, request.params)
        return {"plugin_name": plugin_name, "result": result}
    except Exception as e:
        raise HTTPException(500, str(e))
