import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routers import files, analysis, plugins
from .services.plugin_manager import plugin_manager

app = FastAPI(
    title="Mistras DTA Analyzer",
    description="AEwin-like acoustic emission data analysis platform with plugin support",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(files.router)
app.include_router(analysis.router)
app.include_router(plugins.router)

plugins_dir = os.path.join(os.path.dirname(__file__), 'plugins')
plugin_manager.load_plugins_from_dir(plugins_dir)

frontend_dist = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'dist')
if os.path.isdir(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
