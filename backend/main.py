"""FastAPI 入口 — KataGo 复盘 Web 服务"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import load_config

app = FastAPI(title="KataGo Review", version="0.1.0")

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")

cfg = load_config()


@app.get("/api/health")
async def health():
    return {"status": "ok"}
