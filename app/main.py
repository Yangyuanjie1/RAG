from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.core.config import STATIC_DIR, settings
from app.services.rag_store import rag_store


app = FastAPI(title=settings.app_name)
app.include_router(chat_router, prefix=settings.api_prefix)
app.include_router(documents_router, prefix=settings.api_prefix)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def startup_event() -> None:
    rag_store.rebuild_from_disk()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "app.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
