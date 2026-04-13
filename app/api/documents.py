from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import UPLOAD_DIR
from app.schemas.document import DocumentInfo, DocumentListResponse, UploadResponse
from app.services.rag_store import rag_store


router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    return DocumentListResponse(
        documents=[DocumentInfo(**doc) for doc in rag_store.list_documents()]
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise HTTPException(status_code=400, detail="Only .txt, .md, and .pdf files are supported.")

    safe_name = Path(file.filename).name
    target_path = UPLOAD_DIR / safe_name
    content = await file.read()
    target_path.write_bytes(content)

    try:
        chunk_count = rag_store.add_document(target_path, rebuild=True)
    except RuntimeError as exc:
        if target_path.exists():
            target_path.unlink()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return UploadResponse(filename=safe_name, chunk_count=chunk_count)
