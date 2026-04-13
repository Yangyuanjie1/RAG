from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.chat import ChatRequest, ChatResponse, SourceChunk
from app.services.llm import answer_with_context
from app.services.rag_store import rag_store


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        sources = rag_store.search(request.question, settings.retrieval_k)
        answer = answer_with_context(request.question, sources)
        return ChatResponse(
            answer=answer,
            sources=[SourceChunk(**source) for source in sources],
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
