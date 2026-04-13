from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)


class SourceChunk(BaseModel):
    document_name: str
    chunk_id: str
    content: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
