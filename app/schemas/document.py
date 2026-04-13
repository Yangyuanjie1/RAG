from pydantic import BaseModel


class DocumentInfo(BaseModel):
    name: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class UploadResponse(BaseModel):
    filename: str
    chunk_count: int
