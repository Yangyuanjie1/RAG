import re
from typing import Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from app.core.config import UPLOAD_DIR, settings


@dataclass
class ChunkRecord:
    chunk_id: str
    document_name: str
    content: str
    embedding: np.ndarray


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    clean_text = re.sub(r"\s+", " ", text).strip()
    if not clean_text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(clean_text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = max(end - overlap, start + 1)
    return chunks


class LocalRagStore:
    def __init__(self) -> None:
        self.chunks: list[ChunkRecord] = []
        self.documents: dict[str, int] = {}
        self._embedding_model: Any | None = None

    def _get_embedding_model(self) -> Any:
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer(settings.embedding_model)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load the embedding model. Install requirements and ensure model download access is available."
                ) from exc
        return self._embedding_model

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        model = self._get_embedding_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def rebuild_from_disk(self) -> None:
        self.chunks = []
        self.documents = {}
        for file_path in sorted(UPLOAD_DIR.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in {".txt", ".md", ".pdf"}:
                self.add_document(file_path, rebuild=False)

    def add_document(self, file_path: Path, rebuild: bool = False) -> int:
        text = self._read_file(file_path)
        raw_chunks = _chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        if rebuild:
            self.chunks = [chunk for chunk in self.chunks if chunk.document_name != file_path.name]

        embeddings = self._encode_texts(raw_chunks)
        kept_chunks: list[ChunkRecord] = []
        for index, chunk_text in enumerate(raw_chunks, start=1):
            if not chunk_text.strip():
                continue
            kept_chunks.append(
                ChunkRecord(
                    chunk_id=f"{file_path.stem}-{index}",
                    document_name=file_path.name,
                    content=chunk_text,
                    embedding=embeddings[index - 1],
                )
            )

        self.chunks.extend(kept_chunks)
        self.documents[file_path.name] = len(kept_chunks)
        return len(kept_chunks)

    def list_documents(self) -> list[dict[str, str | int]]:
        return [
            {"name": name, "chunk_count": chunk_count}
            for name, chunk_count in sorted(self.documents.items())
        ]

    def search(self, question: str, top_k: int) -> list[dict[str, str | float]]:
        if not question.strip():
            return []

        question_embedding = self._encode_texts([question])[0]
        scored: list[tuple[float, ChunkRecord]] = []
        for chunk in self.chunks:
            score = float(np.dot(question_embedding, chunk.embedding))
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "document_name": chunk.document_name,
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "score": round(score, 4),
            }
            for score, chunk in scored[:top_k]
        ]

    def _read_file(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            reader = PdfReader(str(file_path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        raise RuntimeError(f"Unsupported file type: {suffix}")


rag_store = LocalRagStore()
