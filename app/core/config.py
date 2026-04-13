import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"


class Settings:
    app_name: str = os.getenv("APP_NAME", "RAG Knowledge Assistant")
    api_prefix: str = os.getenv("API_PREFIX", "/api")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.minimaxi.com/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "MiniMax-M2.5")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "3"))


settings = Settings()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
