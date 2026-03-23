from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Smart Book Search")
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    books_dir: Path = Path(os.getenv("BOOKS_DIR", "data/books"))
    index_file: Path = Path(os.getenv("INDEX_FILE", "data/index.json"))
    retrieval_top_k_default: int = int(os.getenv("RETRIEVAL_TOP_K_DEFAULT", "5"))
    retrieval_top_k_max: int = int(os.getenv("RETRIEVAL_TOP_K_MAX", "10"))
    search_page_size_default: int = int(os.getenv("SEARCH_PAGE_SIZE_DEFAULT", "5"))
    search_page_size_max: int = int(os.getenv("SEARCH_PAGE_SIZE_MAX", "50"))
    min_search_score: float = float(os.getenv("MIN_SEARCH_SCORE", "0.05"))
    min_answer_score: float = float(os.getenv("MIN_ANSWER_SCORE", "0.07"))
    sentence_top_n: int = int(os.getenv("SENTENCE_TOP_N", "3"))
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))


settings = Settings()
