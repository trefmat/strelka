from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import settings


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    book: str
    offset_start: int
    offset_end: int
    text: str


class BookStore:
    def __init__(self) -> None:
        self.books_dir: Path = settings.books_dir
        self.index_file: Path = settings.index_file
        self.books_dir.mkdir(parents=True, exist_ok=True)
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        self._chunks: list[Chunk] = []
        self._load()

    def _load(self) -> None:
        if not self.index_file.exists():
            self._chunks = []
            return
        raw = json.loads(self.index_file.read_text(encoding="utf-8"))
        self._chunks = [Chunk(**item) for item in raw.get("chunks", [])]

    def _save(self) -> None:
        payload = {"chunks": [asdict(c) for c in self._chunks]}
        self.index_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_book(self, filename: str, content: str, chunk_size: int = 1000, overlap: int = 180) -> int:
        safe_name = Path(filename).name
        (self.books_dir / safe_name).write_text(content, encoding="utf-8")

        chunks = self._chunk_text(safe_name, content, chunk_size=chunk_size, overlap=overlap)
                                                                                   
                                                        
        self._chunks = [c for c in self._chunks if c.book != safe_name]
        self._chunks.extend(chunks)
        self._save()
        return len(chunks)

    def _chunk_text(self, book: str, text: str, *, chunk_size: int, overlap: int) -> list[Chunk]:
        clean = text.replace("\r\n", "\n").replace("\r", "\n")
        result: list[Chunk] = []
        start = 0
        n = len(clean)
        while start < n:
            end = min(start + chunk_size, n)
                                                                                 
            boundary = clean.rfind("\n\n", start, end)
            if boundary == -1:
                boundary = clean.rfind(". ", start, end)
            if boundary != -1 and boundary > start + 350:
                end = boundary + 1
            chunk_text = clean[start:end].strip()
            if chunk_text:
                result.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        book=book,
                        offset_start=start,
                        offset_end=end,
                        text=chunk_text,
                    )
                )
            if end >= n:
                break
            start = max(end - overlap, start + 1)
        return result

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def stats(self) -> tuple[int, int]:
        books = {c.book for c in self._chunks}
        return len(books), len(self._chunks)
