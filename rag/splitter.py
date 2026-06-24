from dataclasses import dataclass

from config import CHUNK_OVERLAP, CHUNK_SIZE
from rag.loader import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str


def split_document(document: Document, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    paragraphs = [p.strip() for p in document.text.split("\n\n") if p.strip()]
    merged = "\n\n".join(paragraphs)
    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(merged):
        end = min(start + chunk_size, len(merged))
        text = merged[start:end].strip()
        if text:
            chunks.append(Chunk(chunk_id=f"{document.source}#{index}", source=document.source, text=text))
        if end == len(merged):
            break
        start = max(0, end - overlap)
        index += 1

    return chunks


def split_documents(documents: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(split_document(document))
    return chunks

