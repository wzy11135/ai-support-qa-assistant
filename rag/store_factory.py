import os
from typing import Protocol

from rag.chroma_retriever import ChromaVectorStore
from rag.retriever import LocalVectorStore, SearchResult
from rag.splitter import Chunk


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        ...


def build_retriever(chunks: list[Chunk]) -> tuple[Retriever, str]:
    backend = os.getenv("RETRIEVAL_BACKEND", "local").strip().lower()
    if backend == "chroma":
        try:
            return ChromaVectorStore(chunks), "ChromaDB"
        except Exception as exc:
            short_error = str(exc).splitlines()[0][:80]
            return LocalVectorStore(chunks), f"local fallback ({short_error})"

    return LocalVectorStore(chunks), "local"

