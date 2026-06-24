import hashlib
import math
from collections.abc import Sequence

from rag.retriever import SearchResult, tokenize
from rag.splitter import Chunk

try:
    import chromadb
except ImportError:  # pragma: no cover - optional dependency.
    chromadb = None


CHROMA_AVAILABLE = chromadb is not None


class HashEmbeddingFunction:
    """Deterministic lightweight embedding function for local ChromaDB demos.

    This avoids downloading large models, so the project remains runnable on
    low-spec laptops. It can later be replaced with sentence-transformers or
    an API-based embedding function without changing the retrieval interface.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class ChromaVectorStore:
    def __init__(self, chunks: list[Chunk]) -> None:
        if chromadb is None:
            raise RuntimeError("ChromaDB is not installed. Run: pip install -r requirements-chroma.txt")
        self.chunks = chunks
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.create_collection(
            name="support_qa_chunks",
            embedding_function=HashEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
        if chunks:
            self.collection.add(
                ids=[chunk.chunk_id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                metadatas=[{"source": chunk.source} for chunk in chunks],
            )

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        if not self.chunks:
            return []
        raw = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, len(self.chunks)),
            include=["documents", "metadatas", "distances"],
        )
        results: list[SearchResult] = []
        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            score = max(0.0, 1.0 - float(distance))
            source = metadata.get("source", "unknown") if metadata else "unknown"
            results.append(SearchResult(chunk=Chunk(chunk_id=chunk_id, source=source, text=document), score=score))
        return results

