import math
import re
from collections import Counter
from dataclasses import dataclass

from rag.splitter import Chunk


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(token, 0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class LocalVectorStore:
    """Small local vector index for fast demos without GPU or external services."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.vectors = [Counter(tokenize(chunk.text)) for chunk in chunks]

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        query_vector = Counter(tokenize(query))
        scored = [
            SearchResult(chunk=chunk, score=cosine_similarity(query_vector, vector))
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

