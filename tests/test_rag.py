from rag.loader import Document
from rag.retriever import LocalVectorStore
from rag.splitter import split_document


def test_split_document_creates_chunks():
    document = Document(source="test.md", text="退款政策。" * 200)
    chunks = split_document(document, chunk_size=120, overlap=20)
    assert len(chunks) > 1
    assert chunks[0].source == "test.md"


def test_retriever_returns_relevant_chunk():
    chunks = split_document(Document(source="refund.md", text="超过7天但未超过15天，质量问题支持退货或换货。"))
    store = LocalVectorStore(chunks)
    results = store.search("超过7天可以退款吗", top_k=1)
    assert results
    assert results[0].score > 0

