from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    source: str
    text: str


def load_documents(folder: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(folder.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(Document(source=path.name, text=text))
    return documents

