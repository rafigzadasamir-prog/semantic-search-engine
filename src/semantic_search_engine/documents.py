"""Loading and representing the bundled document corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_CORPUS_PATH = Path(__file__).resolve().parent / "data" / "corpus.json"


@dataclass(frozen=True)
class Document:
    """A single indexed document."""

    id: str
    topic: str
    title: str
    text: str

    @property
    def content(self) -> str:
        """Text used for indexing: title and body concatenated."""
        return f"{self.title}. {self.text}"


def load_corpus(path: str | Path | None = None) -> list[Document]:
    """Load the bundled corpus (or a corpus at ``path``) into ``Document`` objects.

    When ``path`` is omitted, the corpus packaged at
    ``semantic_search_engine/data/corpus.json`` is used.
    """
    corpus_path = Path(path) if path is not None else _DEFAULT_CORPUS_PATH
    records = json.loads(corpus_path.read_text(encoding="utf-8"))
    return [
        Document(id=r["id"], topic=r["topic"], title=r["title"], text=r["text"])
        for r in records
    ]
