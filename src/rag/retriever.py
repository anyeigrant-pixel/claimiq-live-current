"""Local FAISS-backed embedding retrieval for synthetic policy documents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Iterable

import faiss
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import POLICY_DIR, VECTOR_DIR, get_logger

LOGGER = get_logger(__name__)
INDEX_FILE = VECTOR_DIR / "policies.faiss"
META_FILE = VECTOR_DIR / "passages.json"
VECTORIZER_FILE = VECTOR_DIR / "vectorizer.joblib"


@dataclass(frozen=True)
class Passage:
    source: str
    text: str
    score: float
    citation: str


def _chunks() -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        heading = ""
        for paragraph in path.read_text(encoding="utf-8").split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            lines = paragraph.splitlines()
            if lines[0].startswith("#"):
                heading = lines[0].lstrip("#").strip()
                paragraph = "\n".join(lines[1:]).strip()
                if not paragraph:
                    continue
            chunk_id = f"{path.stem}:{len(chunks) + 1}"
            chunks.append({"id": chunk_id, "source": path.name, "heading": heading, "text": paragraph})
    if not chunks:
        raise FileNotFoundError(f"No Markdown policy documents found in {POLICY_DIR}")
    return chunks


class LocalFaissRetriever:
    """Persist a compact, local vector index without external infrastructure."""

    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer | None = None
        self.index: faiss.Index | None = None
        self.passages: list[dict[str, str]] = []
        self._load_or_build()

    def _load_or_build(self) -> None:
        source_snapshot = _chunks()
        if INDEX_FILE.exists() and META_FILE.exists() and VECTORIZER_FILE.exists():
            persisted = json.loads(META_FILE.read_text(encoding="utf-8"))
            if [(item["id"], item["text"]) for item in persisted] == [(item["id"], item["text"]) for item in source_snapshot]:
                self.passages = persisted
                self.vectorizer = joblib.load(VECTORIZER_FILE)
                self.index = faiss.read_index(str(INDEX_FILE))
                return
        self.build(source_snapshot)

    def build(self, passages: Iterable[dict[str, str]] | None = None) -> None:
        self.passages = list(passages or _chunks())
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), norm="l2")
        matrix = self.vectorizer.fit_transform(item["text"] for item in self.passages).toarray().astype("float32")
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_FILE))
        META_FILE.write_text(json.dumps(self.passages, indent=2), encoding="utf-8")
        joblib.dump(self.vectorizer, VECTORIZER_FILE)
        LOGGER.info("Built local FAISS index with %d policy passages", len(self.passages))

    def retrieve(self, query: str, top_k: int = 3) -> list[Passage]:
        if not query.strip():
            raise ValueError("A non-empty policy query is required.")
        if self.vectorizer is None or self.index is None:
            raise RuntimeError("Policy vector index was not initialized.")
        query_vector = self.vectorizer.transform([query]).toarray().astype("float32")
        scores, ids = self.index.search(query_vector, min(top_k, len(self.passages)))
        return [
            Passage(
                source=self.passages[index]["source"],
                text=self.passages[index]["text"],
                score=round(float(score), 4),
                citation=f"[{self.passages[index]['source']} — {self.passages[index]['heading']}]",
            )
            for score, index in zip(scores[0], ids[0], strict=True)
            if index >= 0
        ]


_RETRIEVER: LocalFaissRetriever | None = None


def retrieve(query: str, top_k: int = 3) -> list[Passage]:
    """Retrieve policy passages with a stable public contract."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = LocalFaissRetriever()
    return _RETRIEVER.retrieve(query, top_k)
