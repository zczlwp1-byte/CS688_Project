"""
tools/rag.py — Retrieval-Augmented Generation layer.

Chunks article text, embeds chunks locally via Ollama (nomic-embed-text),
and indexes them in Meta's FAISS for similarity search.
"""

from __future__ import annotations

import numpy as np
import faiss

import llm


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping character chunks (word-boundary aware)."""
    words = text.split()
    chunks, current, length = [], [], 0
    for word in words:
        current.append(word)
        length += len(word) + 1
        if length >= chunk_size:
            chunks.append(" ".join(current))
            # Keep the tail of the chunk as overlap for context continuity.
            tail = " ".join(current)[-overlap:].split()
            current, length = tail, sum(len(w) + 1 for w in tail)
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


class VectorStore:
    """A minimal FAISS-backed vector store with metadata."""

    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[str] = []
        self.metadata: list[dict] = []  # one dict per chunk (title, url, ...)

    def _normalize(self, vectors: list[list[float]]) -> np.ndarray:
        arr = np.array(vectors, dtype="float32")
        faiss.normalize_L2(arr)  # inner product on unit vectors == cosine sim
        return arr

    def add_article(self, article: dict) -> int:
        """Chunk + embed + index one article. Returns number of chunks added."""
        chunks = chunk_text(article["text"])
        if not chunks:
            return 0
        vectors = self._normalize(llm.embed(chunks))
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks.extend(chunks)
        self.metadata.extend(
            [{"title": article.get("title", ""), "url": article.get("url", "")}] * len(chunks)
        )
        return len(chunks)

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Return top-k chunks most similar to the query."""
        if self.index is None or self.index.ntotal == 0:
            return []
        qvec = self._normalize(llm.embed([query]))
        scores, ids = self.index.search(qvec, min(k, self.index.ntotal))
        return [
            {
                "chunk": self.chunks[i],
                "score": float(s),
                **self.metadata[i],
            }
            for s, i in zip(scores[0], ids[0]) if i != -1
        ]

    def answer(self, question: str, k: int = 4) -> dict:
        """Full RAG pipeline: retrieve top chunks, ask the LLM to answer
        grounded ONLY in those chunks, and return answer + sources."""
        hits = self.search(question, k=k)
        if not hits:
            return {"answer": "The knowledge base is empty — fetch some articles first.",
                    "sources": []}
        context = "\n\n---\n\n".join(
            f"[{h['title']}]\n{h['chunk']}" for h in hits
        )
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the context does not contain the answer, say so.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
        )
        answer = llm.generate(prompt, system="You are a precise research assistant.")
        sources = list({(h["title"], h["url"]) for h in hits})
        return {"answer": answer, "sources": [{"title": t, "url": u} for t, u in sources]}
