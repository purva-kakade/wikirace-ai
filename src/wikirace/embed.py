"""Sentence embeddings and similarity ranking."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .wiki import Link

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one L2-normalised row per text."""
        ...


class SentenceTransformerEmbedder:
    """SBERT embeddings with an in-memory memo, so each string is encoded once per run."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, device: str | None = None, batch_size: int = 256):
        from sentence_transformers import SentenceTransformer  # heavy import, done lazily

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self._memo: dict[str, np.ndarray] = {}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        new = [t for t in dict.fromkeys(texts) if t not in self._memo]
        if new:
            vectors = self._model.encode(
                new,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            self._memo.update(zip(new, vectors.astype(np.float32)))
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([self._memo[t] for t in texts])


class SemanticScorer:
    """Scores links by cosine similarity between their embedding and the target's.

    With ``use_descriptions`` each article is embedded as "Title: short description"
    instead of the bare title, which separates homonyms such as "Java" (the island)
    and "Java (programming language)".
    """

    def __init__(self, embedder: Embedder, use_descriptions: bool = True):
        self.embedder = embedder
        self.use_descriptions = use_descriptions

    def scores(self, target: Link, links: Sequence[Link]) -> np.ndarray:
        if not links:
            return np.zeros(0, dtype=np.float32)
        texts = [target.text(self.use_descriptions)] + [link.text(self.use_descriptions) for link in links]
        vectors = self.embedder.encode(texts)
        return vectors[1:] @ vectors[0]

    def rank(self, target: Link, links: Sequence[Link]) -> list[tuple[Link, float]]:
        """Links sorted from most to least similar to the target (ties keep input order)."""
        scores = self.scores(target, links)
        order = np.argsort(-scores, kind="stable")
        return [(links[i], float(scores[i])) for i in order]
