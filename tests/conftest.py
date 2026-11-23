"""Offline fakes so the whole test suite runs without network, models or Ollama."""

from __future__ import annotations

import re
import zlib
from typing import Any, Callable

import numpy as np
import pytest

from wikirace.embed import SemanticScorer
from wikirace.wiki import Link, PageNotFound


class FakeWiki:
    """An in-memory link graph with the same interface as WikipediaClient."""

    def __init__(
        self,
        graph: dict[str, list[str]],
        descriptions: dict[str, str] | None = None,
        redirects: dict[str, str] | None = None,
    ):
        self.graph = graph
        self.descriptions = descriptions or {}
        self.redirects = redirects or {}  # alias -> canonical title
        self.link_calls: list[str] = []

    def resolve(self, title: str) -> str:
        title = self.redirects.get(title, title)
        if title not in self.graph:
            raise PageNotFound(title)
        return title

    def describe(self, title: str) -> str:
        return self.descriptions.get(title, "")

    def search(self, query: str, limit: int = 5) -> list[str]:
        words = set(query.lower().split())
        hits = [t for t in self.graph if words & set(t.lower().split())]
        return sorted(hits)[:limit]

    def links(self, title: str) -> list[Link]:
        self.link_calls.append(title)
        return [Link(t, self.descriptions.get(t, "")) for t in self.graph.get(title, [])]

    def backlinks(self, title: str, limit: int = 25_000) -> tuple[list[str], bool]:
        return sorted(src for src, outs in self.graph.items() if title in outs), True

    def redirect_titles(self, title: str, limit: int = 49) -> list[str]:
        return [alias for alias, canonical in self.redirects.items() if canonical == title]

    def pages_linking_to(self, sources, targets) -> set[str]:
        wanted = set(targets)
        return {s for s in sources if wanted & set(self.graph.get(s, []))}


class BagOfWordsEmbedder:
    """Deterministic stand-in for SBERT: hashed word counts, L2-normalised."""

    dim = 512

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[i, zlib.crc32(word.encode()) % self.dim] += 1
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


class ScriptedLLM:
    """Returns canned replies (a list, or a function of the prompt) and records prompts."""

    model = "scripted"

    def __init__(self, replies: list[dict[str, Any]] | Callable[[str], dict[str, Any]]):
        self.replies = replies
        self.prompts: list[str] = []
        self.schemas: list[dict[str, Any]] = []

    def complete_json(self, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.prompts.append(prompt)
        self.schemas.append(schema)
        if callable(self.replies):
            return self.replies(prompt)
        return self.replies.pop(0)


@pytest.fixture
def scorer() -> SemanticScorer:
    return SemanticScorer(BagOfWordsEmbedder())
