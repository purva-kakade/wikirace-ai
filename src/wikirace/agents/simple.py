"""Baseline players that need no LLM."""

from __future__ import annotations

import random

from ..embed import SemanticScorer
from ..wiki import Link
from .base import Agent, Turn


class RandomAgent(Agent):
    """Clicks a random unvisited link. The floor every other agent should beat."""

    name = "random"

    def __init__(self, seed: int = 0):
        super().__init__()
        self.seed = seed
        self._rng = random.Random(seed)

    def reset(self) -> None:
        super().reset()
        self._rng = random.Random(self.seed)

    def choose(self, turn: Turn) -> Link:
        return self._rng.choice(turn.candidates)


class GreedyAgent(Agent):
    """Clicks the link whose embedding is most similar to the target's.

    Fast (no LLM) but short-sighted: it only ever looks at the current page.
    """

    name = "greedy"

    def __init__(self, scorer: SemanticScorer):
        super().__init__()
        self.scorer = scorer

    def choose(self, turn: Turn) -> Link:
        link, score = self.scorer.rank(turn.target, turn.candidates)[0]
        self.note = f"similarity {score:.2f}"
        return link
