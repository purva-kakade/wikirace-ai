"""A player that peeks one click ahead before committing."""

from __future__ import annotations

from ..embed import SemanticScorer
from ..wiki import Link, WikiError, WikiSource
from .base import Agent, Turn


class LookaheadAgent(Agent):
    """Embedding search with one step of lookahead.

    The ``k`` candidates most similar to the target are opened (their outgoing
    links are loaded, the way a human hovers over a link to preview it). Each
    candidate is then scored as::

        alpha * sim(candidate, target) + (1 - alpha) * max sim(link on candidate, target)

    i.e. by how close it is now *and* how close it can get you on the next
    click. A candidate whose page links straight to the target wins outright.
    Greedy search gets stuck on pages that sound right but are dead ends; this
    agent avoids most of them at the cost of ``k`` extra page loads per move.
    """

    name = "lookahead"

    def __init__(self, wiki: WikiSource, scorer: SemanticScorer, k: int = 5, alpha: float = 0.3):
        super().__init__()
        self.wiki = wiki
        self.scorer = scorer
        self.k = k
        self.alpha = alpha

    def choose(self, turn: Turn) -> Link:
        shortlist = self.scorer.rank(turn.target, turn.candidates)[: self.k]
        visited = set(turn.path)
        best_link, best_score = shortlist[0][0], float("-inf")

        for link, own_score in shortlist:
            try:
                outlinks = self.wiki.links(link.title)
            except WikiError:
                continue
            self.peeks += 1
            if any(out.title == turn.target.title for out in outlinks):
                self.note = f"{link.title!r} links directly to the target"
                return link
            onward = [out for out in outlinks if out.title not in visited and out.title != link.title]
            next_score = float(self.scorer.scores(turn.target, onward).max()) if onward else 0.0
            score = self.alpha * own_score + (1 - self.alpha) * next_score
            if score > best_score:
                best_link, best_score = link, score
                self.note = f"score {score:.2f} (now {own_score:.2f}, next click up to {next_score:.2f})"

        return best_link
