"""Shortest-path oracle: how many clicks would a perfect player need?

Exhaustive breadth-first search is out of reach on live Wikipedia (an article
has hundreds of links, so depth 3 is tens of millions of pages). Instead the
oracle meets in the middle, checking path lengths in order:

1. **1 click**: the target is linked from the start page.
2. **2 clicks**: some page linked from the start links to the target (or to a
   redirect to it). Checked exactly, 50 pages per request, via ``pltitles``.
3. **3 clicks**: start -> a -> b -> target, where ``b`` is among the pages that
   link to the target (its backlinks). First-hop pages are expanded most
   promising first, within a request budget.

Because lengths 1 and 2 are ruled out exactly before length 3 is tried, any
path the oracle returns is a proven shortest path. If no path of length 3 or
less turns up within the budget, only a lower bound is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .embed import SemanticScorer
from .wiki import Link, WikiError


class GraphSource(Protocol):
    def resolve(self, title: str) -> str: ...

    def describe(self, title: str) -> str: ...

    def links(self, title: str) -> list[Link]: ...

    def backlinks(self, title: str, limit: int = ...) -> tuple[list[str], bool]: ...

    def redirect_titles(self, title: str, limit: int = ...) -> list[str]: ...

    def pages_linking_to(self, sources: Sequence[str], targets: Sequence[str]) -> set[str]: ...


@dataclass
class OracleResult:
    start: str
    target: str
    path: list[str] | None  # a shortest path, when one was found
    lower_bound: int  # the optimum is at least this many clicks
    pages_expanded: int = 0

    @property
    def length(self) -> int | None:
        """Exact optimal number of clicks, or None if unknown."""
        return None if self.path is None else len(self.path) - 1


def shortest_path(
    wiki: GraphSource,
    start: str,
    target: str,
    *,
    budget: int = 400,
    backlink_limit: int = 25_000,
    scorer: SemanticScorer | None = None,
) -> OracleResult:
    """Find a proven shortest path of up to 3 clicks, or a lower bound."""
    start, target = wiki.resolve(start), wiki.resolve(target)
    if start == target:
        return OracleResult(start, target, [start], 0)

    first_hop = [link.title for link in wiki.links(start)]
    if target in first_hop:
        return OracleResult(start, target, [start, target], 1)

    aliases = [target, *wiki.redirect_titles(target)]
    bridges = wiki.pages_linking_to(first_hop, aliases)
    if bridges:
        bridge = next((title for title in first_hop if title in bridges), min(bridges))
        return OracleResult(start, target, [start, bridge, target], 2)

    linkers, linkers_complete = wiki.backlinks(target, limit=backlink_limit)
    linker_set = set(linkers) - {start}

    order = first_hop
    if scorer is not None:
        goal = Link(target, wiki.describe(target))
        order = [link.title for link, _ in scorer.rank(goal, [Link(t) for t in first_hop])]

    expanded = 0
    for page in order:
        if expanded >= budget:
            break
        try:
            second_hop = wiki.links(page)
        except WikiError:
            continue
        expanded += 1
        for link in second_hop:
            if link.title in linker_set:
                return OracleResult(start, target, [start, page, link.title, target], 3, expanded)

    searched_everything = expanded == len(first_hop) and linkers_complete
    return OracleResult(start, target, None, 4 if searched_everything else 3, expanded)
