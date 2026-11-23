"""The game loop and its rules.

Rules (the classic Wikipedia Game):

* Start on one article and reach the target by clicking links only.
* No back button and no search: once you leave a page you can't return to it,
  so a page is never visited twice.
* If the target is linked from the current page, it gets clicked (any human
  would), so agents are only asked to choose when the answer isn't obvious.
* The game is lost on hitting ``max_steps`` clicks or a page with no unvisited links.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .agents.base import Agent, Turn
from .llm import LLMError
from .wiki import Link, WikiError, WikiSource


@dataclass
class Step:
    from_page: str
    to_page: str
    options: int  # unvisited links that were available on from_page
    seconds: float
    note: str = ""


@dataclass
class GameResult:
    agent: str
    start: str
    target: str
    success: bool
    outcome: str  # "reached target", "step limit", "dead end" or "error: ..."
    path: list[str]
    steps: list[Step] = field(default_factory=list)
    seconds: float = 0.0
    pages_loaded: int = 0  # pages whose links were read: one per move plus lookahead peeks
    llm_calls: int = 0

    @property
    def clicks(self) -> int:
        return len(self.path) - 1

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["clicks"] = self.clicks
        return data


def play(
    wiki: WikiSource,
    agent: Agent,
    start: str,
    target: str,
    *,
    max_steps: int = 20,
    on_step: Callable[[Step], None] | None = None,
) -> GameResult:
    """Play one game and return what happened. Network/LLM failures end the game as an error."""
    agent.reset()
    began = time.perf_counter()
    path = [start]
    steps: list[Step] = []
    moves = 0
    outcome = "reached target"

    try:
        start = wiki.resolve(start)
        target = wiki.resolve(target)
        path = [start]
        goal = Link(target, wiki.describe(target))
        visited = {start}

        while path[-1] != target:
            if len(steps) >= max_steps:
                outcome = "step limit"
                break
            step_began = time.perf_counter()
            current = path[-1]
            links = wiki.links(current)
            moves += 1
            candidates = tuple(link for link in links if link.title not in visited)

            if any(link.title == target for link in candidates):
                choice, note = goal, "target is linked from this page"
            elif not candidates:
                outcome = "dead end"
                break
            else:
                agent.note = ""
                choice = agent.choose(Turn(current, goal, tuple(path), candidates))
                if choice.title not in {c.title for c in candidates}:
                    raise ValueError(f"{agent.name} chose {choice.title!r}, which is not a legal move")
                note = agent.note

            path.append(choice.title)
            visited.add(choice.title)
            step = Step(current, choice.title, len(candidates), time.perf_counter() - step_began, note)
            steps.append(step)
            if on_step:
                on_step(step)
    except (WikiError, LLMError) as exc:
        outcome = f"error: {exc}"

    return GameResult(
        agent=agent.name,
        start=start,
        target=target,
        success=path[-1] == target,
        outcome=outcome if path[-1] != target else "reached target",
        path=path,
        steps=steps,
        seconds=time.perf_counter() - began,
        pages_loaded=moves + agent.peeks,
        llm_calls=agent.llm_calls,
    )
