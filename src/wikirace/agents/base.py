"""The interface every Wikipedia Game player implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..wiki import Link


@dataclass(frozen=True)
class Turn:
    """Everything a player can see when it is their move."""

    current: str  # article we are on
    target: Link  # article we are trying to reach (with its short description)
    path: tuple[str, ...]  # articles visited so far, ending with `current`
    candidates: tuple[Link, ...]  # links on `current` that have not been visited yet


class Agent(ABC):
    """A player. The game calls :meth:`reset` before each game, then :meth:`choose` once per move."""

    name = "agent"

    def __init__(self) -> None:
        self.llm_calls = 0  # LLM requests made during the current game
        self.peeks = 0  # extra pages opened to look ahead during the current game
        self.note = ""  # optional explanation of the latest move, shown in logs

    def reset(self) -> None:
        self.llm_calls = 0
        self.peeks = 0
        self.note = ""

    @abstractmethod
    def choose(self, turn: Turn) -> Link:
        """Return one of ``turn.candidates``."""
