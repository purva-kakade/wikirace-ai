"""Wikipedia Game players and a factory to build them by name."""

from __future__ import annotations

from ..embed import SemanticScorer
from ..llm import LLM
from ..wiki import WikiSource
from .base import Agent, Turn
from .lookahead import LookaheadAgent
from .llm_agent import LLMAgent
from .simple import GreedyAgent, RandomAgent

AGENT_NAMES = ("random", "greedy", "lookahead", "llm", "llm-cot")
NEEDS_EMBEDDINGS = {"greedy", "lookahead", "llm", "llm-cot"}
NEEDS_LLM = {"llm", "llm-cot"}


def build_agent(
    name: str,
    *,
    wiki: WikiSource,
    scorer: SemanticScorer | None = None,
    llm: LLM | None = None,
    shortlist: int = 8,
    lookahead_k: int = 5,
    seed: int = 0,
) -> Agent:
    """Create an agent from its name (one of :data:`AGENT_NAMES`)."""
    if name not in AGENT_NAMES:
        raise ValueError(f"Unknown agent {name!r}; choose from {', '.join(AGENT_NAMES)}")
    if name in NEEDS_EMBEDDINGS and scorer is None:
        raise ValueError(f"Agent {name!r} needs a SemanticScorer")
    if name in NEEDS_LLM and llm is None:
        raise ValueError(f"Agent {name!r} needs an LLM")

    if name == "random":
        return RandomAgent(seed=seed)
    if name == "greedy":
        return GreedyAgent(scorer)
    if name == "lookahead":
        return LookaheadAgent(wiki, scorer, k=lookahead_k)
    return LLMAgent(scorer, llm, shortlist=shortlist, reasoning=(name == "llm-cot"))


__all__ = [
    "AGENT_NAMES",
    "Agent",
    "GreedyAgent",
    "LLMAgent",
    "LookaheadAgent",
    "RandomAgent",
    "Turn",
    "build_agent",
]
