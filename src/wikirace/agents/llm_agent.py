"""Players that ask a local LLM which link to click."""

from __future__ import annotations

from typing import Any

from ..embed import SemanticScorer
from ..llm import LLM, LLMReplyError
from ..wiki import Link
from .base import Agent, Turn

SYSTEM_PROMPT = """\
You are an expert player of the Wikipedia Game. Starting from one Wikipedia \
article, you must reach a target article by clicking links inside articles. \
You cannot use the back button and you cannot search.

Strategy:
- If a link is clearly about the target, or closely related to it, click it.
- Otherwise move to a broad "hub" article (a country, a field of study, a major \
historical event, a famous person) from which the target is likely reachable.
- Beware of links that only share a word with the target but are about \
something unrelated."""


class LLMAgent(Agent):
    """An embedding model shortlists the links; a local LLM makes the final choice.

    The shortlist keeps prompts small (a page can have 1,000+ links) while the
    LLM brings world knowledge the embedding lacks: that "Apollo 11" is reached
    via "NASA", or that "Java" in the target means the island, not the
    programming language. With ``reasoning=True`` the model must explain its
    thinking before it picks (chain-of-thought); otherwise it answers with a
    number only.
    """

    def __init__(
        self,
        scorer: SemanticScorer,
        llm: LLM,
        shortlist: int = 8,
        reasoning: bool = False,
    ):
        super().__init__()
        self.scorer = scorer
        self.llm = llm
        self.shortlist = shortlist
        self.reasoning = reasoning
        self.name = "llm-cot" if reasoning else "llm"
        self.fallbacks = 0  # moves where the LLM reply was unusable

    def reset(self) -> None:
        super().reset()
        self.fallbacks = 0

    def build_prompt(self, turn: Turn, options: list[Link]) -> str:
        with_desc = self.scorer.use_descriptions
        lines = [
            f"Target article: {turn.target.text(with_desc)}",
            f"Path so far: {' -> '.join(turn.path)}",
            f"You are on: {turn.current}",
            "",
            "Links you can click:",
        ]
        lines += [f"{i}. {link.text(with_desc)}" for i, link in enumerate(options, 1)]
        lines.append("")
        if self.reasoning:
            lines.append(
                "First explain in two to four sentences how the most promising options could lead "
                'to the target, then pick one. Reply as JSON with keys "reasoning" and "choice" '
                "(the number of the link)."
            )
        else:
            lines.append('Reply as JSON with one key, "choice": the number of the link to click.')
        return "\n".join(lines)

    def _schema(self, n_options: int) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        if self.reasoning:
            properties["reasoning"] = {"type": "string"}
        properties["choice"] = {"type": "integer", "minimum": 1, "maximum": n_options}
        return {"type": "object", "properties": properties, "required": list(properties)}

    def choose(self, turn: Turn) -> Link:
        options = [link for link, _ in self.scorer.rank(turn.target, turn.candidates)[: self.shortlist]]
        if len(options) == 1:
            self.note = "only one option"
            return options[0]

        self.llm_calls += 1
        try:
            reply = self.llm.complete_json(SYSTEM_PROMPT, self.build_prompt(turn, options), self._schema(len(options)))
        except LLMReplyError as exc:
            reply = {"choice": None, "error": str(exc)}

        try:
            index = int(reply.get("choice")) - 1
        except (TypeError, ValueError):
            index = -1
        if not 0 <= index < len(options):
            self.fallbacks += 1
            self.note = f"unusable LLM reply {reply.get('choice')!r}; took the top embedding match"
            return options[0]

        self.note = str(reply.get("reasoning", "")).strip()
        return options[index]
