"""Run every agent on a set of scenarios and score them against the oracle."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .agents.base import Agent
from .embed import SemanticScorer
from .game import play
from .oracle import shortest_path
from .wiki import WikiError


@dataclass(frozen=True)
class Scenario:
    start: str
    target: str
    category: str = "general"


def load_scenarios(path: str | Path) -> list[Scenario]:
    raw = json.loads(Path(path).read_text())
    return [Scenario(item["start"], item["target"], item.get("category", "general")) for item in raw]


def run_benchmark(
    wiki: Any,
    agents: Sequence[Agent],
    scenarios: Sequence[Scenario],
    *,
    max_steps: int = 20,
    oracle_budget: int | None = 400,
    scorer: SemanticScorer | None = None,
    out_path: str | Path | None = None,
    meta: dict[str, Any] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Play every (scenario, agent) pair. Results are saved after each game, so a crash loses nothing."""
    report: dict[str, Any] = {
        "meta": {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "max_steps": max_steps,
            "agents": [agent.name for agent in agents],
            **(meta or {}),
        },
        "scenarios": [],
        "games": [],
    }

    for i, scenario in enumerate(scenarios, 1):
        log(f"\n[{i}/{len(scenarios)}] {scenario.start} -> {scenario.target}  ({scenario.category})")
        optimal = None
        if oracle_budget:
            try:
                oracle = shortest_path(wiki, scenario.start, scenario.target, budget=oracle_budget, scorer=scorer)
            except WikiError as exc:
                log(f"    skipped: {exc}")
                continue
            optimal = oracle.length
            report["scenarios"].append(
                {
                    "start": oracle.start,
                    "target": oracle.target,
                    "category": scenario.category,
                    "optimal": optimal,
                    "lower_bound": oracle.lower_bound,
                    "oracle_path": oracle.path,
                }
            )
            shown = f"{optimal} clicks via {' -> '.join(oracle.path)}" if oracle.path else f">= {oracle.lower_bound} clicks"
            log(f"    optimal: {shown}")

        for agent in agents:
            result = play(wiki, agent, scenario.start, scenario.target, max_steps=max_steps)
            row = result.to_dict()
            row.update(category=scenario.category, optimal=optimal)
            report["games"].append(row)
            mark = "WIN " if result.success else "LOSS"
            log(
                f"    {agent.name:<10} {mark} {result.clicks:>2} clicks  {result.seconds:6.1f}s  "
                f"{' -> '.join(result.path) if result.success else result.outcome}"
            )
            if out_path:
                save_report(report, out_path)

    return report


def save_report(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    tmp.replace(path)


def summarize(games: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-agent metrics, in the order agents first appear."""
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for game in games:
        by_agent.setdefault(game["agent"], []).append(game)

    rows = []
    for agent, played in by_agent.items():
        wins = [g for g in played if g["success"]]
        scored = [g for g in wins if g.get("optimal") is not None]
        rows.append(
            {
                "agent": agent,
                "games": len(played),
                "wins": len(wins),
                "success_rate": len(wins) / len(played),
                "mean_clicks": _mean([g["clicks"] for g in wins]),
                "median_clicks": statistics.median([g["clicks"] for g in wins]) if wins else None,
                "mean_extra_clicks": _mean([g["clicks"] - g["optimal"] for g in scored]),
                "optimal_rate": (sum(g["clicks"] == g["optimal"] for g in scored) / len(scored)) if scored else None,
                "mean_seconds": _mean([g["seconds"] for g in played]),
                "mean_seconds_per_click": _mean([g["seconds"] / g["clicks"] for g in played if g["clicks"]]),
                "mean_llm_calls": _mean([g["llm_calls"] for g in played]),
                "mean_pages_loaded": _mean([g["pages_loaded"] for g in played]),
                "errors": sum(g["outcome"].startswith("error") for g in played),
            }
        )
    return rows


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summary_markdown(rows: Sequence[dict[str, Any]]) -> str:
    def fmt(value: Any, spec: str) -> str:
        return "n/a" if value is None else format(value, spec)

    lines = [
        "| Agent | Success | Mean clicks (wins) | Extra clicks vs optimal | Optimal paths | Time / game | LLM calls / game | Pages loaded / game |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['agent']}` | {r['wins']}/{r['games']} ({r['success_rate']:.0%}) "
            f"| {fmt(r['mean_clicks'], '.1f')} | {fmt(r['mean_extra_clicks'], '+.1f')} "
            f"| {fmt(r['optimal_rate'], '.0%')} | {fmt(r['mean_seconds'], '.1f')}s "
            f"| {fmt(r['mean_llm_calls'], '.1f')} | {fmt(r['mean_pages_loaded'], '.1f')} |"
        )
    return "\n".join(lines)
