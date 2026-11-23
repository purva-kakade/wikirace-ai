"""Command-line interface: ``wikirace play | shortest | bench | report``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .agents import AGENT_NAMES, NEEDS_EMBEDDINGS, NEEDS_LLM, Agent, build_agent
from .benchmark import load_scenarios, run_benchmark, summarize, summary_markdown
from .cache import DiskCache
from .embed import DEFAULT_EMBED_MODEL, SemanticScorer
from .game import Step, play
from .llm import DEFAULT_MODEL, LLMError, OllamaLLM
from .oracle import shortest_path
from .wiki import WikiError, WikipediaClient

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")


def _wiki(args: argparse.Namespace) -> WikipediaClient:
    if not (os.environ.get("WIKIRACE_CONTACT") or os.environ.get("WIKIRACE_USER_AGENT")):
        print(
            "note: Wikipedia allows 10 requests/minute without contact details in the User-Agent.\n"
            "      Set WIKIRACE_CONTACT to an email or URL to get 200/minute.",
            file=sys.stderr,
        )
    return WikipediaClient(lang=args.lang, cache=DiskCache(args.cache))


def _scorer(args: argparse.Namespace) -> SemanticScorer:
    from .embed import SentenceTransformerEmbedder

    print(f"Loading embedding model {args.embedder} ...", file=sys.stderr)
    return SemanticScorer(SentenceTransformerEmbedder(args.embedder), use_descriptions=not args.titles_only)


def _agents(names: list[str], args: argparse.Namespace, wiki: WikipediaClient) -> tuple[list[Agent], SemanticScorer | None]:
    scorer = _scorer(args) if any(n in NEEDS_EMBEDDINGS for n in names) else None
    llm = None
    if any(n in NEEDS_LLM for n in names):
        llm = OllamaLLM(model=args.model)
        llm.ensure_ready()
    agents = [
        build_agent(n, wiki=wiki, scorer=scorer, llm=llm, shortlist=args.shortlist, lookahead_k=args.lookahead_k)
        for n in names
    ]
    return agents, scorer


def cmd_play(args: argparse.Namespace) -> None:
    wiki = _wiki(args)
    if args.start and args.target:
        (agent,), _ = _agents([args.agent], args, wiki)
        _play_and_show(wiki, agent, args.start, args.target, args, chrome=args.chrome)
    else:
        _interactive(wiki, args)


def _play_and_show(
    wiki, agent: Agent, start: str, target: str, args: argparse.Namespace, chrome: bool = False
) -> None:
    view = None
    if chrome:
        from .browser import ChromeView

        try:
            view = ChromeView(lang=args.lang, delay=args.delay)
            view.start(wiki.resolve(start))
        except WikiError:
            view = None  # the game itself will report the bad title

    def show(step: Step) -> None:
        print(f"  {step.from_page} -> {step.to_page}   ({step.seconds:.1f}s, {step.options} options)")
        if step.note and args.verbose:
            print(f"      {step.note}")
        if view:
            view.move(step.to_page)

    print(f"\n{start} -> {target}   [agent: {agent.name}]" + ("   (watch it in Chrome)" if view else "") + "\n")
    result = play(wiki, agent, start, target, max_steps=args.max_steps, on_step=show)
    verdict = "WIN" if result.success else f"LOSS ({result.outcome})"
    print(
        f"\n{verdict}: {result.clicks} clicks in {result.seconds:.1f}s, "
        f"{result.pages_loaded} pages loaded, {result.llm_calls} LLM calls"
    )


def _ask(prompt: str, default: str | None = None) -> str:
    shown = f"{prompt} [{default}]: " if default else f"{prompt}: "
    return input(shown).strip() or (default or "")


def _ask_title(wiki, prompt: str) -> str | None:
    """Ask for an article title until it exists, offering the closest match for typos."""
    while True:
        text = _ask(prompt)
        if not text:
            continue
        try:
            return wiki.resolve(text)
        except WikiError:
            pass
        matches = wiki.search(text)
        if not matches:
            print(f"  No Wikipedia article matches {text!r}. Try again.")
            continue
        answer = _ask(f"  No article called {text!r}. Did you mean {matches[0]!r}? (y/n)", "y")
        if answer.lower().startswith("y"):
            return wiki.resolve(matches[0])
        if len(matches) > 1:
            print("  Other matches: " + ", ".join(matches[1:]))


def _interactive(wiki, args: argparse.Namespace) -> None:
    """Prompt for start, target and agent; play; repeat. Models load once, on first use."""
    print("\nWikipedia Game: pick two articles and an agent. Ctrl-C quits.\n")
    print(
        "Agents: random | greedy (embeddings) | lookahead (previews 5 pages) | "
        "llm (local LLM picks) | llm-cot (LLM explains its reasoning)\n"
    )
    scorer: SemanticScorer | None = None
    llm: OllamaLLM | None = None
    agents: dict[str, Agent] = {}
    agent_name = args.agent
    args.verbose = True  # interactive play always shows the agents' reasoning
    try:
        while True:
            start = _ask_title(wiki, "Start article")
            target = _ask_title(wiki, "Target article")
            while True:
                agent_name = _ask("Agent", agent_name)
                if agent_name in AGENT_NAMES:
                    break
                print(f"  Choose one of: {', '.join(AGENT_NAMES)}")
            if agent_name not in agents:
                if agent_name in NEEDS_EMBEDDINGS and scorer is None:
                    scorer = _scorer(args)
                if agent_name in NEEDS_LLM and llm is None:
                    llm = OllamaLLM(model=args.model)
                    llm.ensure_ready()
                agents[agent_name] = build_agent(
                    agent_name, wiki=wiki, scorer=scorer, llm=llm, shortlist=args.shortlist, lookahead_k=args.lookahead_k
                )
            chrome = _ask("Watch it in Chrome? (y/n)", "y" if args.chrome or sys.platform == "darwin" else "n")
            _play_and_show(wiki, agents[agent_name], start, target, args, chrome=chrome.lower().startswith("y"))
            if not _ask("\nPlay again? (y/n)", "y").lower().startswith("y"):
                return
            print()
    except EOFError:
        print()


def cmd_shortest(args: argparse.Namespace) -> None:
    wiki = _wiki(args)
    scorer = None if args.no_embeddings else _scorer(args)
    result = shortest_path(wiki, args.start, args.target, budget=args.budget, scorer=scorer)
    if result.path:
        print(f"Shortest path ({result.length} clicks): {' -> '.join(result.path)}")
    else:
        print(f"No path of 3 clicks or fewer found; the shortest path is at least {result.lower_bound} clicks.")
    print(
        f"({result.pages_expanded} second-hop pages expanded, "
        f"{wiki.network_requests} network requests, {wiki.retries} retries)"
    )


def cmd_bench(args: argparse.Namespace) -> None:
    wiki = _wiki(args)
    names = [n.strip() for n in args.agents.split(",") if n.strip()]
    agents, scorer = _agents(names, args, wiki)
    scenarios = load_scenarios(args.scenarios)[: args.limit]
    meta = {
        "llm_model": args.model if any(n in NEEDS_LLM for n in names) else None,
        "embedding_model": args.embedder,
        "use_descriptions": not args.titles_only,
        "shortlist": args.shortlist,
        "lookahead_k": args.lookahead_k,
        "scenarios_file": str(args.scenarios),
    }
    report = run_benchmark(
        wiki,
        agents,
        scenarios,
        max_steps=args.max_steps,
        oracle_budget=args.oracle_budget or None,
        scorer=scorer,
        out_path=args.out,
        meta=meta,
    )
    print(
        f"\nSaved {len(report['games'])} games to {args.out} "
        f"({wiki.network_requests} network requests, {wiki.retries} retries)\n"
    )
    _write_summary(report, Path(args.out))


def cmd_report(args: argparse.Namespace) -> None:
    report = json.loads(Path(args.results).read_text())
    _write_summary(report, Path(args.results))


def _write_summary(report: dict, results_path: Path) -> None:
    rows = summarize(report["games"])
    table = summary_markdown(rows)
    print(table)
    summary_path = results_path.with_name(results_path.stem + "_summary.md")
    summary_path.write_text(table + "\n")
    print(f"\nWrote {summary_path}")
    try:
        from .report import make_charts
    except ImportError:
        print("Install matplotlib (pip install 'wikirace-ai[report]') to also draw charts.")
        return
    for path in make_charts(report, rows, results_path.with_suffix(".png")):
        print(f"Wrote {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wikirace", description="AI agents that play the Wikipedia Game.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--cache", default=".cache/wikipedia.sqlite", help="SQLite file for cached API answers")
        p.add_argument("--lang", default="en", help="Wikipedia language edition (default: en)")
        p.add_argument("--embedder", default=DEFAULT_EMBED_MODEL, help="sentence-transformers model")
        p.add_argument("--titles-only", action="store_true", help="embed bare titles, without short descriptions")

    def models(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model for LLM agents (default: {DEFAULT_MODEL})")
        p.add_argument("--shortlist", type=int, default=8, help="links shown to the LLM per move (default: 8)")
        p.add_argument("--lookahead-k", type=int, default=5, help="pages the lookahead agent previews (default: 5)")
        p.add_argument("--max-steps", type=int, default=20, help="click limit per game (default: 20)")

    p = sub.add_parser("play", help="play a game and print each click (no arguments: interactive prompts)")
    p.add_argument("start", nargs="?", help="start article (omit both to be prompted)")
    p.add_argument("target", nargs="?", help="target article")
    p.add_argument("--agent", choices=AGENT_NAMES, default="lookahead")
    p.add_argument("-v", "--verbose", action="store_true", help="show each agent's reasoning")
    p.add_argument("--chrome", action="store_true", help="open the start page in Chrome and each click in a new tab")
    p.add_argument("--delay", type=float, default=1.5, help="seconds to pause after each Chrome tab (default: 1.5)")
    common(p)
    models(p)
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("shortest", help="find the true shortest path (up to 3 clicks)")
    p.add_argument("start")
    p.add_argument("target")
    p.add_argument("--budget", type=int, default=400, help="max second-hop pages to expand (default: 400)")
    p.add_argument("--no-embeddings", action="store_true", help="expand pages in link order instead of by similarity")
    common(p)
    p.set_defaults(func=cmd_shortest)

    p = sub.add_parser("bench", help="run agents on a scenario file and score them against the oracle")
    p.add_argument("--scenarios", default="scenarios/curated.json")
    p.add_argument("--agents", default=",".join(AGENT_NAMES), help="comma-separated agent names")
    p.add_argument("--out", default="results/benchmark.json")
    p.add_argument("--oracle-budget", type=int, default=200, help="0 disables the shortest-path oracle (default: 200)")
    p.add_argument("--limit", type=int, default=None, help="only run the first N scenarios")
    common(p)
    models(p)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("report", help="rebuild the summary table and charts from a results file")
    p.add_argument("results", nargs="?", default="results/benchmark.json")
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (WikiError, LLMError) as exc:
        sys.exit(f"error: {exc}")
    except KeyboardInterrupt:
        sys.exit("interrupted")


if __name__ == "__main__":
    main()
