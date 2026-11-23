import json

import pytest

from conftest import FakeWiki
from wikirace.agents import GreedyAgent, RandomAgent
from wikirace.benchmark import Scenario, load_scenarios, run_benchmark, summarize, summary_markdown
from wikirace.llm import LLMError, LLMReplyError, OllamaLLM, parse_json_object

GRAPH = {
    "Start": ["Apple", "Quantum physics"],
    "Quantum physics": ["Quantum field theory"],
    "Quantum field theory": ["Quantum mechanics"],
    "Apple": [],
    "Quantum mechanics": [],
}


def game(agent, success, clicks, optimal, seconds=1.0, outcome=None):
    return {
        "agent": agent,
        "success": success,
        "clicks": clicks,
        "optimal": optimal,
        "seconds": seconds,
        "llm_calls": 0,
        "pages_loaded": clicks,
        "outcome": outcome or ("reached target" if success else "step limit"),
    }


def test_summarize():
    rows = summarize(
        [
            game("a", True, 3, 3),
            game("a", True, 5, 3),
            game("a", False, 30, 3),
            game("b", True, 4, None),
        ]
    )
    a, b = rows
    assert a["agent"] == "a" and a["games"] == 3 and a["wins"] == 2
    assert a["success_rate"] == pytest.approx(2 / 3)
    assert a["mean_clicks"] == 4
    assert a["mean_extra_clicks"] == 1
    assert a["optimal_rate"] == 0.5
    assert b["mean_extra_clicks"] is None and b["optimal_rate"] is None
    table = summary_markdown(rows)
    assert "| `a` | 2/3 (67%)" in table and "n/a" in table


def test_run_benchmark_saves_every_game(tmp_path, scorer):
    out = tmp_path / "results.json"
    scenarios = [Scenario("Start", "Quantum mechanics", "test"), Scenario("Atlantis", "Apple")]
    report = run_benchmark(
        FakeWiki(GRAPH),
        [GreedyAgent(scorer), RandomAgent()],
        scenarios,
        max_steps=5,
        scorer=scorer,
        out_path=out,
        log=lambda msg: None,
    )
    saved = json.loads(out.read_text())
    assert saved == report
    assert len(report["games"]) == 2  # the missing "Atlantis" scenario is skipped
    assert report["scenarios"][0]["optimal"] == 3
    greedy = report["games"][0]
    assert greedy["success"] and greedy["clicks"] == 3 and greedy["optimal"] == 3


def test_charts_are_written(tmp_path):
    pytest.importorskip("matplotlib")
    from wikirace.report import make_charts

    games = [game("a", True, 3, 3) | {"start": "S", "target": "T"}, game("b", False, 9, 3) | {"start": "S", "target": "T"}]
    (path,) = make_charts({"games": games, "meta": {"llm_model": "x"}}, summarize(games), tmp_path / "run.png")
    assert path == tmp_path / "run.png"
    assert path.exists() and path.stat().st_size > 10_000


def test_load_scenarios(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps([{"start": "A", "target": "B", "category": "homonym"}, {"start": "C", "target": "D"}]))
    assert load_scenarios(path) == [Scenario("A", "B", "homonym"), Scenario("C", "D", "general")]


def test_parse_json_object_tolerates_surrounding_text():
    assert parse_json_object('{"choice": 2}') == {"choice": 2}
    assert parse_json_object('Sure! {"choice": 3} hope that helps') == {"choice": 3}
    with pytest.raises(LLMReplyError):
        parse_json_object("I pick option two")
    with pytest.raises(LLMReplyError):
        parse_json_object("[1, 2]")


class FakeOllama:
    def __init__(self, status=200, payload=None):
        self.status = status
        self.payload = payload or {}
        self.sent = None

    def post(self, url, json, timeout):
        self.sent = json
        return type("R", (), {"status_code": self.status, "ok": self.status < 400, "text": "", "json": lambda s: self.payload})()


def test_ollama_sends_schema_and_parses_reply():
    session = FakeOllama(payload={"message": {"content": '{"choice": 1}'}})
    llm = OllamaLLM(model="qwen2.5:7b", host="localhost:11434", session=session)
    assert llm.host == "http://localhost:11434"
    assert llm.complete_json("sys", "prompt", {"type": "object"}) == {"choice": 1}
    assert session.sent["format"] == {"type": "object"}
    assert session.sent["options"]["temperature"] == 0.0
    assert session.sent["stream"] is False


def test_ollama_missing_model_has_a_helpful_error():
    llm = OllamaLLM(model="nope", session=FakeOllama(status=404))
    with pytest.raises(LLMError, match="ollama pull nope"):
        llm.complete_json("sys", "prompt", {})
