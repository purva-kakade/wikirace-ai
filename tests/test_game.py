import pytest

from conftest import FakeWiki, ScriptedLLM
from wikirace.agents import GreedyAgent, LLMAgent, RandomAgent
from wikirace.agents.base import Agent
from wikirace.game import play
from wikirace.llm import LLMError
from wikirace.wiki import Link

GRAPH = {
    "Start": ["Apple", "Banana", "Quantum physics"],
    "Quantum physics": ["Apple", "Quantum mechanics"],
    "Apple": ["Banana"],
    "Banana": ["Apple"],
    "Quantum mechanics": [],
}


class NeverAsked(Agent):
    name = "never-asked"

    def choose(self, turn):
        raise AssertionError("agent should not have been asked")


class Cheater(Agent):
    name = "cheater"

    def choose(self, turn):
        return Link(turn.path[0])  # tries to go back to the start page


class Broken(Agent):
    name = "broken"

    def choose(self, turn):
        raise LLMError("ollama is down")


def test_greedy_reaches_target(scorer):
    result = play(FakeWiki(GRAPH), GreedyAgent(scorer), "Start", "Quantum mechanics")
    assert result.success
    assert result.path == ["Start", "Quantum physics", "Quantum mechanics"]
    assert result.clicks == 2
    assert result.outcome == "reached target"
    assert result.pages_loaded == 2


def test_target_on_page_is_clicked_without_asking_the_agent():
    wiki = FakeWiki({"Start": ["Elsewhere", "Goal"], "Goal": [], "Elsewhere": []})
    result = play(wiki, NeverAsked(), "Start", "Goal")
    assert result.success and result.clicks == 1
    assert result.steps[0].note == "target is linked from this page"


def test_pages_are_never_revisited_and_dead_ends_lose(scorer):
    # Apple <-> Banana is a loop; once both are visited there is nowhere left to go.
    wiki = FakeWiki({"Start": ["Apple"], "Apple": ["Banana"], "Banana": ["Apple"], "Goal": []})
    result = play(wiki, GreedyAgent(scorer), "Start", "Goal")
    assert not result.success
    assert result.outcome == "dead end"
    assert result.path == ["Start", "Apple", "Banana"]
    assert len(set(result.path)) == len(result.path)


def test_step_limit():
    chain = {f"P{i}": [f"P{i + 1}"] for i in range(10)} | {"P10": [], "Goal": []}
    result = play(FakeWiki(chain), RandomAgent(), "P0", "Goal", max_steps=3)
    assert not result.success
    assert result.outcome == "step limit"
    assert result.clicks == 3


def test_start_equal_to_target_is_an_instant_win():
    result = play(FakeWiki({"Same": []}), NeverAsked(), "Same", "Same")
    assert result.success and result.clicks == 0


def test_redirects_are_resolved_before_playing(scorer):
    wiki = FakeWiki(GRAPH, redirects={"QM": "Quantum mechanics"})
    result = play(wiki, GreedyAgent(scorer), "Start", "QM")
    assert result.success and result.target == "Quantum mechanics"


def test_illegal_moves_are_rejected():
    with pytest.raises(ValueError, match="not a legal move"):
        play(FakeWiki(GRAPH), Cheater(), "Start", "Quantum mechanics")


def test_service_errors_end_the_game_cleanly():
    result = play(FakeWiki(GRAPH), Broken(), "Start", "Quantum mechanics")
    assert not result.success
    assert result.outcome == "error: ollama is down"


def test_missing_page_is_reported_as_error():
    result = play(FakeWiki(GRAPH), RandomAgent(), "Atlantis", "Quantum mechanics")
    assert not result.success and result.outcome.startswith("error")


def test_llm_calls_are_counted(scorer):
    llm = ScriptedLLM(lambda prompt: {"choice": 1})
    result = play(FakeWiki(GRAPH), LLMAgent(scorer, llm), "Start", "Quantum mechanics")
    assert result.success
    assert result.llm_calls == len(llm.prompts) == 1


def test_result_serialises_with_click_count(scorer):
    data = play(FakeWiki(GRAPH), GreedyAgent(scorer), "Start", "Quantum mechanics").to_dict()
    assert data["clicks"] == 2
    assert data["steps"][0]["to_page"] == "Quantum physics"
