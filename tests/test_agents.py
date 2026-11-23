import pytest

from conftest import FakeWiki, ScriptedLLM
from wikirace.agents import AGENT_NAMES, GreedyAgent, LLMAgent, LookaheadAgent, build_agent
from wikirace.agents.base import Turn
from wikirace.llm import LLMReplyError
from wikirace.wiki import Link

TARGET = Link("Quantum mechanics", "Description of nature at the scale of atoms")


def turn(*titles: str, path=("Start",)) -> Turn:
    return Turn(path[-1], TARGET, tuple(path), tuple(Link(t) for t in titles))


def test_greedy_picks_the_most_similar_link(scorer):
    agent = GreedyAgent(scorer)
    assert agent.choose(turn("Banana", "Quantum physics", "Opera")).title == "Quantum physics"


def test_descriptions_break_ties_between_homonyms(scorer):
    target = Link("Java", "Island of Indonesia")
    options = (
        Link("Java (programming language)", "Object-oriented programming language"),
        Link("Sumatra", "Island of Indonesia"),
    )
    pick = GreedyAgent(scorer).choose(Turn("Start", target, ("Start",), options))
    assert pick.title == "Sumatra"


def test_lookahead_prefers_a_page_that_links_to_the_target(scorer):
    wiki = FakeWiki({"Quantum foam": ["Bubble"], "Physics": ["Quantum mechanics"]})
    agent = LookaheadAgent(wiki, scorer, k=2)
    # Greedy would take "Quantum foam" (shares a word); lookahead sees Physics -> target.
    assert GreedyAgent(scorer).choose(turn("Quantum foam", "Physics")).title == "Quantum foam"
    assert agent.choose(turn("Quantum foam", "Physics")).title == "Physics"
    assert agent.peeks == 2


def test_lookahead_scores_the_next_click(scorer):
    wiki = FakeWiki(
        {
            "Quantum dot": ["Television"],
            "Science": ["Quantum theory", "Biology"],
        }
    )
    agent = LookaheadAgent(wiki, scorer, k=2, alpha=0.3)
    assert agent.choose(turn("Quantum dot", "Science")).title == "Science"


def test_lookahead_ignores_pages_already_visited(scorer):
    wiki = FakeWiki({"Science": ["Quantum nature scale"], "Quantum dot": ["Quantum field"]})
    agent = LookaheadAgent(wiki, scorer, k=2, alpha=0.0)
    # Science's only onward link is very similar to the target but already visited,
    # so it must not count; Quantum dot's weaker onward link wins.
    pick = agent.choose(turn("Science", "Quantum dot", path=("Start", "Quantum nature scale", "Hub")))
    assert pick.title == "Quantum dot"


def test_llm_agent_maps_the_choice_number_to_a_link(scorer):
    llm = ScriptedLLM([{"choice": 2}])
    agent = LLMAgent(scorer, llm, shortlist=3)
    pick = agent.choose(turn("Quantum physics", "Banana", "Physics", "Opera", path=("Start", "Fruit")))
    prompt = llm.prompts[0]
    assert "Target article: Quantum mechanics: Description of nature at the scale of atoms" in prompt
    assert "Path so far: Start -> Fruit" in prompt
    assert "1. Quantum physics" in prompt
    assert "4." not in prompt  # only the shortlist is shown
    assert pick.title == prompt.split("2. ")[1].split("\n")[0]
    assert agent.llm_calls == 1
    assert llm.schemas[0]["properties"]["choice"]["maximum"] == 3


@pytest.mark.parametrize("reply", [{"choice": 9}, {"choice": "two"}, {}])
def test_llm_agent_falls_back_to_the_top_match_on_bad_replies(scorer, reply):
    agent = LLMAgent(scorer, ScriptedLLM([reply]), shortlist=3)
    assert agent.choose(turn("Banana", "Quantum physics", "Opera")).title == "Quantum physics"
    assert agent.fallbacks == 1


def test_llm_agent_survives_non_json_replies(scorer):
    def garbage(prompt):
        raise LLMReplyError("not json")

    agent = LLMAgent(scorer, ScriptedLLM(garbage))
    assert agent.choose(turn("Banana", "Quantum physics")).title == "Quantum physics"
    assert agent.fallbacks == 1


def test_chain_of_thought_agent_asks_for_reasoning_and_keeps_it(scorer):
    llm = ScriptedLLM([{"reasoning": "Physics is the parent field.", "choice": 1}])
    agent = LLMAgent(scorer, llm, reasoning=True)
    agent.choose(turn("Banana", "Quantum physics"))
    assert agent.name == "llm-cot"
    assert llm.schemas[0]["required"] == ["reasoning", "choice"]
    assert agent.note == "Physics is the parent field."


def test_single_option_skips_the_llm(scorer):
    llm = ScriptedLLM([])
    agent = LLMAgent(scorer, llm)
    assert agent.choose(turn("Only option")).title == "Only option"
    assert agent.llm_calls == 0


def test_build_agent_knows_every_name(scorer):
    wiki = FakeWiki({})
    llm = ScriptedLLM([])
    built = {name: build_agent(name, wiki=wiki, scorer=scorer, llm=llm) for name in AGENT_NAMES}
    assert {name: agent.name for name, agent in built.items()} == {n: n for n in AGENT_NAMES}


def test_build_agent_validates_requirements(scorer):
    wiki = FakeWiki({})
    with pytest.raises(ValueError, match="Unknown agent"):
        build_agent("psychic", wiki=wiki)
    with pytest.raises(ValueError, match="SemanticScorer"):
        build_agent("greedy", wiki=wiki)
    with pytest.raises(ValueError, match="LLM"):
        build_agent("llm", wiki=wiki, scorer=scorer)
