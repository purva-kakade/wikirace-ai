from conftest import FakeWiki
from wikirace.oracle import shortest_path

GRAPH = {
    "S": ["A", "B", "C"],
    "A": ["D"],
    "B": ["E", "F"],
    "C": ["G"],
    "D": [],
    "E": [],
    "F": ["T"],
    "G": [],
    "T": [],
    "X": ["T"],
}


def test_zero_and_one_click():
    wiki = FakeWiki({"S": ["T"], "T": []})
    assert shortest_path(wiki, "S", "S").length == 0
    assert shortest_path(wiki, "S", "T").path == ["S", "T"]


def test_two_clicks():
    wiki = FakeWiki({"S": ["A", "B"], "A": [], "B": ["T"], "T": []})
    result = shortest_path(wiki, "S", "T")
    assert result.path == ["S", "B", "T"] and result.length == 2


def test_two_clicks_through_a_redirect():
    # B links to "Tee", which redirects to T: still a 2-click path.
    wiki = FakeWiki({"S": ["B"], "B": ["Tee"], "T": []}, redirects={"Tee": "T"})
    assert shortest_path(wiki, "S", "T").length == 2


def test_three_clicks():
    result = shortest_path(FakeWiki(GRAPH), "S", "T")
    assert result.path == ["S", "B", "F", "T"]
    assert result.length == 3
    assert result.lower_bound == 3


def test_no_short_path_gives_a_proven_lower_bound():
    wiki = FakeWiki({"S": ["A"], "A": ["B"], "B": ["C"], "C": ["T"], "T": []})
    result = shortest_path(wiki, "S", "T")
    assert result.path is None and result.length is None
    assert result.lower_bound == 4


def test_running_out_of_budget_only_proves_three():
    result = shortest_path(FakeWiki(GRAPH), "S", "T", budget=1)
    assert result.path is None
    assert result.lower_bound == 3
    assert result.pages_expanded == 1


def test_similarity_ordering_finds_the_path_first(scorer):
    graph = {
        "Start": ["Zebra", "Yak", "Quantum physics"],
        "Zebra": [],
        "Yak": [],
        "Quantum physics": ["Quantum field theory"],
        "Quantum field theory": ["Quantum mechanics"],
        "Quantum mechanics": [],
    }
    result = shortest_path(FakeWiki(graph), "Start", "Quantum mechanics", budget=1, scorer=scorer)
    assert result.length == 3
