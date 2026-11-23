"""WikipediaClient against canned API responses (no network)."""

import pytest

from wikirace.wiki import PageNotFound, WikiError, WikipediaClient


class FakeResponse:
    def __init__(self, payload, status=200, headers=None):
        self._payload = payload
        self.status_code = status
        self.ok = status < 400
        self.headers = headers or {}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def get(self, url, params, timeout):
        self.calls.append(params)
        item = self.responses.pop(0)
        return item if isinstance(item, FakeResponse) else FakeResponse(item)


def client(*responses):
    session = FakeSession(*responses)
    return WikipediaClient(session=session, min_interval=0), session


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("wikirace.wiki.time.sleep", lambda s: None)


def test_links_follow_continuation_and_drop_red_links_and_self_links():
    wiki, session = client(
        {
            "continue": {"gplcontinue": "123|0|M", "continue": "gplcontinue||"},
            "query": {
                "pages": [
                    {"title": "Tomato", "ns": 0, "description": "Edible berry"},
                    {"title": "Nonexistent page", "ns": 0, "missing": True},
                    {"title": "Pizza", "ns": 0},
                ]
            },
        },
        {"query": {"pages": [{"title": "Italy", "ns": 0, "description": "Country in Europe"}]}},
    )
    links = wiki.links("Pizza")
    assert [(l.title, l.description) for l in links] == [
        ("Italy", "Country in Europe"),
        ("Tomato", "Edible berry"),
    ]
    assert session.calls[0]["generator"] == "links" and session.calls[0]["redirects"] == 1
    assert session.calls[1]["gplcontinue"] == "123|0|M"
    assert wiki.network_requests == 2


def test_answers_are_cached():
    wiki, session = client({"query": {"pages": [{"title": "Italy", "ns": 0}]}})
    assert wiki.links("Pizza") == wiki.links("Pizza")
    assert len(session.calls) == 1


def test_resolve_follows_redirects_and_reports_missing_pages():
    wiki, session = client(
        {"query": {"redirects": [{"from": "USA", "to": "United States"}], "pages": [{"title": "United States"}]}},
        {"query": {"pages": [{"title": "Atlantis xyz", "missing": True}]}},
    )
    assert wiki.resolve("USA") == "United States"
    with pytest.raises(PageNotFound):
        wiki.resolve("Atlantis xyz")
    with pytest.raises(PageNotFound):  # the miss is cached too
        wiki.resolve("Atlantis xyz")
    assert len(session.calls) == 2


def test_retries_on_server_errors():
    wiki, session = client(
        FakeResponse({}, status=503),
        FakeResponse({}, status=429, headers={"Retry-After": "1"}),
        {"query": {"pages": [{"title": "Italy", "description": "Country"}]}},
    )
    assert wiki.describe("Italy") == "Country"
    assert len(session.calls) == 3


class NotJson(FakeResponse):
    def json(self):
        raise ValueError("Expecting value")


def test_non_json_replies_are_retried():
    wiki, session = client(
        NotJson(None),
        {"error": {"code": "maxlag", "info": "Waiting for a database server"}},
        {"query": {"pages": [{"title": "Italy", "description": "Country"}]}},
    )
    assert wiki.describe("Italy") == "Country"
    assert wiki.retries == 2


def test_api_errors_raise():
    wiki, _ = client({"error": {"code": "badvalue", "info": "Unrecognized value"}})
    with pytest.raises(WikiError, match="badvalue"):
        wiki.describe("Italy")


def test_gives_up_after_max_retries():
    session = FakeSession(*[FakeResponse({}, status=503) for _ in range(3)])
    wiki = WikipediaClient(session=session, min_interval=0, max_retries=3)
    with pytest.raises(WikiError, match="after 3 attempts"):
        wiki.describe("Italy")


def test_backlinks_include_pages_that_link_through_redirects():
    wiki, _ = client(
        {
            "query": {
                "backlinks": [
                    {"title": "Albedo"},
                    {"title": "Photosynthesise", "redirect": True, "redirlinks": [{"title": "Leaf"}]},
                ]
            }
        }
    )
    titles, complete = wiki.backlinks("Photosynthesis")
    assert titles == ["Albedo", "Leaf"] and complete


def test_backlinks_respect_the_limit():
    wiki, _ = client(
        {"continue": {"blcontinue": "x"}, "query": {"backlinks": [{"title": "A"}, {"title": "B"}]}},
    )
    titles, complete = wiki.backlinks("T", limit=2)
    assert titles == ["A", "B"] and not complete


def test_pages_linking_to_checks_fifty_titles_per_request():
    sources = [f"Page {i:03d}" for i in range(60)]
    wiki, session = client(
        {"query": {"pages": [{"title": "Page 007", "links": [{"title": "T"}]}, {"title": "Page 008"}]}},
        {"query": {"pages": [{"title": "Page 055", "links": [{"title": "T"}]}]}},
    )
    assert wiki.pages_linking_to(sources, ["T", "Tee"]) == {"Page 007", "Page 055"}
    assert len(session.calls[0]["titles"].split("|")) == 50
    assert session.calls[0]["pltitles"] == "T|Tee"


def test_search_returns_titles_best_first():
    wiki, session = client({"query": {"search": [{"title": "Albert Einstein"}, {"title": "Einstein family"}]}})
    assert wiki.search("albert einstien") == ["Albert Einstein", "Einstein family"]
    assert session.calls[0]["list"] == "search" and session.calls[0]["srsearch"] == "albert einstien"


def test_search_follows_the_spelling_suggestion():
    wiki, session = client(
        {"query": {"searchinfo": {"suggestion": "mount everest"}, "search": [{"title": "Rolando Garibotti"}]}},
        {"query": {"search": [{"title": "Mount Everest"}, {"title": "Rolando Garibotti"}]}},
    )
    assert wiki.search("mount everst") == ["Mount Everest", "Rolando Garibotti"]
    assert session.calls[1]["srsearch"] == "mount everest"


def test_user_agent_override(monkeypatch):
    monkeypatch.setenv("WIKIRACE_USER_AGENT", "my-bot/1.0 (me@example.org)")
    wiki, session = client()
    assert session.headers["User-Agent"] == "my-bot/1.0 (me@example.org)"


def test_user_agent_includes_contact_details(monkeypatch):
    monkeypatch.delenv("WIKIRACE_USER_AGENT", raising=False)
    monkeypatch.setenv("WIKIRACE_CONTACT", "https://example.org/me")
    wiki, session = client()
    assert session.headers["User-Agent"] == "wikirace-ai/0.1 (https://example.org/me; educational bot)"
