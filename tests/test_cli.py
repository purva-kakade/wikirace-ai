"""Interactive `wikirace play` with scripted keyboard input."""

from conftest import FakeWiki
from wikirace import cli

GRAPH = {"Pizza": ["Italy", "Physics"], "Italy": [], "Physics": ["Albert Einstein"], "Albert Einstein": []}


def run_interactive(monkeypatch, answers, wiki=None):
    replies = iter(answers)
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr("builtins.input", fake_input)
    args = cli.build_parser().parse_args(["play", "--agent", "random"])
    cli._interactive(wiki or FakeWiki(GRAPH), args)
    return prompts


def test_play_without_titles_prompts_for_them(monkeypatch, capsys):
    prompts = run_interactive(monkeypatch, ["Pizza", "Albert Einstein", "", "n"])
    out = capsys.readouterr().out
    assert prompts[:3] == ["Start article: ", "Target article: ", "Agent [random]: "]
    assert "Pizza -> Albert Einstein   [agent: random]" in out
    assert "WIN" in out or "LOSS" in out


def test_typos_get_a_suggestion(monkeypatch, capsys):
    prompts = run_interactive(monkeypatch, ["Einstien Albert", "y", "Pizza", "random", "n"], FakeWiki(GRAPH))
    assert "Did you mean 'Albert Einstein'?" in prompts[1]
    assert "Albert Einstein -> Pizza" in capsys.readouterr().out


def test_unknown_agent_is_asked_again(monkeypatch, capsys):
    prompts = run_interactive(monkeypatch, ["Pizza", "Italy", "psychic", "random", "n"])
    assert prompts.count("Agent [random]: ") == 1
    assert "Choose one of" in capsys.readouterr().out


def test_ctrl_d_exits_cleanly(monkeypatch):
    def eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    cli._interactive(FakeWiki(GRAPH), cli.build_parser().parse_args(["play"]))


def test_both_titles_on_the_command_line_skip_the_prompts():
    args = cli.build_parser().parse_args(["play", "Pizza", "Albert Einstein"])
    assert (args.start, args.target) == ("Pizza", "Albert Einstein")
