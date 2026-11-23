"""ChromeView with a fake AppleScript runner: no real browser is opened."""

from wikirace.browser import BrowserError, ChromeView, article_url


class FakeChrome:
    def __init__(self, fail_after=None):
        self.scripts = []
        self.fail_after = fail_after

    def __call__(self, lines):
        self.scripts.append("\n".join(lines))
        if self.fail_after is not None and len(self.scripts) > self.fail_after:
            raise BrowserError("Not authorised to send Apple events to Google Chrome.")
        return "4242"


def test_article_urls():
    assert article_url("Ancient Egypt") == "https://en.wikipedia.org/wiki/Ancient_Egypt"
    assert article_url("Java (programming language)") == "https://en.wikipedia.org/wiki/Java_(programming_language)"
    assert article_url("AC/DC", "de") == "https://de.wikipedia.org/wiki/AC/DC"
    assert article_url('Say "hi"') == "https://en.wikipedia.org/wiki/Say_%22hi%22"  # quotes can't break the script


def test_start_opens_a_new_window_and_moves_add_tabs_to_it():
    chrome = FakeChrome()
    view = ChromeView(delay=0, runner=chrome)
    view.start("Pizza")
    view.move("Italy")
    view.move("Albert Einstein")
    assert view.window_id == 4242
    assert "make new window" in chrome.scripts[0] and "/wiki/Pizza" in chrome.scripts[0]
    assert "tell window id 4242" in chrome.scripts[1] and "/wiki/Italy" in chrome.scripts[1]
    assert "/wiki/Albert_Einstein" in chrome.scripts[2]
    assert view.tabs_opened == 3


def test_falls_back_to_plain_opening_without_permission():
    opened = []
    view = ChromeView(delay=0, runner=FakeChrome(fail_after=0), opener=opened.append)
    view.start("Pizza")
    view.move("Italy")
    assert opened == ["https://en.wikipedia.org/wiki/Pizza", "https://en.wikipedia.org/wiki/Italy"]


def test_a_closed_window_is_replaced():
    class ClosedWindow(FakeChrome):
        def __call__(self, lines):
            script = "\n".join(lines)
            self.scripts.append(script)
            if "tell window id" in script:
                raise BrowserError("Can't get window id 4242.")
            return "777"

    chrome = ClosedWindow()
    view = ChromeView(delay=0, runner=chrome)
    view.start("Pizza")
    view.move("Italy")
    assert view.window_id == 777
    assert "make new window" in chrome.scripts[-1] and "/wiki/Italy" in chrome.scripts[-1]
