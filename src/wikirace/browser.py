"""Watch a game live in Google Chrome.

The start article opens in a new Chrome window. Every click the agent makes
opens the next article in a new tab of that same window, so the tabs read left
to right as the path the agent took.

On macOS the window is driven with AppleScript, which lets us add tabs to that
exact window even if you click into another one. The first time, macOS asks
whether your terminal may control Google Chrome. If you say no (or on other
systems) each page simply opens in a new Chrome tab instead.
"""

from __future__ import annotations

import subprocess
import sys
import time
import webbrowser
from typing import Callable, Sequence
from urllib.parse import quote

Runner = Callable[[Sequence[str]], str]


class BrowserError(RuntimeError):
    """Chrome could not be controlled."""


def article_url(title: str, lang: str = "en") -> str:
    """The Wikipedia web address of an article ("Ancient Egypt" -> .../wiki/Ancient_Egypt)."""
    return f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'), safe='()_,-.!~*:/')}"


def _osascript(lines: Sequence[str]) -> str:
    """Run AppleScript lines with macOS's `osascript` and return what the script returns."""
    args = ["osascript"]
    for line in lines:
        args += ["-e", line]
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BrowserError(str(exc)) from exc
    if result.returncode != 0:
        raise BrowserError(result.stderr.strip() or "osascript failed")
    return result.stdout.strip()


def _open_in_chrome(url: str) -> None:
    """Fallback: open a URL in Chrome (or the default browser) without AppleScript."""
    if sys.platform == "darwin":
        subprocess.run(["open", "-a", "Google Chrome", url], check=False)
        return
    try:
        webbrowser.get("chrome").open_new_tab(url)
    except webbrowser.Error:
        webbrowser.open_new_tab(url)


class ChromeView:
    """Shows each move of a game as a new tab in one Chrome window."""

    def __init__(
        self,
        lang: str = "en",
        delay: float = 1.5,
        runner: Runner | None = None,
        opener: Callable[[str], None] | None = None,
    ):
        self.lang = lang
        self.delay = delay  # seconds to pause after each tab so you can follow along
        self._run = runner or _osascript
        self._open = opener or _open_in_chrome
        self.scripted = runner is not None or sys.platform == "darwin"
        self.window_id: int | None = None
        self.tabs_opened = 0

    def start(self, title: str) -> None:
        """Open the start article in a brand-new Chrome window."""
        url = article_url(title, self.lang)
        if self.scripted:
            try:
                reply = self._run(
                    [
                        'tell application "Google Chrome"',
                        "activate",
                        "set w to make new window",
                        f'set URL of active tab of w to "{url}"',
                        "return id of w",
                        "end tell",
                    ]
                )
                self.window_id = int(reply)
                self.tabs_opened = 1
                return
            except (BrowserError, ValueError):
                self.scripted = False  # no permission or no Chrome: fall back for the rest of the game
        self._open(url)
        self.tabs_opened = 1

    def move(self, title: str) -> None:
        """Open the article the agent just clicked in a new tab of the same window."""
        url = article_url(title, self.lang)
        if self.scripted and self.window_id is not None:
            try:
                self._run(
                    [
                        'tell application "Google Chrome"',
                        f"tell window id {self.window_id}",
                        f'make new tab with properties {{URL:"{url}"}}',
                        "end tell",
                        "end tell",
                    ]
                )
            except BrowserError:
                # The window was closed mid-game: start a fresh one from here.
                self.start(title)
        else:
            self._open(url)
        self.tabs_opened += 1
        if self.delay > 0:
            time.sleep(self.delay)
