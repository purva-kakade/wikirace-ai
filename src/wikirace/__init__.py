"""wikirace: AI agents that play the Wikipedia Game."""

from .game import GameResult, Step, play
from .wiki import Link, PageNotFound, WikiError, WikipediaClient

__version__ = "0.1.0"

__all__ = ["GameResult", "Link", "PageNotFound", "Step", "WikiError", "WikipediaClient", "play"]
