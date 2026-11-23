"""wikirace: AI agents that play the Wikipedia Game."""

from .wiki import Link, PageNotFound, WikiError, WikipediaClient

__version__ = "0.1.0"

__all__ = ["Link", "PageNotFound", "WikiError", "WikipediaClient"]
