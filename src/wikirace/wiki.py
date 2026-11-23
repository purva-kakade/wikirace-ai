"""Wikipedia access layer.

Everything the game, the agents and the shortest-path oracle need from
Wikipedia goes through :class:`WikipediaClient`, which talks to the public
MediaWiki Action API and caches every answer on disk.

Link lists come from ``generator=links`` with ``redirects=1``, so every link is
reported under its canonical title ("USA" -> "United States") and red links
(articles that don't exist) are dropped. The same request also returns each
linked article's short description, which the agents use to tell homonyms
apart ("Mercury (planet)" vs. "Freddie Mercury").
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Protocol, Sequence

import requests

from .cache import DiskCache

_BATCH = 50  # the API accepts at most 50 titles per request


def default_user_agent() -> str:
    """User-Agent per the Wikimedia policy: client/version (contact) and the word "bot".

    Wikimedia allows anonymous clients 10 requests/minute, but 200/minute when the
    User-Agent carries contact details (an email or URL), so set ``WIKIRACE_CONTACT``.
    ``WIKIRACE_USER_AGENT`` replaces the whole header.
    """
    contact = os.environ.get("WIKIRACE_CONTACT", "").strip()
    return f"wikirace-ai/0.1 ({contact}; educational bot)" if contact else "wikirace-ai/0.1 (educational bot)"


class WikiError(RuntimeError):
    """Wikipedia could not answer a request."""


class PageNotFound(WikiError):
    """The requested article does not exist."""


@dataclass(frozen=True)
class Link:
    """A link to an article, as it appears on a page."""

    title: str
    description: str = ""

    def text(self, with_description: bool = True) -> str:
        """The string that gets embedded or shown to an LLM."""
        if with_description and self.description:
            return f"{self.title}: {self.description}"
        return self.title


class WikiSource(Protocol):
    """The part of Wikipedia the game and the agents use. Tests swap in a fake."""

    def resolve(self, title: str) -> str: ...

    def describe(self, title: str) -> str: ...

    def links(self, title: str) -> list[Link]: ...


class WikipediaClient:
    """Cached, polite client for the MediaWiki Action API."""

    def __init__(
        self,
        lang: str = "en",
        cache: DiskCache | None = None,
        session: requests.Session | None = None,
        user_agent: str | None = None,
        min_interval: float = 0.35,  # ~170 requests/minute, under Wikimedia's 200/minute
        timeout: float = 30.0,
        max_retries: int = 5,
    ):
        self.lang = lang
        self.api_url = f"https://{lang}.wikipedia.org/w/api.php"
        self.cache = cache if cache is not None else DiskCache(None)
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = (
            user_agent or os.environ.get("WIKIRACE_USER_AGENT") or default_user_agent()
        )
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self.network_requests = 0
        self.retries = 0  # requests repeated after rate limiting or server errors
        self._last_request = 0.0

    # ------------------------------------------------------------------ HTTP

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        params = {"action": "query", "format": "json", "formatversion": "2", "maxlag": "5", **params}
        delay = 5.0  # Wikimedia asks for at least 5 s before retrying without a Retry-After
        last_error: Exception | None = None
        for _ in range(self.max_retries):
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()  # spacing is measured between request starts
            try:
                resp = self.session.get(self.api_url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
            else:
                self.network_requests += 1
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_error = WikiError(f"HTTP {resp.status_code}")
                    delay = max(delay, float(resp.headers.get("Retry-After", 0) or 0))
                elif not resp.ok:
                    raise WikiError(f"HTTP {resp.status_code} from Wikipedia")
                else:
                    try:
                        data = resp.json()
                    except ValueError:  # e.g. an HTML error page served with status 200
                        data = {"error": {"code": "badjson"}}
                    error = data.get("error")
                    if not error:
                        return data
                    if error.get("code") not in ("maxlag", "badjson"):
                        raise WikiError(f"{error.get('code')}: {error.get('info')}")
                    last_error = WikiError(f"Wikipedia sent an unusable reply ({error.get('code')})")
            self.retries += 1
            time.sleep(delay)
            delay = min(delay * 2, 120.0)
        raise WikiError(f"Wikipedia request failed after {self.max_retries} attempts: {last_error}")

    def _query_all(self, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield the ``query`` block of every page of results, following continuation."""
        cont: dict[str, Any] = {}
        while True:
            data = self._get({**params, **cont})
            yield data.get("query", {})
            if "continue" not in data:
                return
            cont = data["continue"]

    def _cached(self, key: str, compute: Callable[[], Any]) -> Any:
        key = f"{self.lang}:{key}"
        value = self.cache.get(key)
        if value is None:
            value = compute()
            self.cache.set(key, value)
        return value

    # ------------------------------------------------------------ game queries

    def resolve(self, title: str) -> str:
        """Canonical title of an article, following redirects."""

        def compute() -> str:
            data = self._get({"titles": title, "redirects": 1})
            pages = data.get("query", {}).get("pages", [])
            if not pages or pages[0].get("missing") or pages[0].get("invalid"):
                return ""
            return pages[0]["title"]

        canonical = self._cached(f"resolve:{title}", compute)
        if not canonical:
            raise PageNotFound(f"No Wikipedia article called {title!r}")
        return canonical

    def search(self, query: str, limit: int = 5) -> list[str]:
        """Article titles matching a free-text query, best match first (for typo suggestions)."""

        def run(text: str) -> dict[str, Any]:
            params = {"list": "search", "srsearch": text, "srnamespace": 0, "srlimit": limit, "srprop": ""}
            return self._get({**params, "srinfo": "suggestion"}).get("query", {})

        def compute() -> list[str]:
            result = run(query)
            titles = [hit["title"] for hit in result.get("search", [])]
            # Full-text search doesn't fix typos ("mount everst"), but it suggests a spelling.
            suggestion = result.get("searchinfo", {}).get("suggestion")
            if suggestion and suggestion.lower() != query.lower():
                corrected = [hit["title"] for hit in run(suggestion).get("search", [])]
                titles = corrected + [t for t in titles if t not in corrected]
            return titles[:limit]

        return self._cached(f"search:{limit}:{query}", compute)

    def describe(self, title: str) -> str:
        """The article's one-line short description (may be empty)."""

        def compute() -> str:
            data = self._get({"titles": title, "redirects": 1, "prop": "description"})
            pages = data.get("query", {}).get("pages", [])
            return (pages[0].get("description") or "") if pages else ""

        return self._cached(f"describe:{title}", compute)

    def links(self, title: str) -> list[Link]:
        """Every existing article linked from ``title``, sorted by title."""

        def compute() -> list[list[str]]:
            found: dict[str, str] = {}
            params = {
                "generator": "links",
                "titles": title,
                "gplnamespace": 0,
                "gpllimit": "max",
                "prop": "description",
                "redirects": 1,
            }
            for query in self._query_all(params):
                for page in query.get("pages", []):
                    if page.get("missing") or page.get("invalid") or page.get("ns", 0) != 0:
                        continue
                    name = page["title"]
                    found[name] = page.get("description") or found.get(name, "")
            found.pop(title, None)  # ignore self-links
            return [[name, found[name]] for name in sorted(found)]

        return [Link(name, desc) for name, desc in self._cached(f"links:{title}", compute)]

    # ----------------------------------------------------------- oracle queries

    def backlinks(self, title: str, limit: int = 25_000) -> tuple[list[str], bool]:
        """Articles that link to ``title`` (directly or through a redirect).

        Returns ``(titles, complete)``; ``complete`` is False when ``limit`` cut the list short.
        """

        def compute() -> list[Any]:
            found: set[str] = set()
            complete = True
            params = {
                "list": "backlinks",
                "bltitle": title,
                "blnamespace": 0,
                "bllimit": "max",
                "blredirect": 1,
                "blfilterredir": "nonredirects",
            }
            for query in self._query_all(params):
                for item in query.get("backlinks", []):
                    if item.get("redirect"):
                        found.update(r["title"] for r in item.get("redirlinks", []))
                    else:
                        found.add(item["title"])
                if len(found) >= limit:
                    complete = False
                    break
            return [sorted(found), complete]

        titles, complete = self._cached(f"backlinks:{title}:{limit}", compute)
        return titles, complete

    def redirect_titles(self, title: str, limit: int = 49) -> list[str]:
        """Titles that redirect to ``title`` (e.g. "USA" for "United States")."""

        def compute() -> list[str]:
            found: list[str] = []
            params = {"prop": "redirects", "titles": title, "rdnamespace": 0, "rdlimit": "max"}
            for query in self._query_all(params):
                for page in query.get("pages", []):
                    found.extend(r["title"] for r in page.get("redirects", []))
                if len(found) >= limit:
                    break
            return found[:limit]

        return self._cached(f"redirects:{title}:{limit}", compute)

    def pages_linking_to(self, sources: Sequence[str], targets: Sequence[str]) -> set[str]:
        """The subset of ``sources`` whose page contains a link to any of ``targets``.

        Uses ``pltitles`` so 50 pages are checked per request without downloading
        their full link lists.
        """
        targets = list(targets)[:_BATCH]
        hits: set[str] = set()
        for i in range(0, len(sources), _BATCH):
            batch = sorted(sources[i : i + _BATCH])
            digest = hashlib.sha1("|".join(batch + ["#"] + targets).encode()).hexdigest()

            def compute(batch: list[str] = batch) -> list[str]:
                linked: set[str] = set()
                params = {
                    "prop": "links",
                    "titles": "|".join(batch),
                    "pltitles": "|".join(targets),
                    "pllimit": "max",
                }
                for query in self._query_all(params):
                    for page in query.get("pages", []):
                        if page.get("links"):
                            linked.add(page["title"])
                return sorted(linked)

            hits.update(self._cached(f"linking:{digest}", compute))
        return hits
