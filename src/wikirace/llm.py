"""Minimal client for a local LLM served by Ollama (https://ollama.com)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

import requests

DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_HOST = "http://localhost:11434"


class LLMError(RuntimeError):
    """The LLM could not be reached or failed."""


class LLMReplyError(LLMError):
    """The LLM answered, but not with the JSON we asked for."""


class LLM(Protocol):
    model: str

    def complete_json(self, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]: ...


def _normalise_host(host: str) -> str:
    host = host.strip().rstrip("/")
    return host if host.startswith(("http://", "https://")) else f"http://{host}"


class OllamaLLM:
    """Chat completion against Ollama with JSON-schema constrained output."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str | None = None,
        temperature: float = 0.0,
        seed: int = 42,
        timeout: float = 180.0,
        session: requests.Session | None = None,
    ):
        self.model = model
        self.host = _normalise_host(host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST)
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self.session = session or requests.Session()

    def ensure_ready(self) -> None:
        """Fail fast with a helpful message if Ollama or the model is missing."""
        try:
            resp = self.session.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"Could not reach Ollama at {self.host}. Start it with `ollama serve`.") from exc
        names = {m.get("name") for m in resp.json().get("models", [])} | {
            m.get("model") for m in resp.json().get("models", [])
        }
        wanted = {self.model, f"{self.model}:latest"}
        if not names & wanted:
            raise LLMError(f"Ollama has no model {self.model!r}. Pull it with `ollama pull {self.model}`.")

    def complete_json(self, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "format": schema,
            "stream": False,
            "options": {"temperature": self.temperature, "seed": self.seed},
        }
        try:
            resp = self.session.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise LLMError(f"Could not reach Ollama at {self.host}. Start it with `ollama serve`.") from exc
        except requests.RequestException as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        if resp.status_code == 404:
            raise LLMError(f"Ollama has no model {self.model!r}. Pull it with `ollama pull {self.model}`.")
        if not resp.ok:
            raise LLMError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}")
        content = resp.json().get("message", {}).get("content", "")
        return parse_json_object(content)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating prose around it."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise LLMReplyError(f"LLM reply was not JSON: {text[:200]!r}") from None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMReplyError(f"LLM reply was not JSON: {text[:200]!r}") from exc
    if not isinstance(value, dict):
        raise LLMReplyError(f"Expected a JSON object, got {type(value).__name__}")
    return value
