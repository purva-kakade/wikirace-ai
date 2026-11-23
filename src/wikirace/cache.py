"""A tiny persistent key/value cache backed by SQLite.

Wikipedia articles barely change during a benchmark, so every API answer is
stored on disk. Re-running a benchmark, or running a second agent on the same
scenario, costs no extra network requests.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class DiskCache:
    """JSON values stored by string key. ``path=None`` keeps everything in memory."""

    def __init__(self, path: str | Path | None = None):
        if path is None:
            target = ":memory:"
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            target = str(path)
        self._conn = sqlite3.connect(target, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self._conn.commit()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, json.dumps(value))
            )
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM kv").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
