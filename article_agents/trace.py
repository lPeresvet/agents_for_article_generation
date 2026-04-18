"""Minimal stderr tracing for agent activity.

Enabled via env var ARTICLE_AGENTS_TRACE=1 (set by CLI --verbose/--trace, or in
``.env``). Model *selection* is printed unconditionally from ``cli``; trace adds
timestamps for node-level steps (including ``llm route`` inside nodes).
"""

from __future__ import annotations

import os
import sys
import time


def enabled() -> bool:
    return os.environ.get("ARTICLE_AGENTS_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}


def log(msg: str) -> None:
    if not enabled():
        return
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)

