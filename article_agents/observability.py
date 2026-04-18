"""LangSmith tracing: env-based enablement + per-run metadata/tags for LangGraph.

Tracing sends runs to LangSmith when:

- ``LANGCHAIN_TRACING_V2=true``
- ``LANGCHAIN_API_KEY`` is set (from https://smith.langchain.com )

Optional: ``LANGCHAIN_PROJECT``, ``LANGCHAIN_ENDPOINT`` (self-hosted).

See: https://docs.smith.langchain.com/
"""

from __future__ import annotations

import os
from typing import Any


def _int_env(key: str, default: int) -> int:
    """Parse int from env; empty or invalid uses ``default`` (Docker often passes ``\"\"``)."""
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def build_graph_run_config(
    *,
    recursion_limit: int,
    topic: str,
    model_route: str,
    verbose: bool,
) -> dict[str, Any]:
    """RunnableConfig fields for ``graph.invoke`` / ``graph.stream`` (LangSmith filters)."""
    cfg: dict[str, Any] = {"recursion_limit": recursion_limit}
    meta: dict[str, Any] = {
        "model_route": model_route,
        "verbose": verbose,
        "topic_len": len(topic),
    }
    preview_len = _int_env("LANGSMITH_TOPIC_PREVIEW_CHARS", 200)
    if preview_len > 0 and topic.strip():
        meta["topic_preview"] = topic.strip()[:preview_len]

    tags = ["article-agents", f"route:{model_route}"]
    extra = os.environ.get("LANGCHAIN_TAGS", "").strip()
    if extra:
        tags.extend(t.strip() for t in extra.split(",") if t.strip())

    cfg["metadata"] = meta
    cfg["tags"] = tags

    prefix = os.environ.get("LANGSMITH_RUN_NAME_PREFIX", "article").strip() or "article"
    short = topic.replace("\n", " ").strip()[:48] or "run"
    cfg["run_name"] = f"{prefix}:{short}"

    return cfg
