"""Heuristic detection of Russian-language topics and backend routing.

Rules (for maintainers):
- ``MODEL_ROUTE_OVERRIDE`` in the environment, if set to ``gigachat`` or ``ollama``,
  wins over language detection.
- Otherwise ``topic_is_russian`` compares Cyrillic vs Latin letter counts;
  Russian topics use GigaChat, everything else Ollama (e.g. qwen3.5).
"""

from __future__ import annotations

import os
import re

from article_agents.state import ModelRoute

_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


def topic_is_russian(text: str) -> bool:
    """True when the topic is predominantly Russian (Cyrillic).

    Uses letter counts: Cyrillic vs Latin. If both appear, Russian wins when
    Cyrillic count is greater than or equal to Latin count.
    """
    if not text or not text.strip():
        return False
    cyrillic = len(_CYRILLIC_RE.findall(text))
    if cyrillic == 0:
        return False
    latin = len(re.findall(r"[A-Za-z]", text))
    return cyrillic >= latin


def resolve_model_route(topic: str) -> ModelRoute:
    """Choose GigaChat vs Ollama from env override or topic language.

    Emits no logs here: ``cli`` prints ``model_route`` to stderr; use ``-v`` and
    ``ARTICLE_AGENTS_TRACE=1`` for per-node LLM routing logs in ``nodes._pick_llm``.
    """
    override = os.environ.get("MODEL_ROUTE_OVERRIDE", "").strip().lower()
    if override in ("gigachat", "ollama"):
        return override  # type: ignore[return-value]
    if topic_is_russian(topic):
        return "gigachat"
    return "ollama"
