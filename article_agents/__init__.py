"""Multi-agent article pipeline orchestrated by a main-editor manager (LangGraph + Ollama)."""

from article_agents.graph import build_graph

__all__ = ["build_graph"]
