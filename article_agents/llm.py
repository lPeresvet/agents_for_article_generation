"""Chat model factories: Ollama (default) and GigaChat (Russian topics)."""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama


def make_ollama_llm() -> ChatOllama:
    timeout = float(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "600"))
    num_predict = os.environ.get("OLLAMA_NUM_PREDICT")
    kwargs: dict = {
        "model": os.environ.get("OLLAMA_MODEL", "qwen3.5:2b"),
        "base_url": os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", "0.3")),
        "client_kwargs": {"timeout": timeout},
        "reasoning": False,
    }
    if num_predict is not None and num_predict.strip():
        try:
            kwargs["num_predict"] = int(num_predict)
        except ValueError:
            pass
    return ChatOllama(**kwargs)


def make_gigachat_llm() -> BaseChatModel | None:
    """Return a GigaChat client if credentials are set; otherwise None."""
    creds = (os.environ.get("GIGACHAT_CREDENTIALS") or "").strip()
    if not creds:
        return None
    try:
        from langchain_gigachat import GigaChat
    except ImportError:
        return None

    scope = (os.environ.get("GIGACHAT_SCOPE") or "GIGACHAT_API_PERS").strip()
    model = (os.environ.get("GIGACHAT_MODEL") or "GigaChat").strip()
    verify_raw = os.environ.get("GIGACHAT_VERIFY_SSL_CERTS", "")
    kwargs: dict = {
        "credentials": creds,
        "scope": scope,
        "model": model,
    }
    if verify_raw.strip().lower() in {"0", "false", "no"}:
        kwargs["verify_ssl_certs"] = False
    elif verify_raw.strip().lower() in {"1", "true", "yes"}:
        kwargs["verify_ssl_certs"] = True

    temperature = os.environ.get("GIGACHAT_TEMPERATURE")
    if temperature is not None and str(temperature).strip():
        try:
            kwargs["temperature"] = float(temperature)
        except ValueError:
            pass

    return GigaChat(**kwargs)


def make_llm() -> ChatOllama:
    """Backward-compatible alias for the default (Ollama) chat model."""
    return make_ollama_llm()
