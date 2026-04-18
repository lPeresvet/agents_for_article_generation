"""LangGraph node callables."""

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from article_agents.prompts import (
    CORRESPONDENT_SYSTEM,
    FINALIZE_SYSTEM,
    MANAGER_SYSTEM,
    RESEARCHER_SYSTEM,
    REVIEWER_SYSTEM,
)
from article_agents.memory import ManagerMemory, format_memory_context
from article_agents.routing import compute_next_route, extract_manager_brief
from article_agents.state import MAX_MANAGER_STEPS, ArticleState
from article_agents.trace import log as _trace
from article_agents.tools import internet_search_DDGO

_MAX_TOOL_ROUNDS = 5
_AGENT_SEARCH_TOOLS = [internet_search_DDGO]


def _pick_llm(
    state: ArticleState,
    ollama: ChatOllama,
    gigachat: BaseChatModel | None,
) -> BaseChatModel:
    route = state.get("model_route", "ollama")
    if route == "gigachat" and gigachat is not None:
        _trace("llm route: GigaChat")
        return gigachat
    if route == "gigachat" and gigachat is None:
        _trace("llm route: GigaChat requested but GIGACHAT_CREDENTIALS missing; using Ollama")
    else:
        _trace("llm route: Ollama")
    return ollama


def make_memory_gate_node(memory: ManagerMemory):
    """If RAG already has a final article for this topic, set final_article and skip the pipeline."""

    def memory_gate(state: ArticleState) -> dict:
        topic = state.get("topic", "")
        _trace(f"Node start: memory_gate (topic={topic[:120]!r})")
        if os.environ.get("MEMORY_CACHE_RETURN", "1") in {"0", "false", "False"}:
            _trace("memory_gate: MEMORY_CACHE_RETURN disabled")
            return {}
        existing = memory.find_existing_article(topic)
        if existing:
            _trace(f"memory_gate: returning cached article for topic {topic[:80]!r}")
            return {"final_article": existing}
        _trace("memory_gate: no cached article for this topic")
        return {}

    return memory_gate


def _invoke(llm: BaseChatModel, system: str, user: str):
    """Qwen3.x via Ollama may use thinking mode; disable it and use non-streaming for stability."""
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    if isinstance(llm, ChatOllama):
        return llm.invoke(messages, think=False, stream=False)
    # GigaChat: avoid importing class (optional dep); match by runtime type name.
    if type(llm).__name__ == "GigaChat":
        messages = [
            SystemMessage(content=system + "\n\nПодготовь текст на русском языке."),
            HumanMessage(content=user),
        ]
    return llm.invoke(messages)


def _content(response) -> str:
    c = getattr(response, "content", response)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(c)


def _tool_call_name(tc) -> str:
    if isinstance(tc, dict):
        return str(tc.get("name", ""))
    return str(getattr(tc, "name", "") or "")


def _tool_call_id(tc) -> str:
    if isinstance(tc, dict):
        tid = tc.get("id")
    else:
        tid = getattr(tc, "id", None)
    return str(tid) if tid else "tool_call"


def _tool_call_args(tc) -> dict:
    if isinstance(tc, dict):
        args = tc.get("args")
    else:
        args = getattr(tc, "args", None)
    if isinstance(args, dict):
        return args
    return {}


def _invoke_with_tools(llm: BaseChatModel, system: str, user: str):
    """Multi-turn chat with DuckDuckGo search when the model emits tool calls."""
    llm_tools = llm.bind_tools(_AGENT_SEARCH_TOOLS)
    by_name = {t.name: t for t in _AGENT_SEARCH_TOOLS}
    messages: list = [SystemMessage(content=system), HumanMessage(content=user)]
    ollama_kwargs = {"think": False, "stream": False} if isinstance(llm, ChatOllama) else {}
    for _ in range(_MAX_TOOL_ROUNDS):
        response = llm_tools.invoke(messages, **ollama_kwargs)
        messages.append(response)
        tcalls = getattr(response, "tool_calls", None) or []
        if not tcalls:
            _trace("No tool calls; returning model output.")
            return _content(response)
        for tc in tcalls:
            name = _tool_call_name(tc)
            args = _tool_call_args(tc)
            _trace(f"Tool call: {name}({args})")
            tool = by_name.get(name)
            if tool is None:
                out = f"Unknown tool: {name}"
            else:
                try:
                    out = tool.invoke(_tool_call_args(tc))
                except Exception as exc:  # noqa: BLE001 — surface errors to the model
                    out = f"Tool error: {exc}"
            if not isinstance(out, str):
                out = str(out)
            _trace(f"Tool result ({name}): {out[:400].strip()}{'…' if len(out) > 400 else ''}")
            messages.append(ToolMessage(content=out, tool_call_id=_tool_call_id(tc)))
    return _content(messages[-1])


def make_manager_node(
    ollama: ChatOllama,
    gigachat: BaseChatModel | None,
    memory: ManagerMemory | None = None,
):
    mem = memory if memory is not None else ManagerMemory()

    def manager(state: ArticleState) -> dict:
        llm = _pick_llm(state, ollama, gigachat)
        step = state.get("step_count", 0) + 1
        _trace(f"Node start: manager (step {step}/{MAX_MANAGER_STEPS})")
        topic = state.get("topic", "")

        memory_block = ""
        if step == 1 and topic.strip():
            snippets = mem.retrieve(topic)
            ctx = format_memory_context(snippets)
            if ctx:
                memory_block = (
                    "Relevant memory from previous runs (read-only context, may be partially relevant):\n"
                    f"{ctx}\n\n"
                )

        user = (
            f"{memory_block}"
            f"Topic:\n{topic}\n\n"
            f"Research:\n{state.get('research', '') or '(empty)'}\n\n"
            f"Draft:\n{state.get('draft', '') or '(empty)'}\n\n"
            f"Reviewer notes:\n{state.get('review_feedback', '') or '(empty)'}\n\n"
            f"Manager step: {step} (max {MAX_MANAGER_STEPS} before forced finalize)."
        )
        msg = _invoke(llm, MANAGER_SYSTEM, user)
        text = _content(msg)
        nxt = compute_next_route(
            step_count=step,
            manager_output=text,
            research=state.get("research", ""),
            draft=state.get("draft", ""),
        )
        _trace(f"Node end: manager -> NEXT: {nxt}")
        return {
            "manager_output": text,
            "manager_brief": extract_manager_brief(text),
            "step_count": step,
            "next_route": nxt,
        }

    return manager


def make_researcher_node(ollama: ChatOllama, gigachat: BaseChatModel | None):
    def researcher(state: ArticleState) -> dict:
        llm = _pick_llm(state, ollama, gigachat)
        _trace("Node start: researcher")
        user = f"Topic:\n{state.get('topic', '')}\n"
        brief = state.get("manager_brief", "")
        if brief:
            user += f"\nEditor brief:\n{brief}\n"
        msg = _invoke_with_tools(llm, RESEARCHER_SYSTEM, user)
        _trace("Node end: researcher")
        return {"research": msg}

    return researcher


def make_correspondent_node(ollama: ChatOllama, gigachat: BaseChatModel | None):
    def correspondent(state: ArticleState) -> dict:
        llm = _pick_llm(state, ollama, gigachat)
        _trace("Node start: correspondent")
        user = (
            f"Topic:\n{state.get('topic', '')}\n\n"
            f"Research:\n{state.get('research', '')}\n\n"
            f"Prior draft (if any, revise):\n{state.get('draft', '') or '(none)'}\n\n"
            f"Reviewer feedback:\n{state.get('review_feedback', '') or '(none)'}\n"
        )
        brief = state.get("manager_brief", "")
        if brief:
            user += f"\nEditor brief:\n{brief}\n"
        msg = _invoke_with_tools(llm, CORRESPONDENT_SYSTEM, user)
        _trace("Node end: correspondent")
        return {"draft": msg}

    return correspondent


def make_reviewer_node(ollama: ChatOllama, gigachat: BaseChatModel | None):
    def reviewer(state: ArticleState) -> dict:
        llm = _pick_llm(state, ollama, gigachat)
        _trace("Node start: reviewer")
        user = (
            f"Topic:\n{state.get('topic', '')}\n\n"
            f"Research:\n{state.get('research', '')}\n\n"
            f"Draft:\n{state.get('draft', '')}\n"
        )
        brief = state.get("manager_brief", "")
        if brief:
            user += f"\nEditor brief:\n{brief}\n"
        msg = _invoke(llm, REVIEWER_SYSTEM, user)
        _trace("Node end: reviewer")
        return {"review_feedback": _content(msg)}

    return reviewer


def make_finalize_node(
    ollama: ChatOllama,
    gigachat: BaseChatModel | None,
    memory: ManagerMemory | None = None,
):
    mem = memory if memory is not None else ManagerMemory()

    def finalize(state: ArticleState) -> dict:
        llm = _pick_llm(state, ollama, gigachat)
        _trace("Node start: finalize")
        topic = state.get("topic", "")
        user = (
            f"Topic:\n{topic}\n\n"
            f"Research:\n{state.get('research', '')}\n\n"
            f"Latest draft:\n{state.get('draft', '')}\n\n"
            f"Reviewer notes:\n{state.get('review_feedback', '') or '(none)'}\n"
        )
        msg = _invoke(llm, FINALIZE_SYSTEM, user)
        final_text = _content(msg)
        try:
            mem.add(topic, final_text)
        except Exception as exc:  # noqa: BLE001
            _trace(f"finalize: memory write failed ({exc})")
        _trace("Node end: finalize")
        return {"final_article": final_text}

    return finalize
