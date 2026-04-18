"""CLI entrypoint."""

import argparse
import os
import sys

from dotenv import load_dotenv

from article_agents.graph import build_graph
from article_agents.language import resolve_model_route
from article_agents.memory import ManagerMemory
from article_agents.observability import build_graph_run_config
from article_agents.state import MAX_MANAGER_STEPS
from article_agents.trace import enabled as trace_enabled, log as trace_log


def _log_model_route(model_route: str) -> None:
    """Always visible on stderr; use trace with timestamp when ARTICLE_AGENTS_TRACE is on."""
    msg = f"article_agents: model_route={model_route}"
    if trace_enabled():
        trace_log(msg)
    else:
        print(msg, file=sys.stderr, flush=True)


def _run_topic(graph, topic: str, verbose: bool) -> int:
    model_route = resolve_model_route(topic)
    _log_model_route(model_route)

    # Resolve cache before the graph so we return immediately even if the image
    # is missing an older graph change; avoids re-running the full pipeline.
    if os.environ.get("MEMORY_CACHE_RETURN", "1") not in {"0", "false", "False"}:
        cached = ManagerMemory().find_existing_article(topic)
        if cached:
            if verbose:
                print("\n=== memory_cache (hit) ===", file=sys.stderr, flush=True)
                s = cached if len(cached) <= 1200 else cached[:1200] + "…"
                print(f"- final_article:\n{s}\n", file=sys.stderr, flush=True)
            print(cached)
            return 0

    rec_limit = max(64, MAX_MANAGER_STEPS * 4 + 8)
    config = build_graph_run_config(
        recursion_limit=rec_limit,
        topic=topic,
        model_route=model_route,
        verbose=verbose,
    )
    initial = {"topic": topic, "model_route": model_route}
    if verbose:
        result: dict = dict(initial)
        for update in graph.stream(initial, config=config, stream_mode="updates"):
            # Updates look like {"node_name": {"field": "value", ...}}
            if not isinstance(update, dict):
                continue
            for node, delta in update.items():
                print(f"\n=== {node} ===", file=sys.stderr, flush=True)
                if isinstance(delta, dict):
                    for k, v in delta.items():
                        if v is None or v == "":
                            continue
                        s = str(v)
                        if len(s) > 1200:
                            s = s[:1200] + "…"
                        print(f"- {k}:\n{s}\n", file=sys.stderr, flush=True)
                    result.update(delta)
    else:
        result = graph.invoke(initial, config=config)

    final = result.get("final_article", "").strip()
    if not final:
        print("No final article produced.", file=sys.stderr)
        return 1
    print(final)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the LangGraph article pipeline (Ollama + manager-coordinated agents)."
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default="",
        help="Article topic (or pass via stdin if empty)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Stream agent activity to stderr while running",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Alias for --verbose (stream all internal steps)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive mode and accept multiple topics",
    )
    args = parser.parse_args(argv)

    load_dotenv()

    topic = args.topic.strip()
    if not topic and not args.interactive and not sys.stdin.isatty():
        topic = sys.stdin.read().strip()
    if not topic and not args.interactive:
        parser.error("topic is required (argument or stdin)")

    if args.verbose or args.trace:
        os.environ["ARTICLE_AGENTS_TRACE"] = "1"

    graph = build_graph()
    verbose = args.verbose or args.trace

    if args.interactive:
        print("Interactive mode: enter a topic (empty line, 'exit', or Ctrl-D to quit).", file=sys.stderr)
        while True:
            print("topic> ", end="", file=sys.stderr, flush=True)
            raw = sys.stdin.readline()
            if raw == "":
                print("", file=sys.stderr)
                return 0
            current = raw.strip()
            if not current or current.lower() in {"exit", "quit"}:
                return 0
            rc = _run_topic(graph, current, verbose)
            if rc != 0:
                print("Run failed. Try another topic or type 'exit'.", file=sys.stderr)
            print("", file=sys.stderr)

    return _run_topic(graph, topic, verbose)
