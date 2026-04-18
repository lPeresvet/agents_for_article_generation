"""Render the article pipeline LangGraph (Jupyter PNG or text fallbacks).

Notebook::

    from article_agents.visualize_graph import display_article_graph
    display_article_graph()

CLI::

    python -m article_agents.visualize_graph -o article_graph.png
    python -m article_agents.visualize_graph --mermaid graph.mmd

PNG is built **locally first** via the Graphviz ``dot`` binary (install ``brew install graphviz``
on macOS, or ``apt install graphviz`` in Linux/Docker). If ``dot`` is missing, the tool falls
back to the mermaid.ink API (needs network), then optional pygraphviz. If PNG still fails but
``-o`` was given, a sibling ``.mmd`` file is written for `mermaid.live <https://mermaid.live>`_.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _env_for_fast_build() -> None:
    """Avoid long Chroma retries when we only need graph structure for drawing."""
    os.environ.setdefault("MANAGER_MEMORY_ENABLED", "0")


def _fallback_print(drawable) -> None:
    """ASCII art if ``grandalf`` is installed; else Mermaid source."""
    try:
        print(drawable.draw_ascii())
    except ImportError:
        print(drawable.draw_mermaid(), file=sys.stderr)


def _agent_tool_names() -> dict[str, tuple[str, ...]]:
    """Maps LangGraph node id → tool names used inside that node (see ``nodes._invoke_with_tools``)."""
    from article_agents.tools import internet_search_DDGO

    name = internet_search_DDGO.name
    return {"researcher": (name,), "correspondent": (name,)}


def _render_png_via_dot(drawable) -> bytes | None:
    """Offline PNG via the ``graphviz`` PyPI package + ``dot`` on PATH (``brew install graphviz``)."""
    try:
        import graphviz as gv  # pip package; system binary is ``dot``
    except ImportError:
        return None
    try:
        dot = gv.Digraph(format="png")
        dot.attr(fontsize="10")
        for node in drawable.nodes.values():
            dot.node(node.id, label=node.name or node.id)
        for edge in drawable.edges:
            kwargs: dict = {}
            if edge.data is not None:
                kwargs["label"] = str(edge.data)
            if edge.conditional:
                kwargs["style"] = "dashed"
            dot.edge(edge.source, edge.target, **kwargs)

        agent_tools = _agent_tool_names()
        tools_used: set[str] = set()
        for tnames in agent_tools.values():
            tools_used.update(tnames)
        if tools_used:
            with dot.subgraph(name="cluster_tools") as tg:
                tg.attr(label="LangChain tools", style="rounded", color="gray", fontcolor="gray")
                for tname in sorted(tools_used):
                    tid = f"tool_{tname}"
                    tg.node(
                        tid,
                        label=tname,
                        shape="component",
                        style="filled",
                        fillcolor="#e8f5e9",
                        fontname="Helvetica",
                    )
            for agent_id, tnames in agent_tools.items():
                if agent_id not in drawable.nodes:
                    continue
                for tname in tnames:
                    dot.edge(
                        agent_id,
                        f"tool_{tname}",
                        style="dotted",
                        color="#2e7d32",
                        arrowhead="open",
                        label="bind_tools",
                        fontsize="8",
                        fontcolor="#2e7d32",
                    )

        return dot.pipe()
    except Exception:
        return None


def _mermaid_with_tools(drawable) -> str:
    """Mermaid source: pipeline graph plus a second chart for tool bindings."""
    base = drawable.draw_mermaid().rstrip()
    agent_tools = _agent_tool_names()
    tools_used = sorted({t for names in agent_tools.values() for t in names})
    if not tools_used:
        return base

    def _tid(n: str) -> str:
        return "v_" + "".join(c if c.isalnum() or c == "_" else "_" for c in n)

    aid = {"researcher": "r_x", "correspondent": "c_x"}
    extra_lines = [
        "---",
        "%% bind_tools inside nodes (not extra LangGraph steps)",
        "flowchart LR",
    ]
    for tname in tools_used:
        tid = _tid(tname)
        extra_lines.append(f"    {tid}[[{tname}]]")
    for agent_id, tnames in agent_tools.items():
        for tname in tnames:
            extra_lines.append(f"    {aid[agent_id]}[{agent_id}] -.-> {_tid(tname)}")

    return base + "\n" + "\n".join(extra_lines) + "\n"


def _render_png_bytes(drawable) -> bytes:
    """PNG: local ``dot`` first (fast/offline), then mermaid.ink, then LangChain ``draw_png`` (pygraphviz)."""
    png = _render_png_via_dot(drawable)
    if png:
        return png
    try:
        return drawable.draw_mermaid_png(max_retries=2, retry_delay=1.0)
    except Exception as mermaid_exc:
        try:
            raw = drawable.draw_png()
        except ImportError:
            raise mermaid_exc from None
        except Exception as gv_exc:
            raise RuntimeError(
                f"No local Graphviz dot; Mermaid API failed ({mermaid_exc!s}); "
                f"pygraphviz draw_png failed ({gv_exc!s})"
            ) from gv_exc
        if raw:
            return raw
        raise mermaid_exc from None


def display_article_graph() -> None:
    """Jupyter / IPython: PNG via Mermaid API, then Graphviz, else ASCII / Mermaid text."""
    from IPython.display import Image, display

    _env_for_fast_build()
    from article_agents.graph import build_graph

    drawable = build_graph().get_graph()
    try:
        display(Image(_render_png_bytes(drawable)))
    except Exception as e:
        print(f"Could not render graph as PNG: {e}")
        _fallback_print(drawable)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Export the article pipeline graph as PNG (Mermaid API or Graphviz).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="FILE.png",
        help="Write PNG (tries mermaid.ink, then local graphviz+pygraphviz)",
    )
    parser.add_argument(
        "-m",
        "--mermaid",
        type=Path,
        metavar="FILE.mmd",
        help="Write Mermaid source (https://mermaid.live)",
    )
    args = parser.parse_args(argv)

    _env_for_fast_build()
    from article_agents.graph import build_graph

    drawable = build_graph().get_graph()

    if args.mermaid:
        args.mermaid.write_text(_mermaid_with_tools(drawable), encoding="utf-8")
        print(f"Wrote Mermaid source to {args.mermaid.resolve()}", file=sys.stderr)

    if args.mermaid and not args.output:
        return 0

    try:
        png = _render_png_bytes(drawable)
    except Exception as e:
        print(f"Could not render PNG (mermaid.ink + graphviz): {e}", file=sys.stderr)
        if args.output:
            mmd_path = args.output.with_suffix(".mmd")
            mmd_path.write_text(_mermaid_with_tools(drawable), encoding="utf-8")
            print(
                f"Wrote Mermaid source to {mmd_path.resolve()} — paste into "
                "https://mermaid.live or export PNG from there.",
                file=sys.stderr,
            )
        elif not args.mermaid:
            _fallback_print(drawable)
        return 0 if (args.output or args.mermaid) else 1

    if args.output:
        args.output.write_bytes(png)
        print(f"Wrote PNG to {args.output.resolve()}", file=sys.stderr)
    elif not args.mermaid:
        print("PNG OK. Use -o FILE.png to save; -m FILE.mmd saves Mermaid source.", file=sys.stderr)
        _fallback_print(drawable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
