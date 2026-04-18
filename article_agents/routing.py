"""Parse manager decisions and apply safety routing."""

from article_agents.state import MAX_MANAGER_STEPS, ArticleState, NextRoute


def parse_next_route(text: str) -> NextRoute | None:
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip()
        if not stripped.upper().startswith("NEXT:"):
            continue
        token = stripped.split(":", 1)[1].strip().lower()
        if token in ("researcher", "correspondent", "reviewer", "finalize"):
            return token  # type: ignore[return-value]
    return None


def extract_manager_brief(text: str) -> str:
    key = "Brief for next step:"
    if key not in text:
        return ""
    tail = text.split(key, 1)[1]
    lines: list[str] = []
    for line in tail.splitlines():
        if line.strip().upper().startswith("NEXT:"):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def compute_next_route(
    *,
    step_count: int,
    manager_output: str,
    research: str,
    draft: str,
) -> NextRoute:
    if step_count >= MAX_MANAGER_STEPS:
        return "finalize"

    parsed = parse_next_route(manager_output)
    research_ok = bool(research.strip())
    draft_ok = bool(draft.strip())

    # Fallbacks when the manager forgot NEXT:
    if parsed is None:
        if draft_ok and research_ok:
            return "finalize"
        if research_ok:
            return "correspondent"
        return "researcher"

    # Never finalize before we have both research and a draft.
    if parsed == "finalize" and not (research_ok and draft_ok):
        if not research_ok:
            return "researcher"
        return "correspondent"

    # If manager asks for correspondent but we have no research yet, fall back to researcher.
    if parsed == "correspondent" and not research_ok:
        return "researcher"

    # If manager asks for reviewer but we have no draft yet, route through correspondent / researcher first.
    if parsed == "reviewer" and not draft_ok:
        if research_ok:
            return "correspondent"
        return "researcher"

    # Prevent endless manager↔researcher loops:
    # once we have some research, force progress through correspondent / reviewer.
    if parsed == "researcher" and research_ok:
        # If there is no draft yet, go to correspondent to create one.
        if not draft_ok:
            return "correspondent"
        # If both research and draft exist, prefer reviewer over more research.
        return "reviewer"

    return parsed


def route_from_state(state: ArticleState) -> NextRoute:
    return state.get("next_route", "researcher")
