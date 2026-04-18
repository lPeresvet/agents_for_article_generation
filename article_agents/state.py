"""Shared graph state and limits."""

from typing import Literal, TypedDict

MAX_MANAGER_STEPS = 6

NextRoute = Literal["researcher", "correspondent", "reviewer", "finalize"]

ModelRoute = Literal["gigachat", "ollama"]


class ArticleState(TypedDict, total=False):
    topic: str
    model_route: ModelRoute
    research: str
    draft: str
    review_feedback: str
    final_article: str
    manager_brief: str
    manager_output: str
    next_route: NextRoute
    step_count: int
