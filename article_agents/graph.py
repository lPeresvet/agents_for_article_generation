"""Compiled LangGraph for the article pipeline."""

from langgraph.graph import END, START, StateGraph

from article_agents.llm import make_gigachat_llm, make_ollama_llm
from article_agents.memory import ManagerMemory
from article_agents.nodes import (
    make_correspondent_node,
    make_finalize_node,
    make_manager_node,
    make_memory_gate_node,
    make_researcher_node,
    make_reviewer_node,
)
from article_agents.routing import route_from_state
from article_agents.state import ArticleState, NextRoute


def build_graph():
    ollama = make_ollama_llm()
    gigachat = make_gigachat_llm()
    memory = ManagerMemory()
    graph = StateGraph(ArticleState)
    graph.add_node("memory_gate", make_memory_gate_node(memory))
    graph.add_node("manager", make_manager_node(ollama, gigachat, memory=memory))
    graph.add_node("researcher", make_researcher_node(ollama, gigachat))
    graph.add_node("correspondent", make_correspondent_node(ollama, gigachat))
    graph.add_node("reviewer", make_reviewer_node(ollama, gigachat))
    graph.add_node("finalize", make_finalize_node(ollama, gigachat, memory=memory))

    graph.add_edge(START, "memory_gate")

    def route_memory_gate(state: ArticleState) -> str:
        if state.get("final_article", "").strip():
            return "end"
        return "manager"

    graph.add_conditional_edges(
        "memory_gate",
        route_memory_gate,
        {"end": END, "manager": "manager"},
    )

    def route_manager(state: ArticleState) -> NextRoute:
        return route_from_state(state)

    graph.add_conditional_edges(
        "manager",
        route_manager,
        {
            "researcher": "researcher",
            "correspondent": "correspondent",
            "reviewer": "reviewer",
            "finalize": "finalize",
        },
    )
    graph.add_edge("researcher", "manager")
    graph.add_edge("correspondent", "manager")
    graph.add_edge("reviewer", "manager")
    graph.add_edge("finalize", END)

    return graph.compile()
