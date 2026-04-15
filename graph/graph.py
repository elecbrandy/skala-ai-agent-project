from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import AgentState
from graph.edges import route_after_search, route_after_review
from agents.intent import intent_node
from agents.rag import rag_node
from agents.web_search import web_search_node
from agents.draft import draft_node
from agents.supervisor import supervisor_review_node, supervisor_final_node
from agents.formatting import formatting_node


def increment_retry(state: AgentState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def build_graph():
    builder = StateGraph(AgentState)

    # ── 노드 등록 ──────────────────────────────────────────────
    builder.add_node("intent_node",       intent_node)
    builder.add_node("rag_node",          rag_node)
    builder.add_node("web_node",          web_search_node)
    builder.add_node("supervisor_review", supervisor_review_node)
    builder.add_node("increment_retry",   increment_retry)
    builder.add_node("draft_node",        draft_node)
    builder.add_node("supervisor_final",  supervisor_final_node)
    builder.add_node("formatting_node",   formatting_node)

    # ── 진입점 ─────────────────────────────────────────────────
    builder.set_entry_point("intent_node")

    # ── 엣지 ──────────────────────────────────────────────────
    # Intent → RAG + Web 병렬
    from langgraph.constants import Send

    def fan_out(state: AgentState):
        return [Send("rag_node", state), Send("web_node", state)]

    builder.add_conditional_edges("intent_node", fan_out)

    # RAG + Web → supervisor_review (join)
    builder.add_edge("rag_node", "supervisor_review")
    builder.add_edge("web_node", "supervisor_review")

    # supervisor_review → 조건부
    builder.add_conditional_edges(
        "supervisor_review",
        route_after_review,
        {
            "draft_node":      "increment_retry",
            "formatting_node": "formatting_node",
        },
    )
    builder.add_edge("increment_retry", "draft_node")

    # draft_node → supervisor_review (Reflection 루프)
    builder.add_edge("draft_node", "supervisor_review")

    # formatting_node → supervisor_final → END
    builder.add_edge("formatting_node", "supervisor_final")
    builder.add_edge("supervisor_final", END)

    return builder.compile(checkpointer=MemorySaver())


graph = build_graph()
