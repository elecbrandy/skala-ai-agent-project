from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    user_request: str
    parsed_intent: dict           # {keywords, companies, depth, date_range}
    rag_results: list[dict]       # [{content, source, date, score}]
    web_results: list[dict]       # [{url, date, bias_label, content, score}]
    trl_estimates: dict           # {company: {trl: int, confidence: str, evidence: list}}
    draft_report: str
    review_feedback: list[str]
    review_approved: bool
    rag_sufficient: bool
    web_sufficient: bool
    retry_count: int
    final_report: str
    messages: Annotated[list, add_messages]
