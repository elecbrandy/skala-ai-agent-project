from datetime import datetime
from graph.state import AgentState


def formatting_node(state: AgentState) -> dict:
    """
    Supervisor 승인된 초안을 최종 포맷으로 정리.
    실제 환경에서는 PDF/DOCX 변환 라이브러리 연동 가능.
    """
    draft    = state.get("draft_report", "")
    approved = state.get("review_approved", False)
    intent   = state.get("parsed_intent", {})
    now      = datetime.now().strftime("%Y-%m-%d %H:%M")

    companies = intent.get("companies", [])
    keywords  = intent.get("keywords", [])

    subject_line = ""
    if companies or keywords:
        parts = []
        if companies:
            parts.append(" · ".join(companies))
        if keywords:
            parts.append(" / ".join(keywords))
        subject_line = f"> 분석 대상: {' — '.join(parts)}\n"

    status_icon = "✅ Supervisor 승인" if approved else "⚠️ 최대 재시도 초과 — 부분 보고서"

    header = f"""# 기술 전략 분석 보고서

> 생성일시: {now}
> 상태: {status_icon}
{subject_line}
---

"""
    final = header + draft
    return {"final_report": final}
