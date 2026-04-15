from graph.state import AgentState

MAX_RETRY = 3


# NOTE: route_after_search 는 현재 graph.py 에서 사용되지 않습니다.
# graph.py 는 intent_node → fan_out(Send) 으로 rag/web 을 병렬 실행한 뒤
# 두 노드의 공통 후속 노드인 supervisor_review 로 직접 합류(join)합니다.
# 검색 불충분 시 재시도 로직이 필요하다면 이 함수를 fan_out 대신 조건부 엣지로 연결하세요.
def route_after_search(state: AgentState) -> str:
    """RAG + Web 검색 결과 충분성 확인 후 라우팅 (미사용)"""
    rag_ok = state.get("rag_sufficient", True)
    web_ok = state.get("web_sufficient", True)
    retry  = state.get("retry_count", 0)

    if not rag_ok and retry < MAX_RETRY:
        return "rag_node"
    if not web_ok and retry < MAX_RETRY:
        return "web_node"
    return "draft_node"


def route_after_review(state: AgentState) -> str:
    """Supervisor 검토 결과 라우팅"""
    retry    = state.get("retry_count", 0)
    approved = state.get("review_approved", False)

    if approved:
        return "formatting_node"
    if retry >= MAX_RETRY:
        return "formatting_node"   # fallback: 현 상태 출력
    return "draft_node"            # 재작성
