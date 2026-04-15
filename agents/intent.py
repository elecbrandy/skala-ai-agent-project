import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import AgentState

llm = ChatOpenAI(model="gpt-4o", temperature=0)

SYSTEM_PROMPT = """
당신은 기술 전략 분석 요청을 파싱하는 에이전트입니다.
사용자의 자연어 요청에서 아래 정보를 추출하여 JSON으로 반환하세요.

반환 형식:
{
  "keywords": ["HBM", "CoWoS", "Hybrid Bonding"],
  "companies": ["SK하이닉스", "삼성전자", "Micron", "TSMC"],
  "depth": "detailed",
  "date_range": {"recent_months": 6, "mid_years": 3, "long_years": 5}
}

depth 옵션: brief | standard | detailed
JSON만 반환하고 다른 텍스트는 포함하지 마세요.
"""

DEFAULT_INTENT = {
    "keywords": ["HBM", "첨단 패키징"],
    "companies": ["SK하이닉스", "삼성전자", "Micron", "TSMC"],
    "depth": "standard",
    "date_range": {"recent_months": 6, "mid_years": 3, "long_years": 5},
}


def intent_node(state: AgentState) -> dict:
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=state["user_request"]),
    ])
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        parsed = DEFAULT_INTENT
    return {"parsed_intent": parsed}
