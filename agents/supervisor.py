import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ── Reflection 기준: Quality / Compliance / Operational ────────
REVIEW_PROMPT = """
당신은 기술 전략 분석 보고서의 품질을 검토하는 Supervisor입니다.
아래 5가지 기준으로 초안을 평가하고 JSON으로만 결과를 반환하세요.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[1] 완결성 (Completeness)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
다음 6개 섹션이 모두 존재하는지 확인:
- SUMMARY
- 1. 분석 배경 (목적성 포함: 왜 이 기술인가)
- 2. 분석 대상 기술 현황 (TRL 추정 테이블)
- 3. 경쟁사 동향 분석 (TRL 비교 매트릭스 + 위협 수준 + 간접 지표)
- 4. 전략적 시사점 (단기/중기/장기 R&D 로드맵)
- REFERENCE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[2] 근거성 (Evidence)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 분석 대상 모든 기업의 TRL 추정에 출처가 명시되어 있는가?
- 간접 지표(특허 출원 패턴·학회 발표 빈도·채용 공고·고객사 공급 발표) 중
  최소 2종 이상이 언급되어 있는가?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[3] 규정 준수 (Compliance)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- TRL 4~6 추정 시 한계 고지 문구(수율·공정 파라미터 비공개)가 포함되어 있는가?
- 모든 TRL 추정치에 신뢰도 레이블(HIGH/MED/LOW)이 표기되어 있는가?
- 추측성 서술에 "~으로 추정", "~가능성 있음" 등 표현이 사용되었는가?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[4] 균형성 (Bias)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 기술 진전(긍정), 기술 한계·위험(부정), 간접 지표(중립) 관점이
  균형 있게 포함되어 있는가?
- 경쟁사별 위협 수준(High/Mid/Low)이 명시되어 있는가?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[5] 전략성 (Strategic Value)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 4. 전략적 시사점에 R&D 우선순위 관점의 구체적 대응 방향이 있는가?
- 종합 시사점(기술·시장·경쟁 관점 통합)이 포함되어 있는가?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
반환 형식 (JSON만 반환, 다른 텍스트 포함 금지):
{
  "approved": true/false,
  "missing_sections": ["섹션명"],
  "missing_evidence": ["회사명"],
  "missing_indirect_indicators": true/false,
  "compliance_ok": true/false,
  "bias_ok": true/false,
  "strategy_ok": true/false,
  "feedback": "구체적 수정 지시 — 어느 섹션의 무엇을 어떻게 보강할지"
}

승인(approved: true) 조건: 5가지 기준을 모두 충족해야 합니다.
"""


def supervisor_review_node(state: AgentState) -> dict:
    """초안 Reflection — Quality / Compliance / Bias 검토"""
    draft = state.get("draft_report", "")
    if not draft:
        return {
            "review_approved": False,
            "review_feedback": ["초안 없음. Draft Agent 호출 필요."],
        }

    response = llm.invoke([
        SystemMessage(content=REVIEW_PROMPT),
        HumanMessage(content=f"검토할 초안:\n\n{draft}"),
    ])

    try:
        # LLM이 ```json ... ``` 블록으로 감싸는 경우 제거 후 파싱
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw.strip())
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"approved": False, "feedback": response.content}

    feedback: list[str] = []
    if result.get("missing_sections"):
        feedback.append(f"누락 섹션: {', '.join(result['missing_sections'])}")
    if result.get("missing_evidence"):
        feedback.append(f"TRL 근거 누락 기업: {', '.join(result['missing_evidence'])}")
    if result.get("missing_indirect_indicators"):
        feedback.append(
            "간접 지표 보강 필요: 특허 출원 패턴·학회 발표 빈도 변화·채용 공고 키워드·"
            "고객사 공급 발표 중 최소 2종을 3. 경쟁사 동향 섹션에 추가하세요."
        )
    if not result.get("compliance_ok", True):
        feedback.append(
            "규정 준수 보강 필요: TRL 4~6 한계 고지 문구(수율·공정 파라미터 비공개) "
            "및 모든 TRL 추정치 옆 신뢰도 레이블(HIGH/MED/LOW) 표기"
        )
    if not result.get("bias_ok", True):
        feedback.append(
            "균형성 보강 필요: 기술 한계·위험(부정 관점) 추가 및 "
            "경쟁사별 위협 수준(High/Mid/Low) 명시"
        )
    if not result.get("strategy_ok", True):
        feedback.append(
            "전략성 보강 필요: 4. 전략적 시사점에 R&D 우선순위 관점의 "
            "단기/중기/장기 구체적 대응 과제 및 종합 시사점 추가"
        )
    if result.get("feedback"):
        feedback.append(result["feedback"])

    return {
        "review_approved": result.get("approved", False),
        "review_feedback": feedback,
        "retry_count":     state.get("retry_count", 0),
    }


def supervisor_final_node(state: AgentState) -> dict:
    """최종 승인 판단"""
    if state.get("review_approved", False):
        return {"final_report": state["draft_report"]}
    # fallback: retry 초과 시 현재 초안 + 한계 명시
    fallback = (
        state.get("draft_report", "")
        + "\n\n---\n> ⚠️ 이 보고서는 최대 재시도 횟수 초과로 자동 출력되었습니다. "
        "일부 섹션 또는 TRL 근거가 불완전할 수 있습니다."
    )
    return {"final_report": fallback}
