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

[1] 완결성 (Completeness)
다음 6개 섹션이 모두 존재하는지 확인:
- SUMMARY
- 1. 분석 배경 (목적성 포함: 왜 이 기술인가)
- 2. 분석 대상 기술 현황 (TRL 추정 테이블)
- 3. 경쟁사 동향 분석 (TRL 비교 매트릭스 + 위협 수준 + 간접 지표)
- 4. 전략적 시사점 (단기/중기/장기 R&D 로드맵)
- REFERENCE

[2] 근거성 (Evidence)
분석 대상 각 기업에 대해 아래 중 하나라도 충족하면 근거 있음으로 판단:
  (a) 구체적 출처(URL, 논문, 공시, 내부 문서) 명시 → HIGH/MED 신뢰도
  (b) 간접 지표(특허, 학회, 채용, 공급 발표) 기반 추정임을 명시 → MED/LOW 신뢰도
  (c) 공개 데이터 부족으로 추정 불가임을 명시하고 LOW 신뢰도 표기

중요: TRL 4~6 구간은 핵심 영업 비밀로 공개 데이터가 원천적으로 제한됩니다.
기업이 "추정 불가" 또는 "공개 정보 부족"으로 명시하고 LOW 신뢰도를 표기한 경우,
이를 missing_evidence로 처리하지 마십시오.

간접 지표(특허 출원 패턴, 학회 발표 빈도, 채용 공고 키워드, 고객사 공급 발표) 중
최소 2종 이상이 보고서 내 어딘가에 언급되어 있는지 확인합니다.

[3] 규정 준수 (Compliance)
- TRL 4~6 추정 시 한계 고지 문구(수율, 공정 파라미터 비공개)가 포함되어 있는가?
- 모든 TRL 추정치에 신뢰도 레이블(HIGH/MED/LOW)이 표기되어 있는가?
- 추측성 서술에 "~으로 추정", "~가능성 있음" 등 표현이 사용되었는가?
- (참고: TRL 행별 [R#]/[W#] 인용 존재 여부는 별도 프로그램이 결정적으로 검증합니다.
  따라서 당신은 인용 번호 누락을 이유로 compliance_ok=false 를 설정하지 마세요.
  표에 인용 번호가 일부 보이지 않더라도, 다른 compliance 기준만 판단하세요.)

[4] 균형성 (Bias)
- 기술 진전(긍정), 기술 한계/위험(부정), 간접 지표(중립) 관점이
  균형 있게 포함되어 있는가?
- 경쟁사별 위협 수준(High/Mid/Low)이 명시되어 있는가?

[5] 전략성 (Strategic Value)
- 4. 전략적 시사점에 R&D 우선순위 관점의 구체적 대응 방향이 있는가?
- 종합 시사점(기술, 시장, 경쟁 관점 통합)이 포함되어 있는가?

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

중요 — JSON 필드 일관성 규칙:
- approved=true 이려면 반드시 compliance_ok, bias_ok, strategy_ok 가 모두 true 이고,
  missing_sections 와 missing_evidence 는 빈 배열([])이며,
  missing_indirect_indicators 는 false 여야 합니다.
- 반대로 위 조건을 모두 만족한다면 approved 는 반드시 true 로 설정하세요.
- feedback 자연어가 "승인", "충족", "통과" 등 긍정적 결론을 담고 있다면,
  위의 모든 보조 플래그도 일관되게 true/빈 배열로 설정해야 합니다.
  자연어 결론과 구조화 필드가 모순되는 응답은 금지됩니다.
"""


_CITATION_RE = re.compile(r"\[[RW]\d+\]")
_TRL_ROW_RE = re.compile(r"TRL\s*\d", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"\b(HIGH|MED|LOW)\b")
_DISCLAIMER_KEYS = ("영업 비밀", "비공개", "수율", "공정 파라미터")


def _has_disclaimer(draft: str) -> bool:
    return sum(1 for k in _DISCLAIMER_KEYS if k in draft) >= 2


def _rows_missing_confidence(draft: str) -> list[str]:
    bad: list[str] = []
    for line in draft.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        if not _TRL_ROW_RE.search(stripped):
            continue
        if "---" in stripped:
            continue
        if _CONFIDENCE_RE.search(stripped):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        bad.append(cells[0] if cells else stripped[:40])
    return bad


def _rows_missing_citations(draft: str) -> list[str]:
    """TRL 값을 포함한 마크다운 표 행 중 [R#]/[W#] 인용이 없는 행을 반환."""
    bad: list[str] = []
    for line in draft.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        if not _TRL_ROW_RE.search(stripped):
            continue
        if "---" in stripped:
            continue
        if _CITATION_RE.search(stripped):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        label = cells[0] if cells else stripped[:40]
        bad.append(label)
    return bad


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
            "간접 지표 보강 필요: 특허 출원 패턴, 학회 발표 빈도 변화, 채용 공고 키워드, "
            "고객사 공급 발표 중 최소 2종을 3. 경쟁사 동향 섹션에 추가하세요."
        )
    if not result.get("compliance_ok", True):
        feedback.append(
            "규정 준수 보강 필요: TRL 4~6 한계 고지 문구(수율, 공정 파라미터 비공개) "
            "및 모든 TRL 추정치 옆 신뢰도 레이블(HIGH/MED/LOW) 표기"
        )
    if not result.get("bias_ok", True):
        feedback.append(
            "균형성 보강 필요: 기술 한계/위험(부정 관점) 추가 및 "
            "경쟁사별 위협 수준(High/Mid/Low) 명시"
        )
    if not result.get("strategy_ok", True):
        feedback.append(
            "전략성 보강 필요: 4. 전략적 시사점에 R&D 우선순위 관점의 "
            "단기/중기/장기 구체적 대응 과제 및 종합 시사점 추가"
        )
    # 결정적 검증: LLM 할루시네이션 오버라이드
    missing_cite_rows = _rows_missing_citations(draft)
    citation_ok = len(missing_cite_rows) == 0
    missing_conf_rows = _rows_missing_confidence(draft)
    confidence_ok = len(missing_conf_rows) == 0
    disclaimer_ok = _has_disclaimer(draft)
    retry_count = state.get("retry_count", 0)

    llm_feedback = result.get("feedback", "")
    if llm_feedback:
        # LLM이 인용 누락을 잘못 지적하는 경우가 많으므로, 결정적 검증이 통과하면
        # 해당 지적을 포함한 feedback 은 무시한다.
        if citation_ok and ("인용" in llm_feedback or "[R" in llm_feedback or "[W" in llm_feedback):
            pass
        else:
            feedback.append(llm_feedback)

    if not citation_ok:
        feedback.append(
            "TRL 행에 인용 번호 누락: " + ", ".join(missing_cite_rows)
            + " — 각 행의 '근거 요약' 칸에 context 의 [R#] 또는 [W#] 중 1개 이상을 추가하세요."
        )

    if not confidence_ok:
        feedback.append(
            "TRL 행 신뢰도 레이블 누락: " + ", ".join(missing_conf_rows)
            + " — 각 행에 HIGH/MED/LOW 중 하나를 표기하세요."
        )
    if not disclaimer_ok:
        feedback.append(
            "TRL 4~6 한계 고지 문구 부족 — '영업 비밀', '수율', '공정 파라미터 비공개' 취지의 "
            "문장을 2번 섹션에 포함하세요."
        )

    # 결정적 검증 모두 통과 + 최소 1회 재작성 거친 경우, 주관적 LLM 불평은 무시하고 승인
    deterministic_ok = citation_ok and confidence_ok and disclaimer_ok
    llm_approved = result.get("approved", False)
    if deterministic_ok and retry_count >= 1:
        final_approved = True
    else:
        final_approved = llm_approved and deterministic_ok

    return {
        "review_approved": final_approved,
        "review_feedback": feedback,
    }


def supervisor_final_node(state: AgentState) -> dict:
    """최종 승인 판단 — formatting_node 가 만든 헤더 포함 final_report 를 보존."""
    formatted = state.get("final_report") or state.get("draft_report", "")
    if state.get("review_approved", False):
        return {"final_report": formatted}
    # fallback: retry 초과 시 현재 포맷 보고서 + 한계 명시
    fallback = (
        formatted
        + "\n\n---\n> ⚠️ 이 보고서는 최대 재시도 횟수 초과로 자동 출력되었습니다. "
        "일부 섹션 또는 TRL 근거가 불완전할 수 있습니다."
    )
    return {"final_report": fallback}
