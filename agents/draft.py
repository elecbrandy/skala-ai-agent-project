from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

DRAFT_PROMPT = """
당신은 반도체·HBM·패키징 기술 전략 분석 보고서 작성 전문가입니다.
수집된 정보를 기반으로 아래 목차 구조로 보고서를 작성하세요.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
목차 구조 및 각 섹션 작성 지침
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## SUMMARY (½ 페이지 이내)
- 핵심 TRL 추정 결과 요약 (기업별 TRL 단계와 신뢰도)
- "왜 지금 이 기술인가" — 전략적 메시지 1~2문장
- 가장 시급한 R&D 대응 방향 1~2개

## 1. 분석 배경 : 왜 지금 이 기술을 분석해야 하는가
- **목적성**: 이 기술이 경쟁 우위에 미치는 구체적 영향 (시장 점유율, 매출, 고객사 Lock-in 등)
- 시장 타이밍: 해당 기술이 임박한 전환점(inflection point)에 있는 근거
- R&D 투자 근거: 지금 투자하지 않으면 발생할 경쟁 격차

## 2. 분석 대상 기술 현황 : 현재 기술 수준과 개발 방향
- 기술 개요: 핵심 원리·구조·성능 지표
- **TRL 추정 테이블** (각 기업별):

  | 기업 | TRL 추정 | 신뢰도 | 근거 요약 |
  |------|----------|--------|-----------|
  | ...  | TRL X    | HIGH/MED/LOW | ... |

- TRL 구간별 공개 가능성 원칙을 명시:
  - TRL 1~3: 논문·특허·학회 발표로 대부분 공개
  - TRL 4~6: **핵심 영업 비밀 구간** — 이하 추정은 공개 간접 지표(특허 출원 패턴,
    학회 발표 빈도 변화, 채용 공고 키워드, 고객사 공급 발표 등) 기반이며,
    실제 수율·공정 파라미터는 비공개로 직접 확인 불가
  - TRL 7~9: 고객사 샘플 공급·양산 발표·실적 공시 등으로 일부 공개

## 3. 경쟁사 동향 분석 : 경쟁사의 기술 전략과 최신 움직임
- **기업별 TRL 비교 매트릭스** (기술 성숙도 + 위협 수준):

  | 기업 | TRL | 기술 성숙도 평가 | 위협 수준 | 핵심 전략 |
  |------|-----|----------------|-----------|-----------|
  | ...  | ... | ...            | High/Mid/Low | ... |

- 각 기업의 **간접 지표** 분석:
  - 특허 출원 패턴: 출원 건수 추이, 주요 특허 IPC 분류
  - 학회 발표 빈도: IEDM·ISSCC·Hot Chips 등 최근 발표 추이
  - 채용 공고 키워드: 핵심 기술 포지션 채용 여부
  - 고객사 공급 동향: 샘플 공급·양산 계약 발표 여부
- 경쟁사별 **강점·약점·위협** 요인 (SWOT 보조 관점)

## 4. 전략적 시사점 : R&D 우선순위 관점에서의 대응 방향 제언
- **R&D 우선순위 매트릭스**: 기술 중요도 × 자사 역량 갭 기준
- 단기 대응 (0~12개월): 즉시 착수 가능한 과제
- 중기 대응 (1~3년): 핵심 역량 확보 과제
- 장기 대응 (3년+): 차세대 포지셔닝 과제
- **종합 시사점**: 기술·시장·경쟁 관점을 통합한 전략 메시지

## REFERENCE
- 형식: [번호] "제목", 출처명, 날짜, URL, [신뢰도: HIGH/MED/LOW]
- 신뢰도 기준:
  - HIGH: 공식 발표·공시·학회 논문 등 직접 근거
  - MED: 간접 지표 2개 이상 교차 확인
  - LOW: 단일 간접 지표 또는 미확인 보도

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
필수 준수사항
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 모든 TRL 추정에 반드시 출처(RAG 청크 또는 웹 URL)를 명시하세요.
2. TRL 4~6 추정 시 반드시 한계 고지 문구를 포함하세요 (위 2섹션 참조).
3. 신뢰도 레이블(HIGH/MED/LOW)을 모든 TRL 추정치 옆에 표기하세요.
4. 긍정(기술 진전)·부정(한계·위험)·중립(간접 지표) 관점을 균형 있게 서술하세요.
5. 수정 피드백이 있다면 해당 섹션을 구체적으로 보강하세요.
6. 추측성 서술은 "~으로 추정", "~가능성 있음" 등 추정 표현으로 명확히 구분하세요.
"""


def draft_node(state: AgentState) -> dict:
    rag      = state.get("rag_results", [])
    web      = state.get("web_results", [])
    feedback = state.get("review_feedback", [])
    intent   = state.get("parsed_intent", {})

    rag_text = "\n".join(
        f"- [{r['source']}][{r['date']}] {r['content'][:300]}"
        for r in rag[:10]
    )
    web_text = "\n".join(
        f"- [{r['bias_label']}][{r['date']}] {r['url']}\n  {r['content'][:200]}"
        for r in web[:10]
    )
    feedback_text = "\n".join(feedback) if feedback else "없음"

    context = f"""
[분석 대상]
키워드: {intent.get('keywords', [])}
경쟁사: {intent.get('companies', [])}

[RAG 검색 결과]
{rag_text or '결과 없음'}

[웹 검색 결과]
{web_text or '결과 없음'}

[수정 피드백]
{feedback_text}
"""

    response = llm.invoke([
        SystemMessage(content=DRAFT_PROMPT),
        HumanMessage(content=context),
    ])
    return {"draft_report": response.content}
