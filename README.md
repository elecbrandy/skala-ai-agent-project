# Tech Strategy Agent

LangGraph 기반 멀티 에이전트 시스템으로, 반도체·HBM·첨단 패키징 분야의 경쟁사 기술 동향을 자동 수집·분석하여 TRL 기반 기술 전략 보고서를 생성합니다.

## Overview

- **Objective** : 공개 간접 지표(특허·학회·채용·공급 발표)를 기반으로 경쟁사 TRL을 추정하고, R&D 우선순위 관점의 전략 시사점을 도출
- **Method** : RAG(Hybrid Retrieval) + 웹 검색 병렬 수집 → Reflection 루프 기반 초안 품질 검토 → Supervisor 승인 후 최종 보고서 출력
- **Tools** : LangGraph, LangChain, ChromaDB, Tavily Search, LangSmith

## Features

- **TRL 기반 기술 성숙도 분석** : NASA 9단계 TRL 척도로 경쟁사별 기술 위치를 정량적으로 추정
- **TRL 4~6 한계 고지** : 핵심 영업 비밀 구간(수율·공정 파라미터)은 간접 지표 기반 추정임을 명시하여 보고서 신뢰성 확보
- **Hybrid RAG 검색** : Dense(MMR) + BM25 앙상블 검색으로 관련 문서 재현율 향상
- **확증 편향 방지 전략** : 웹 검색 쿼리를 `positive / negative / indicator` 3가지 편향 레이블로 분리 생성, 동일 도메인 2건 초과 수집 제한
- **Reflection 루프** : Supervisor가 완결성·근거성·규정 준수·균형성·전략성 5개 기준으로 초안을 검토하고 미충족 시 재작성 지시
- **신뢰도 레이블 자동 부여** : 모든 TRL 추정치에 HIGH / MED / LOW 레이블 표기

## Tech Stack

| Category  | Details                                        |
|-----------|------------------------------------------------|
| Framework | LangGraph, LangChain, Python                   |
| LLM       | GPT-4o (Draft·Supervisor), GPT-4o-mini (Intent) via OpenAI API |
| Retrieval | ChromaDB(MMR) + BM25 Hybrid Ensemble           |
| Embedding | BAAI/bge-m3 via sentence-transformers          |
| Web Search| Tavily Search API                              |
| Observability | LangSmith                                  |

## Agents

- **Intent Agent** : 사용자 자연어 요청에서 키워드·기업·분석 깊이·기간 범위를 파싱
- **RAG Agent** : 키워드 × 기업 조합 다중 쿼리로 내부 벡터 DB에서 관련 문서 검색
- **Web Search Agent** : Tavily로 긍정·반론·간접 지표 쿼리를 병렬 실행, 편향 필터링 적용
- **Draft Agent** : 수집된 RAG·웹 결과와 Supervisor 피드백을 반영하여 TRL 분석 보고서 초안 작성
- **Supervisor Agent** : 초안을 5개 기준(완결성·근거성·규정 준수·균형성·전략성)으로 검토하고 승인 또는 재작성 지시
- **Formatting Agent** : 승인된 초안에 메타 헤더(생성 일시·분석 대상·승인 상태)를 추가하여 최종 보고서 포맷 완성

## Architecture

```
사용자 요청
    │
    ▼
[Intent Agent]  ── 키워드·기업·깊이 파싱
    │
    ├─────────────────────┐
    ▼                     ▼
[RAG Agent]         [Web Search Agent]   ← 병렬 실행
    │                     │
    └──────────┬──────────┘
               ▼
     [Supervisor Review]  ◄──────────────────┐
          │                                  │
     approved?                               │
    ┌──No──┤                                 │
    │      └─► [Increment Retry] ──► [Draft Agent] ─┘
    │Yes
    ▼
[Formatting Agent]
    │
    ▼
[Supervisor Final]
    │
    ▼
 output_report.md
```

## Directory Structure

```
tech-strategy-agent/
├── agents/
│   ├── intent.py          # 사용자 요청 파싱
│   ├── rag.py             # Hybrid RAG 검색
│   ├── web_search.py      # 웹 검색 + 편향 필터링
│   ├── draft.py           # 보고서 초안 작성
│   ├── supervisor.py      # 품질 검토 및 최종 승인
│   └── formatting.py      # 최종 포맷 완성
├── graph/
│   ├── state.py           # AgentState 정의
│   ├── graph.py           # LangGraph 그래프 구성
│   └── edges.py           # 조건부 라우팅 로직
├── chroma_db/             # 벡터 DB (로컬, .gitignore)
├── main.py                # 실행 진입점
├── .env.example           # 환경 변수 템플릿
├── requirements.txt
└── README.md
```

## Getting Started

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경 변수 설정
cp .env.example .env
# .env 에 OPENAI_API_KEY, TAVILY_API_KEY 입력

# 3. 실행 (기본 요청)
python main.py

# 4. 커스텀 요청
python main.py --request "삼성전자, SK하이닉스 HBM4 TRL 분석"
```

## Contributors

- 박병주 :
- 김동우 : 
