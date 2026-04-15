from datetime import datetime
from langchain_community.tools.tavily_search import TavilySearchResults
from graph.state import AgentState

WEB_SCORE_THRESHOLD = 0.5
MAX_SAME_DOMAIN     = 2   # 동일 도메인 허용 최대 수

tavily = TavilySearchResults(
    max_results=5,
    include_answer=False,
    include_raw_content=False,
    include_images=False,
)


def build_queries(intent: dict) -> list[dict]:
    """
    확증 편향 방지: 긍정 / 반론 / 간접 지표 쿼리를 병렬 생성
    bias_label: positive | negative | indicator
    """
    keywords  = intent.get("keywords", [])
    companies = intent.get("companies", [])
    year      = datetime.now().year
    queries: list[dict] = []

    for kw in keywords:
        queries.append({"q": f"{kw} 최신 개발 동향 {year}",        "bias": "positive"})
        queries.append({"q": f"{kw} 기술적 한계 양산 도전 과제",   "bias": "negative"})
        for company in companies:
            queries.append({"q": f"{company} {kw} 특허 출원 {year}", "bias": "indicator"})
            queries.append({"q": f"{company} {kw} IEDM ISSCC",        "bias": "indicator"})

    return queries


def web_search_node(state: AgentState) -> dict:
    intent  = state["parsed_intent"]
    queries = build_queries(intent)

    results: list[dict]      = []
    domain_count: dict[str, int] = {}

    for item in queries:
        try:
            raw = tavily.invoke(item["q"])
            for r in raw:
                domain = r["url"].split("/")[2]

                # 동일 도메인 2개 초과 제한 (편향 방지)
                if domain_count.get(domain, 0) >= MAX_SAME_DOMAIN:
                    continue
                domain_count[domain] = domain_count.get(domain, 0) + 1

                results.append({
                    "url":        r["url"],
                    "date":       r.get("published_date", ""),
                    "bias_label": item["bias"],
                    "content":    r["content"],
                    "score":      r.get("score", 0.0),
                })
        except Exception:
            continue

    avg_score = (
        sum(r["score"] for r in results) / len(results) if results else 0.0
    )

    return {
        "web_results":    results,
        "web_sufficient": avg_score >= WEB_SCORE_THRESHOLD,
    }
