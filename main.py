"""
tech-strategy-agent 실행 진입점

사용법:
    python main.py
    python main.py --request "SK하이닉스, 삼성전자 HBM4 분석"
"""

import asyncio
import argparse
from graph.graph import graph

DEFAULT_REQUEST = (
    "SK하이닉스, 삼성전자, Micron의 HBM4 및 CoWoS 기술 성숙도를 분석하고 "
    "R&D 우선순위 관점에서 전략적 시사점을 도출해주세요."
)


async def run(user_request: str) -> str:
    config = {"configurable": {"thread_id": "tech-strategy-001"}}

    initial_state = {
        "user_request":    user_request,
        "parsed_intent":   {},
        "rag_results":     [],
        "web_results":     [],
        "trl_estimates":   {},
        "draft_report":    "",
        "review_feedback": [],
        "review_approved": False,
        "rag_sufficient":  False,
        "web_sufficient":  False,
        "retry_count":     0,
        "final_report":    "",
        "messages":        [],
    }

    print("=" * 60)
    print("🚀 Tech Strategy Agent 시작")
    print("=" * 60)

    async for event in graph.astream(initial_state, config=config):
        for node_name, node_output in event.items():
            print(f"[{node_name}] 완료")
            if node_name == "supervisor_review" and node_output.get("review_feedback"):
                for fb in node_output["review_feedback"]:
                    print(f"  → 피드백: {fb}")

    final_state = graph.get_state(config).values
    report = final_state.get("final_report", "보고서 생성 실패")

    print("\n" + "=" * 60)
    print("📄 최종 보고서")
    print("=" * 60)
    print(report)

    # 파일 저장
    with open("output_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n✅ output_report.md 저장 완료")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tech Strategy Agent")
    parser.add_argument("--request", type=str, default=DEFAULT_REQUEST)
    args = parser.parse_args()

    asyncio.run(run(args.request))
