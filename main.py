"""
tech-strategy-agent 실행 진입점

사용법:
    python main.py
    python main.py --request "SK하이닉스, 삼성전자 HBM4 분석"
    python main.py --no-sync          # data/ 동기화 건너뜀
    python main.py --reset-db         # ChromaDB 초기화 후 재적재
"""

import asyncio
import argparse
from pathlib import Path

from graph.graph import graph
from ingest import DATA_DIR, SUPPORTED_EXTENSIONS, load_documents, split_documents, build_vectorstore


def sync_data(data_dir: str = DATA_DIR, reset: bool = False) -> None:
    """data/ 폴더의 문서를 ChromaDB에 동기화합니다."""
    files = [
        f for f in Path(data_dir).rglob("*")
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        print(f"[sync] '{data_dir}' 에 문서가 없습니다 — 동기화 건너뜀\n")
        return

    print(f"[sync] {len(files)}개 파일 감지 → ChromaDB 동기화 시작")
    docs   = load_documents(data_dir)
    chunks = split_documents(docs)
    build_vectorstore(chunks, reset=reset)
    print()

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
    parser.add_argument("--request",  type=str, default=DEFAULT_REQUEST)
    parser.add_argument("--no-sync",  action="store_true", help="data/ 동기화 건너뜀")
    parser.add_argument("--reset-db", action="store_true", help="ChromaDB 초기화 후 재적재")
    args = parser.parse_args()

    if not args.no_sync:
        sync_data(reset=args.reset_db)

    asyncio.run(run(args.request))
