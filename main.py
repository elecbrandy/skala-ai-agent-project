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

from graph.graph import graph
from ingest import (
    DATA_DIR,
    needs_sync, load_documents, split_documents, build_vectorstore, _save_manifest,
)


# ── data/ → ChromaDB 동기화 ───────────────────────────────────
def sync_data(data_dir: str = DATA_DIR, reset: bool = False) -> None:
    sync_needed, files = needs_sync(data_dir)

    if not files:
        print(f"[sync] '{data_dir}' 에 문서가 없습니다 — 동기화 건너뜀\n")
        return

    if not sync_needed and not reset:
        print("[sync] ChromaDB 최신 상태 — 동기화 건너뜀\n")
        return

    print(f"[sync] {len(files)}개 파일 변경 감지 → ChromaDB 동기화 시작")
    docs   = load_documents(data_dir)
    chunks = split_documents(docs)
    build_vectorstore(chunks, reset=reset)
    _save_manifest(files)
    print()


# ── MD → PDF 변환 ─────────────────────────────────────────────
def convert_to_pdf(md_text: str, output_path: str = "output_report.pdf") -> bool:
    try:
        import markdown
        from weasyprint import HTML
    except ImportError:
        print("⚠️  PDF 변환 생략: 'pip install markdown weasyprint' 후 재실행하세요.")
        return False

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br"],
    )

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');

  body {{
    font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
    font-size: 11pt;
    line-height: 1.75;
    color: #1a1a1a;
    margin: 0;
    padding: 0;
  }}
  .page {{
    max-width: 800px;
    margin: 0 auto;
    padding: 40px 50px;
  }}
  h1 {{
    font-size: 20pt;
    font-weight: 700;
    border-bottom: 2px solid #1a1a1a;
    padding-bottom: 8px;
    margin-top: 0;
  }}
  h2 {{
    font-size: 14pt;
    font-weight: 700;
    border-left: 4px solid #2563eb;
    padding-left: 10px;
    margin-top: 32px;
  }}
  h3 {{
    font-size: 12pt;
    font-weight: 700;
    margin-top: 20px;
    color: #374151;
  }}
  blockquote {{
    background: #f3f4f6;
    border-left: 4px solid #9ca3af;
    margin: 12px 0;
    padding: 8px 16px;
    color: #4b5563;
    font-size: 10pt;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 10pt;
  }}
  th {{
    background: #1e3a5f;
    color: #ffffff;
    padding: 8px 12px;
    text-align: left;
  }}
  td {{
    padding: 7px 12px;
    border-bottom: 1px solid #e5e7eb;
  }}
  tr:nth-child(even) td {{
    background: #f9fafb;
  }}
  code {{
    background: #f3f4f6;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 10pt;
    font-family: 'Courier New', monospace;
  }}
  pre {{
    background: #1e293b;
    color: #e2e8f0;
    padding: 14px 18px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 9.5pt;
  }}
  pre code {{
    background: none;
    padding: 0;
    color: inherit;
  }}
  hr {{
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 24px 0;
  }}
  a {{ color: #2563eb; }}

  @page {{
    size: A4;
    margin: 20mm 18mm;
    @bottom-center {{
      content: counter(page) " / " counter(pages);
      font-size: 9pt;
      color: #9ca3af;
    }}
  }}
</style>
</head>
<body>
<div class="page">
{html_body}
</div>
</body>
</html>"""

    HTML(string=html).write_pdf(output_path)
    return True


# ── 에이전트 실행 ─────────────────────────────────────────────
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

    # Markdown 저장
    with open("output_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n✅ output_report.md 저장 완료")

    # PDF 변환
    print("[PDF] 변환 중...")
    success = convert_to_pdf(report, "output_report.pdf")
    if success:
        print("✅ output_report.pdf 저장 완료")

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
