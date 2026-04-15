"""
RAG Retrieval 성능 평가
- 지표: Hit Rate@K, MRR (Mean Reciprocal Rank)
- 목표: Hit Rate@5 >= 0.70 / MRR >= 0.55
- Retriever 비교: Dense | Dense+MMR | BM25 | Hybrid(최종 선정)

사용법:
    python rag_evaluation.py
    python rag_evaluation.py --k 5 --save
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["axes.unicode_minus"] = False
try:
    matplotlib.rcParams["font.family"] = "AppleGothic"   # macOS
except Exception:
    pass

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# ── 평가 데이터셋 ───────────────────────────────────────────────
# 실제 환경에서는 반도체·HBM 도메인 문서로 QA 페어를 구성
EVAL_DATASET = [
    {
        "question": "SK하이닉스 HBM3E의 현재 TRL 단계는?",
        "relevant_doc_ids": ["doc_001", "doc_003"],
    },
    {
        "question": "CoWoS-L 패키징의 양산 도전 과제는?",
        "relevant_doc_ids": ["doc_012", "doc_015", "doc_019"],
    },
    {
        "question": "Micron HBM4 특허 출원 현황은?",
        "relevant_doc_ids": ["doc_031"],
    },
    {
        "question": "Hybrid Bonding 기술 성숙도 비교: TSMC vs 인텔",
        "relevant_doc_ids": ["doc_042", "doc_044"],
    },
    {
        "question": "HBM4 로드맵 및 양산 시점 예측은?",
        "relevant_doc_ids": ["doc_055", "doc_058"],
    },
    {
        "question": "삼성전자 CoWoS 공정 TRL 추정 근거는?",
        "relevant_doc_ids": ["doc_060", "doc_063"],
    },
    {
        "question": "SK하이닉스 HBM4 채용 공고 키워드 분석",
        "relevant_doc_ids": ["doc_071"],
    },
    {
        "question": "TSMC SoIC 기술의 학회 발표 빈도 변화는?",
        "relevant_doc_ids": ["doc_082", "doc_085"],
    },
]

# ── 샘플 문서 (실제 환경에서는 VectorStore 문서로 대체) ─────────
SAMPLE_DOCS = [
    Document(
        page_content="SK하이닉스는 2024년 IEDM에서 HBM3E 12Hi 샘플 공급을 발표했다. "
                     "TRL 7~8 수준으로 양산 적합성 검증 단계에 있다.",
        metadata={"doc_id": "doc_001", "company": "SK하이닉스", "date": "2024-12"},
    ),
    Document(
        page_content="HBM3E는 TRL 7~8 수준으로 평가되며 양산 적합성 검증 단계에 있다. "
                     "전력 효율은 전세대 대비 30% 향상되었다.",
        metadata={"doc_id": "doc_003", "company": "SK하이닉스", "date": "2024-06"},
    ),
    Document(
        page_content="CoWoS-L는 TSMC의 이등분 적층 패키징 기술이며 "
                     "수율 문제가 발목을 잡고 있다. TRL 5~6 추정.",
        metadata={"doc_id": "doc_012", "company": "TSMC", "date": "2024-09"},
    ),
    Document(
        page_content="CoWoS-L 양산 도전 과제로는 interposer 수율, 열 관리, "
                     "비용 구조가 꼽힌다.",
        metadata={"doc_id": "doc_015", "company": "TSMC", "date": "2024-07"},
    ),
    Document(
        page_content="TSMC CoWoS-L 공정 난이도는 CoWoS-S 대비 높으며 "
                     "고객사 요구 충족에 어려움이 있다.",
        metadata={"doc_id": "doc_019", "company": "TSMC", "date": "2024-05"},
    ),
    Document(
        page_content="Micron은 2024년 HBM4 관련 특허를 32건 출원했다. "
                     "TSV 밀도 및 열 방출 구조가 핵심 특허 영역이다.",
        metadata={"doc_id": "doc_031", "company": "Micron", "date": "2024-11"},
    ),
    Document(
        page_content="Intel의 Hybrid Bonding 기술은 EMIB 기반으로 "
                     "TRL 6~7 수준으로 추정된다.",
        metadata={"doc_id": "doc_042", "company": "Intel", "date": "2024-08"},
    ),
    Document(
        page_content="TSMC의 SoIC-X Hybrid Bonding은 TRL 7로 추정되며 "
                     "2024 ISSCC에서 기술 세부 사항을 공개했다.",
        metadata={"doc_id": "doc_044", "company": "TSMC", "date": "2024-02"},
    ),
    Document(
        page_content="HBM4 양산 시점은 2026년으로 예측된다. "
                     "주요 업체들의 로드맵은 2025년 샘플 공급을 목표로 한다.",
        metadata={"doc_id": "doc_055", "company": "general", "date": "2024-10"},
    ),
    Document(
        page_content="HBM4는 HBM3E 대비 대역폭 2배를 목표로 하며 "
                     "16Hi 스태킹이 핵심 기술 과제다.",
        metadata={"doc_id": "doc_058", "company": "general", "date": "2024-09"},
    ),
    Document(
        page_content="삼성전자 CoWoS 공정 관련 채용 공고에서 "
                     "interposer 설계 엔지니어 수요가 증가하고 있다.",
        metadata={"doc_id": "doc_060", "company": "삼성전자", "date": "2024-11"},
    ),
    Document(
        page_content="삼성전자는 2024 IEDM에서 2.5D 패키징 공정 발표를 통해 "
                     "CoWoS 유사 기술의 TRL 5~6 수준을 간접 시사했다.",
        metadata={"doc_id": "doc_063", "company": "삼성전자", "date": "2024-12"},
    ),
    Document(
        page_content="SK하이닉스 채용 공고에서 HBM4 TSV 공정, "
                     "어드밴스드 패키징 관련 키워드가 급증했다.",
        metadata={"doc_id": "doc_071", "company": "SK하이닉스", "date": "2024-10"},
    ),
    Document(
        page_content="TSMC SoIC는 IEDM 2022 이후 학회 발표 빈도가 "
                     "연 3→8건으로 증가하며 기술 성숙도 상승을 시사한다.",
        metadata={"doc_id": "doc_082", "company": "TSMC", "date": "2024-06"},
    ),
    Document(
        page_content="SoIC 관련 ISSCC 발표가 2023~2024년 집중되어 "
                     "TRL 6~7 수준 진입을 간접 확인할 수 있다.",
        metadata={"doc_id": "doc_085", "company": "TSMC", "date": "2024-02"},
    ),
]


# ── 평가 함수 ──────────────────────────────────────────────────
def hit_rate_at_k(results: list[dict], relevant_ids: list[str], k: int) -> float:
    """Hit Rate@K: 상위 k개 결과 안에 정답 문서가 1개라도 있으면 1.0"""
    top_k_ids = [r["doc_id"] for r in results[:k]]
    return 1.0 if any(rid in top_k_ids for rid in relevant_ids) else 0.0


def reciprocal_rank(results: list[dict], relevant_ids: list[str]) -> float:
    """Reciprocal Rank: 첫 번째 정답 문서 순위의 역수"""
    for rank, result in enumerate(results, start=1):
        if result["doc_id"] in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retriever(
    retriever,
    eval_dataset: list[dict],
    k: int = 5,
) -> dict:
    """전체 데이터셋에 대해 Hit Rate@K 와 MRR 산정"""
    hit_rates, rrs = [], []

    for item in eval_dataset:
        docs = retriever.invoke(item["question"])
        results = [
            {
                "doc_id":  doc.metadata.get("doc_id", ""),
                "content": doc.page_content[:80],
            }
            for doc in docs
        ]
        hit_rates.append(hit_rate_at_k(results, item["relevant_doc_ids"], k=k))
        rrs.append(reciprocal_rank(results, item["relevant_doc_ids"]))

    return {
        f"hit_rate@{k}":       round(float(np.mean(hit_rates)), 4),
        "mrr":                  round(float(np.mean(rrs)),       4),
        "hit_rates_per_query":  hit_rates,
        "rrs_per_query":        rrs,
    }


# ── Retriever 세터 ─────────────────────────────────────────────
def setup_retrievers(k: int = 5):
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma.from_documents(SAMPLE_DOCS, embedding_model)

    dense = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    mmr = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3, "lambda_mult": 0.7},
    )
    bm25 = BM25Retriever.from_documents(SAMPLE_DOCS)
    bm25.k = k

    hybrid = EnsembleRetriever(
        retrievers=[
            vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": k * 3, "lambda_mult": 0.7},
            ),
            BM25Retriever.from_documents(SAMPLE_DOCS),
        ],
        weights=[0.6, 0.4],
    )
    return {
        "Dense (Baseline)":        dense,
        "Dense + MMR":             mmr,
        "BM25":                    bm25,
        "Hybrid (Dense+BM25+MMR)": hybrid,
    }


# ── 시각화 ─────────────────────────────────────────────────────
def plot_comparison(summary: dict, k: int, save: bool = False):
    names      = list(summary.keys())
    hr_vals    = [summary[n][f"hit_rate@{k}"] for n in names]
    mrr_vals   = [summary[n]["mrr"]           for n in names]
    colors     = ["#d0d0d0", "#9ec8e8", "#9ec8e8", "#2E75B6"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"RAG Retriever 성능 비교  (K={k})", fontsize=14, fontweight="bold")

    for ax, vals, title, threshold, label in [
        (axes[0], hr_vals,  f"Hit Rate@{k}", 0.70, f"목표 0.70"),
        (axes[1], mrr_vals, "MRR",           0.55, "목표 0.55"),
    ]:
        bars = ax.barh(names, vals, color=colors, edgecolor="white", height=0.5)
        ax.axvline(x=threshold, color="red", linestyle="--", linewidth=1.2, label=label)
        ax.set_xlim(0, 1.05)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=10)

    plt.tight_layout()
    if save:
        plt.savefig("retriever_evaluation.png", dpi=150, bbox_inches="tight")
        print("📊 retriever_evaluation.png 저장 완료")
    plt.show()


def plot_hit_rate_by_k(retriever, eval_dataset: list[dict], save: bool = False):
    k_values = [1, 3, 5, 10]
    hr_by_k  = [
        evaluate_retriever(retriever, eval_dataset, k=k)[f"hit_rate@{k}"]
        for k in k_values
    ]

    plt.figure(figsize=(7, 4))
    plt.plot(k_values, hr_by_k, marker="o", color="#2E75B6", linewidth=2, markersize=7)
    plt.axhline(y=0.70, color="red", linestyle="--", linewidth=1.2, label="목표 0.70")
    for x, y in zip(k_values, hr_by_k):
        plt.text(x + 0.1, y + 0.01, f"{y:.2f}", fontsize=10)
    plt.xlabel("K")
    plt.ylabel("Hit Rate@K")
    plt.title("Hybrid Retriever — K별 Hit Rate 추이", fontsize=12)
    plt.xticks(k_values)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig("hit_rate_by_k.png", dpi=150, bbox_inches="tight")
        print("📊 hit_rate_by_k.png 저장 완료")
    plt.show()


# ── 메인 ───────────────────────────────────────────────────────
def main(k: int = 5, save: bool = False):
    print("=" * 60)
    print("🔍 RAG Retrieval 성능 평가 시작")
    print(f"   K={k} | 평가 쿼리 수: {len(EVAL_DATASET)}")
    print("=" * 60)

    print("\n[임베딩 모델 로드 중...] BAAI/bge-m3")
    retrievers = setup_retrievers(k=k)

    summary: dict = {}
    for name, retriever in retrievers.items():
        print(f"\n[{name}] 평가 중...")
        metrics = evaluate_retriever(retriever, EVAL_DATASET, k=k)
        summary[name] = metrics
        hr_ok  = "✅" if metrics[f"hit_rate@{k}"] >= 0.70 else "❌"
        mrr_ok = "✅" if metrics["mrr"] >= 0.55 else "❌"
        print(f"  Hit Rate@{k}: {metrics[f'hit_rate@{k}']:.4f} {hr_ok}")
        print(f"  MRR        : {metrics['mrr']:.4f} {mrr_ok}")

    # DataFrame 출력
    print("\n" + "=" * 60)
    print("📋 최종 비교 결과")
    print("=" * 60)
    df = pd.DataFrame([
        {
            "Retriever":        name,
            f"Hit Rate@{k}":    m[f"hit_rate@{k}"],
            "MRR":              m["mrr"],
            "목표 달성 (HR)":   "✅" if m[f"hit_rate@{k}"] >= 0.70 else "❌",
            "목표 달성 (MRR)":  "✅" if m["mrr"] >= 0.55 else "❌",
        }
        for name, m in summary.items()
    ])
    print(df.to_string(index=False))

    # 쿼리별 상세 분석 (Hybrid)
    print("\n" + "=" * 60)
    print("🔎 쿼리별 상세 분석 — Hybrid Retriever")
    print("=" * 60)
    hybrid_m = summary["Hybrid (Dense+BM25+MMR)"]
    for i, item in enumerate(EVAL_DATASET):
        hr  = hybrid_m["hit_rates_per_query"][i]
        rr  = hybrid_m["rrs_per_query"][i]
        icon = "✅" if hr == 1.0 else "❌"
        print(f"{icon} Q{i+1}: {item['question'][:45]}...")
        print(f"     Hit@{k}={hr:.1f}  RR={rr:.3f}")

    # GitHub README 포맷 출력
    print("\n" + "=" * 60)
    print("📝 GitHub README 반영용 테이블")
    print("=" * 60)
    print(f"| Retriever | Hit Rate@{k} | MRR |")
    print("|-----------|------------|-----|")
    for name, m in summary.items():
        bold = "**" if name == "Hybrid (Dense+BM25+MMR)" else ""
        print(f"| {bold}{name}{bold} | {bold}{m[f'hit_rate@{k}']:.2f}{bold} | {bold}{m['mrr']:.2f}{bold} |")

    # 시각화
    plot_comparison(summary, k=k, save=save)
    hybrid_retriever = list(setup_retrievers(k=k).values())[-1]
    plot_hit_rate_by_k(hybrid_retriever, EVAL_DATASET, save=save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Retrieval 평가")
    parser.add_argument("--k",    type=int,  default=5,     help="Hit Rate@K의 K값 (기본: 5)")
    parser.add_argument("--save", action="store_true",      help="차트 PNG 파일 저장")
    args = parser.parse_args()

    main(k=args.k, save=args.save)
