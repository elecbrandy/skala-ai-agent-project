"""
RAG Retrieval 성능 평가 — 실제 data/ 기반

기존 문제: 평가 쿼리를 샘플 문서 보고 직접 작성 → 쿼리-문서 완벽 일치 → Hit Rate 1.0 (무의미)
개선 방식:
  1. ChromaDB에서 실제 청크를 무작위 샘플링
  2. GPT-4o-mini로 각 청크에 대한 질문 자동 생성
  3. 전체 코퍼스에서 검색 후 원본 청크 회수 여부로 평가

사용법:
    python rag_evaluation.py                          # QA 쌍 생성 + 평가
    python rag_evaluation.py --load eval_dataset.json # 기존 QA 쌍 재사용
    python rag_evaluation.py --n 30 --k 5 --save     # 30쌍, K=5, 차트 저장
"""

import os
import json
import random
import hashlib
import argparse
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    matplotlib.rcParams["font.family"] = "AppleGothic"
except Exception:
    pass
matplotlib.rcParams["axes.unicode_minus"] = False

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

CHROMA_DIR  = os.getenv("CHROMA_DIR", "./chroma_db")
EMBED_MODEL = "BAAI/bge-m3"
RANDOM_SEED = 42


# ── 문서 로드 ─────────────────────────────────────────────────
def _chunk_id(text: str) -> str:
    """청크 고유 식별자: 앞 200자 MD5 해시"""
    return hashlib.md5(text[:200].encode()).hexdigest()[:12]


def load_corpus() -> tuple[list[Document], Chroma]:
    print("[1/4] ChromaDB 로드 중...")
    embed = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=embed)
    raw = vs.get()

    docs = []
    for content, meta in zip(
        raw.get("documents", []),
        raw.get("metadatas", [{}] * len(raw.get("documents", []))),
    ):
        if len(content.strip()) < 150:   # 너무 짧은 청크 제외 (목차, 헤더 등)
            continue
        doc = Document(page_content=content, metadata=dict(meta))
        doc.metadata["_chunk_id"] = _chunk_id(content)
        docs.append(doc)

    print(f"     총 {len(docs)}개 청크 로드 (짧은 청크 제외)")
    return docs, vs


# ── QA 쌍 생성 ────────────────────────────────────────────────
def sample_chunks(docs: list[Document], n: int, seed: int = RANDOM_SEED) -> list[Document]:
    """소스별 균등 샘플링으로 편향 방지"""
    by_source: dict[str, list[Document]] = {}
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        by_source.setdefault(src, []).append(doc)

    rng = random.Random(seed)
    sampled: list[Document] = []
    per_src = max(1, n // len(by_source))

    for chunks in by_source.values():
        k = min(per_src, len(chunks))
        sampled.extend(rng.sample(chunks, k))

    rng.shuffle(sampled)
    return sampled[:n]


def generate_qa_pairs(chunks: list[Document]) -> list[dict]:
    print(f"[2/4] GPT-4o-mini로 QA 쌍 {len(chunks)}개 생성 중...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    system_prompt = (
        "당신은 반도체·HBM·첨단 패키징 기술 문서에 대한 검색 평가용 질문을 만드는 전문가입니다.\n"
        "주어진 문서 청크를 읽고, 아래 조건에 맞는 질문 1개만 한국어로 반환하세요.\n\n"
        "조건:\n"
        "1. 이 청크에만 있는 구체적 정보(수치, 기술명, 회사명 등)를 묻는 질문\n"
        "2. 청크 문장을 그대로 복사하지 말 것\n"
        "3. 질문만 반환, 다른 텍스트 없음"
    )

    pairs = []
    for i, chunk in enumerate(chunks):
        try:
            resp = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"문서 청크:\n{chunk.page_content[:600]}"),
            ])
            question = resp.content.strip()
            pairs.append({
                "question":        question,
                "chunk_id":        chunk.metadata["_chunk_id"],
                "source":          chunk.metadata.get("source", "unknown"),
                "content_preview": chunk.page_content[:120],
            })
            print(f"  Q{i+1:02d}: {question[:65]}...")
        except Exception as e:
            print(f"  [오류] Q{i+1}: {e}")

    return pairs


# ── 평가 지표 ─────────────────────────────────────────────────
def hit_at_k(retrieved: list[str], target: str, k: int) -> float:
    return 1.0 if target in retrieved[:k] else 0.0


def reciprocal_rank(retrieved: list[str], target: str) -> float:
    for rank, rid in enumerate(retrieved, start=1):
        if rid == target:
            return 1.0 / rank
    return 0.0


def evaluate(retriever, pairs: list[dict], k: int) -> dict:
    hits, rrs = [], []
    for pair in pairs:
        try:
            results = retriever.invoke(pair["question"])
            retrieved_ids = [_chunk_id(doc.page_content) for doc in results]
            hits.append(hit_at_k(retrieved_ids, pair["chunk_id"], k))
            rrs.append(reciprocal_rank(retrieved_ids, pair["chunk_id"]))
        except Exception:
            hits.append(0.0)
            rrs.append(0.0)
    return {
        f"hit_rate@{k}":      round(float(np.mean(hits)), 4),
        "mrr":                 round(float(np.mean(rrs)),  4),
        "hits_per_query":      hits,
        "rrs_per_query":       rrs,
    }


# ── Retriever 구성 ────────────────────────────────────────────
def build_retrievers(docs: list[Document], vs: Chroma, k: int) -> dict:
    bm25_r = BM25Retriever.from_documents(docs)
    bm25_r.k = k

    return {
        "Dense (Baseline)": vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        ),
        "Dense + MMR": vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 3, "lambda_mult": 0.7},
        ),
        "BM25": bm25_r,
        "Hybrid (Dense+BM25+MMR)": EnsembleRetriever(
            retrievers=[
                vs.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": k, "fetch_k": k * 3, "lambda_mult": 0.7},
                ),
                BM25Retriever.from_documents(docs, k=k),
            ],
            weights=[0.6, 0.4],
        ),
    }


# ── 시각화 ───────────────────────────────────────────────────
def plot_comparison(summary: dict, k: int, save: bool):
    names   = list(summary.keys())
    hr_vals = [summary[n][f"hit_rate@{k}"] for n in names]
    mr_vals = [summary[n]["mrr"]            for n in names]
    colors  = ["#d0d0d0", "#9ec8e8", "#9ec8e8", "#2E75B6"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"RAG Retriever 성능 비교  (K={k}, 실제 data/ 기반)", fontsize=13, fontweight="bold")

    for ax, vals, title, thr, lbl in [
        (axes[0], hr_vals, f"Hit Rate@{k}", 0.70, "목표 0.70"),
        (axes[1], mr_vals, "MRR",           0.55, "목표 0.55"),
    ]:
        bars = ax.barh(names, vals, color=colors, edgecolor="white", height=0.5)
        ax.axvline(x=thr, color="red", linestyle="--", linewidth=1.2, label=lbl)
        ax.set_xlim(0, 1.1)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=10)

    plt.tight_layout()
    if save:
        plt.savefig("retriever_evaluation.png", dpi=150, bbox_inches="tight")
        print("  저장: retriever_evaluation.png")
    plt.close()


def plot_hit_by_k(vs: Chroma, docs: list[Document], pairs: list[dict], save: bool):
    k_vals = [1, 3, 5, 10]
    hybrid = EnsembleRetriever(
        retrievers=[
            vs.as_retriever(search_type="mmr", search_kwargs={"k": max(k_vals), "fetch_k": max(k_vals) * 3, "lambda_mult": 0.7}),
            BM25Retriever.from_documents(docs, k=max(k_vals)),
        ],
        weights=[0.6, 0.4],
    )
    hr_by_k = [evaluate(hybrid, pairs, k)[f"hit_rate@{k}"] for k in k_vals]

    plt.figure(figsize=(7, 4))
    plt.plot(k_vals, hr_by_k, marker="o", color="#2E75B6", linewidth=2, markersize=7)
    plt.axhline(y=0.70, color="red", linestyle="--", linewidth=1.2, label="목표 0.70")
    for x, y in zip(k_vals, hr_by_k):
        plt.text(x + 0.1, y + 0.015, f"{y:.2f}", fontsize=10)
    plt.xlabel("K")
    plt.ylabel("Hit Rate@K")
    plt.title("Hybrid Retriever — K별 Hit Rate (실제 data/ 기반)", fontsize=12)
    plt.xticks(k_vals)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig("hit_rate_by_k.png", dpi=150, bbox_inches="tight")
        print("  저장: hit_rate_by_k.png")
    plt.close()


# ── 메인 ─────────────────────────────────────────────────────
def main(n: int, k: int, save: bool, load_path: str | None):
    print("=" * 60)
    print("  RAG Retrieval 성능 평가  (실제 data/ 기반)")
    print(f"  QA 쌍: {n}개 | K={k}")
    print("=" * 60 + "\n")

    docs, vs = load_corpus()

    # QA 쌍 로드 or 생성
    if load_path and os.path.exists(load_path):
        with open(load_path, encoding="utf-8") as f:
            pairs = json.load(f)
        print(f"[2/4] QA 쌍 로드: {load_path} ({len(pairs)}개)\n")
    else:
        chunks = sample_chunks(docs, n)
        pairs  = generate_qa_pairs(chunks)
        out_path = "eval_dataset.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        print(f"\n  QA 쌍 저장: {out_path}\n")

    # Retriever 구성 및 평가
    print(f"[3/4] 4종 Retriever 평가 중... (K={k})")
    retrievers = build_retrievers(docs, vs, k)
    summary: dict = {}
    for name, retriever in retrievers.items():
        print(f"\n  [{name}]")
        m = evaluate(retriever, pairs, k)
        summary[name] = m
        hr_ok  = "✅" if m[f"hit_rate@{k}"] >= 0.70 else "❌"
        mrr_ok = "✅" if m["mrr"]            >= 0.55 else "❌"
        print(f"    Hit Rate@{k}: {m[f'hit_rate@{k}']:.4f} {hr_ok}")
        print(f"    MRR        : {m['mrr']:.4f}  {mrr_ok}")

    # 결과 출력
    print("\n" + "=" * 60)
    print("  최종 비교 결과")
    print("=" * 60)
    df = pd.DataFrame([
        {
            "Retriever":       name,
            f"Hit Rate@{k}":   m[f"hit_rate@{k}"],
            "MRR":             m["mrr"],
            "HR 목표 달성":    "✅" if m[f"hit_rate@{k}"] >= 0.70 else "❌",
            "MRR 목표 달성":   "✅" if m["mrr"]            >= 0.55 else "❌",
        }
        for name, m in summary.items()
    ])
    print(df.to_string(index=False))

    # 쿼리별 상세 (Hybrid)
    hybrid_m = summary["Hybrid (Dense+BM25+MMR)"]
    print("\n" + "=" * 60)
    print("  쿼리별 상세 — Hybrid Retriever")
    print("=" * 60)
    for i, pair in enumerate(pairs):
        hr   = hybrid_m["hits_per_query"][i]
        rr   = hybrid_m["rrs_per_query"][i]
        icon = "✅" if hr == 1.0 else "❌"
        print(f"  {icon} Q{i+1:02d} [{pair['source']}]")
        print(f"       Q: {pair['question'][:60]}...")
        print(f"       Hit@{k}={hr:.1f}  RR={rr:.3f}")

    # GitHub README용 테이블
    print("\n" + "=" * 60)
    print("  GitHub README 반영용 테이블")
    print("=" * 60)
    print(f"| Retriever | Hit Rate@{k} | MRR |")
    print("|-----------|:----------:|:---:|")
    for name, m in summary.items():
        bold = "**" if name == "Hybrid (Dense+BM25+MMR)" else ""
        print(f"| {bold}{name}{bold} | {bold}{m[f'hit_rate@{k}']:.2f}{bold} | {bold}{m['mrr']:.2f}{bold} |")

    # 시각화
    print("\n[4/4] 차트 생성 중...")
    plot_comparison(summary, k, save)
    plot_hit_by_k(vs, docs, pairs, save)
    print("\n✅ 평가 완료")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Retrieval 평가 (실제 data/ 기반)")
    parser.add_argument("--n",         type=int,  default=20,   help="생성할 QA 쌍 수 (기본: 20)")
    parser.add_argument("--k",         type=int,  default=5,    help="Hit Rate@K의 K값 (기본: 5)")
    parser.add_argument("--save",      action="store_true",     help="차트 PNG 저장")
    parser.add_argument("--load",      type=str,  default=None, help="기존 QA 쌍 JSON 재사용 경로")
    args = parser.parse_args()

    main(n=args.n, k=args.k, save=args.save, load_path=args.load)
