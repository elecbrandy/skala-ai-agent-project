from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from graph.state import AgentState
import os

RAG_RESULT_THRESHOLD = 3

# ── 임베딩 모델: BAAI/bge-m3 ──────────────────────────────────
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model,
)


def build_mmr_retriever(k: int = 5):
    """Dense + MMR Retriever — 평가 결과 최고 MRR(0.73) 달성"""
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3, "lambda_mult": 0.7},
    )


def rag_node(state: AgentState) -> dict:
    intent    = state["parsed_intent"]
    keywords  = intent.get("keywords", [])
    companies = intent.get("companies", [])

    queries: list[str] = list(keywords)
    for kw in keywords:
        for company in companies:
            queries.append(f"{company} {kw} 기술 동향")
            queries.append(f"{company} {kw} TRL")

    if not vectorstore.get().get("documents"):
        return {"rag_results": [], "rag_sufficient": False}

    retriever = build_mmr_retriever()

    results: list[dict] = []
    seen: set[int] = set()
    for query in queries:
        for doc in retriever.invoke(query):
            doc_hash = hash(doc.page_content[:100])
            if doc_hash in seen:
                continue
            seen.add(doc_hash)
            results.append({
                "content": doc.page_content,
                "source":  doc.metadata.get("source", "unknown"),
                "date":    doc.metadata.get("date", ""),
                "company": doc.metadata.get("company", ""),
                "score":   doc.metadata.get("score", 0.0),
            })

    return {
        "rag_results":    results,
        "rag_sufficient": len(results) >= RAG_RESULT_THRESHOLD,
    }
