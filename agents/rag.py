from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from graph.state import AgentState
import os

RAG_RESULT_THRESHOLD = 3  # 최소 결과 수 임계값

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


def build_hybrid_retriever(docs: list[Document], k: int = 5) -> EnsembleRetriever:
    """Dense(MMR) + BM25 Hybrid Retriever"""
    dense = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3, "lambda_mult": 0.7},
    )
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return EnsembleRetriever(retrievers=[dense, bm25], weights=[0.6, 0.4])


def rag_node(state: AgentState) -> dict:
    intent    = state["parsed_intent"]
    keywords  = intent.get("keywords", [])
    companies = intent.get("companies", [])

    # Query Rewriting: 키워드 × 기업명 조합으로 다중 쿼리 생성
    queries: list[str] = list(keywords)
    for kw in keywords:
        for company in companies:
            queries.append(f"{company} {kw} 기술 동향")
            queries.append(f"{company} {kw} TRL")

    raw_docs = vectorstore.get()
    all_docs = [
        Document(page_content=pc, metadata=meta)
        for pc, meta in zip(
            raw_docs.get("documents", []),
            raw_docs.get("metadatas", [{}] * len(raw_docs.get("documents", []))),
        )
    ]

    if not all_docs:
        return {"rag_results": [], "rag_sufficient": False}

    retriever = build_hybrid_retriever(all_docs)

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
