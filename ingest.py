"""
data/ 폴더의 문서를 ChromaDB에 적재하는 스크립트

지원 형식 : PDF, TXT, MD
사용법    : python ingest.py
           python ingest.py --data_dir ./data --reset
"""

import os
import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── rag.py 와 동일한 설정 ──────────────────────────────────────
CHROMA_DIR  = os.getenv("CHROMA_DIR", "./chroma_db")
DATA_DIR    = os.getenv("DATA_DIR",   "./data")
EMBED_MODEL = "BAAI/bge-m3"

CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_documents(data_dir: str) -> list:
    docs = []
    data_path = Path(data_dir)
    files = [f for f in data_path.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        print(f"[ingest] '{data_dir}' 에 지원 파일이 없습니다 (PDF, TXT, MD)")
        return docs

    for file in files:
        try:
            if file.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file))
            else:
                loader = TextLoader(str(file), encoding="utf-8")

            loaded = loader.load()

            # 파일명 기반 메타데이터 보강
            for doc in loaded:
                doc.metadata.setdefault("source", file.name)
                doc.metadata.setdefault("file_path", str(file))
                doc.metadata.setdefault("date", "")
                doc.metadata.setdefault("company", _extract_company(file.name))

            docs.extend(loaded)
            print(f"  [로드] {file.name}  ({len(loaded)}페이지/청크)")

        except Exception as e:
            print(f"  [오류] {file.name}: {e}")

    return docs


def _extract_company(filename: str) -> str:
    """
    파일명에서 회사명을 추출합니다.
    예: samsung_hbm4_2025.pdf → 삼성전자
    파일명에 회사 키워드가 없으면 빈 문자열 반환.
    """
    name_lower = filename.lower()
    mapping = {
        "samsung":   "삼성전자",
        "skhynix":   "SK하이닉스",
        "hynix":     "SK하이닉스",
        "micron":    "Micron",
        "tsmc":      "TSMC",
        "intel":     "Intel",
        "amd":       "AMD",
        "nvidia":    "NVIDIA",
    }
    for key, company in mapping.items():
        if key in name_lower:
            return company
    return ""


def split_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"\n[ingest] 총 {len(docs)}개 문서 → {len(chunks)}개 청크 분할 완료")
    return chunks


def build_vectorstore(chunks: list, reset: bool = False) -> Chroma:
    print(f"[ingest] 임베딩 모델 로드 중: {EMBED_MODEL}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if reset and Path(CHROMA_DIR).exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print(f"[ingest] 기존 ChromaDB 삭제: {CHROMA_DIR}")

    print(f"[ingest] ChromaDB 적재 중 → {CHROMA_DIR}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
    )
    print(f"[ingest] 적재 완료 — {len(chunks)}개 청크 저장됨")
    return vectorstore


def main(data_dir: str, reset: bool) -> None:
    print("=" * 50)
    print("  Tech Strategy Agent — Document Ingestion")
    print("=" * 50)
    print(f"  데이터 경로 : {data_dir}")
    print(f"  ChromaDB   : {CHROMA_DIR}")
    print(f"  초기화 여부 : {'Yes (기존 DB 삭제)' if reset else 'No (기존 DB에 추가)'}")
    print("=" * 50 + "\n")

    docs   = load_documents(data_dir)
    if not docs:
        return

    chunks = split_documents(docs)
    build_vectorstore(chunks, reset=reset)

    print("\n✅ 완료 — 이제 main.py 를 실행하세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Ingestion for Tech Strategy Agent")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR,  help="문서 폴더 경로")
    parser.add_argument("--reset",    action="store_true",          help="기존 ChromaDB 초기화 후 재적재")
    args = parser.parse_args()

    main(args.data_dir, args.reset)
