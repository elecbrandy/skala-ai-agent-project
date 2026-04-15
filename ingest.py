"""
data/ 폴더의 문서를 ChromaDB에 적재하는 스크립트

지원 형식 : PDF, TXT, MD
사용법    : python ingest.py
           python ingest.py --data_dir ./data --reset
"""

import os
import json
import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── rag.py 와 동일한 설정 ──────────────────────────────────────
CHROMA_DIR    = os.getenv("CHROMA_DIR", "./chroma_db")
DATA_DIR      = os.getenv("DATA_DIR",   "./data")
EMBED_MODEL   = "BAAI/bge-m3"
MANIFEST_PATH = os.path.join(CHROMA_DIR, ".manifest.json")

CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


# ── Manifest 관리 ──────────────────────────────────────────────
def _file_signature(path: Path) -> dict:
    stat = path.stat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def _load_manifest() -> dict:
    if Path(MANIFEST_PATH).exists():
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_manifest(files: list[Path]) -> None:
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    manifest = {str(f): _file_signature(f) for f in files}
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def needs_sync(data_dir: str) -> tuple[bool, list[Path]]:
    """
    data/ 파일 목록과 manifest를 비교하여 동기화 필요 여부를 반환합니다.
    Returns: (sync_needed: bool, files: list[Path])
    """
    files = [
        f for f in Path(data_dir).rglob("*")
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        return False, []

    # ChromaDB 자체가 없으면 무조건 동기화 필요
    if not Path(CHROMA_DIR).exists():
        return True, files

    manifest = _load_manifest()
    if not manifest:
        return True, files

    current = {str(f): _file_signature(f) for f in files}
    if current == manifest:
        return False, files

    # 변경·추가된 파일 출력
    added   = [k for k in current if k not in manifest]
    changed = [k for k in current if k in manifest and current[k] != manifest[k]]
    removed = [k for k in manifest if k not in current]
    if added:
        print(f"[sync] 신규 파일: {', '.join(Path(k).name for k in added)}")
    if changed:
        print(f"[sync] 변경 파일: {', '.join(Path(k).name for k in changed)}")
    if removed:
        print(f"[sync] 삭제 파일: {', '.join(Path(k).name for k in removed)}")

    return True, files


# ── 문서 로드 ─────────────────────────────────────────────────
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

            for doc in loaded:
                doc.metadata.setdefault("source",    file.name)
                doc.metadata.setdefault("file_path", str(file))
                doc.metadata.setdefault("date",      "")
                doc.metadata.setdefault("company",   _extract_company(file.name))

            docs.extend(loaded)
            print(f"  [로드] {file.name}  ({len(loaded)}페이지/청크)")

        except Exception as e:
            print(f"  [오류] {file.name}: {e}")

    return docs


def _extract_company(filename: str) -> str:
    name_lower = filename.lower()
    mapping = {
        "samsung":  "삼성전자",
        "skhynix":  "SK하이닉스",
        "hynix":    "SK하이닉스",
        "micron":   "Micron",
        "tsmc":     "TSMC",
        "intel":    "Intel",
        "amd":      "AMD",
        "nvidia":   "NVIDIA",
    }
    for key, company in mapping.items():
        if key in name_lower:
            return company
    return ""


# ── 청크 분할 ─────────────────────────────────────────────────
def split_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"\n[ingest] 총 {len(docs)}개 문서 → {len(chunks)}개 청크 분할 완료")
    return chunks


# ── ChromaDB 적재 ─────────────────────────────────────────────
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


# ── 진입점 ───────────────────────────────────────────────────
def main(data_dir: str, reset: bool) -> None:
    print("=" * 50)
    print("  Tech Strategy Agent — Document Ingestion")
    print("=" * 50)
    print(f"  데이터 경로 : {data_dir}")
    print(f"  ChromaDB   : {CHROMA_DIR}")
    print(f"  초기화 여부 : {'Yes (기존 DB 삭제)' if reset else 'No (기존 DB에 추가)'}")
    print("=" * 50 + "\n")

    sync_needed, files = needs_sync(data_dir)
    if not sync_needed and not reset:
        print("[ingest] ChromaDB 최신 상태 — 동기화 건너뜀\n")
        return

    docs = load_documents(data_dir)
    if not docs:
        return

    chunks = split_documents(docs)
    build_vectorstore(chunks, reset=reset)
    _save_manifest(files)

    print("\n✅ 완료 — 이제 main.py 를 실행하세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Ingestion for Tech Strategy Agent")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR,  help="문서 폴더 경로")
    parser.add_argument("--reset",    action="store_true",          help="기존 ChromaDB 초기화 후 재적재")
    args = parser.parse_args()

    main(args.data_dir, args.reset)
