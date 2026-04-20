"""
scripts/ingest.py
EK BAAR CHALAO — PDF → Chunks → Embeddings → FAISS Index

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --force
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_loader import load_all_pdfs
from chunker import chunk_all_docs
from embedder import create_and_save_index, index_exists


def ingest(data_dir: str = "data", vector_store_dir: str = "vector_store", force: bool = False):
    start_time = time.time()

    print("\n" + "=" * 55)
    print("  GLA Chatbot - Ingestion Pipeline")
    print("=" * 55)

    if index_exists(vector_store_dir) and not force:
        print(f"\n[warn] Vector store already exists: '{vector_store_dir}/'")
        print("   Dobara banane ke liye: python scripts/ingest.py --force")
        print("   Chatbot chalao: streamlit run app.py")
        return

    # Step 1: PDFs load + extract
    print(f"\n[1/3] PDFs load karo from '{data_dir}/'...")
    try:
        docs = load_all_pdfs(data_dir)
    except FileNotFoundError as e:
        print(f"\n[error] {e}")
        print(f"  Solution: '{data_dir}/' folder banao aur PDFs daalo.")
        sys.exit(1)

    if not docs:
        print("[error] Koi text extract nahi hua. PDFs check karo.")
        sys.exit(1)

    total_chars = sum(len(t) for t in docs.values())
    print(f"  [ok] {len(docs)} PDFs loaded, {total_chars:,} total characters")

    # Step 2: Chunking
    print(f"\n[2/3] Text chunking...")
    chunks = chunk_all_docs(docs)

    if not chunks:
        print("[error] Chunks nahi bane.")
        sys.exit(1)

    # Step 3: Embed + Save
    print(f"\n[3/3] Embedding + FAISS index save karo...")
    vectorstore = create_and_save_index(chunks, save_dir=vector_store_dir)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 55}")
    print(f"  [ok] Ingestion complete in {elapsed:.1f} seconds!")
    print(f"  {len(docs)} PDFs -> {len(chunks)} chunks -> vector_store/")
    print(f"\n  Chatbot chalao:")
    print(f"  streamlit run app.py")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLA Chatbot Ingestion")
    parser.add_argument("--data-dir", default="data", help="PDF folder path")
    parser.add_argument("--vector-store", default="vector_store", help="FAISS save path")
    parser.add_argument("--force", action="store_true", help="Existing index overwrite karo")
    args = parser.parse_args()

    ingest(
        data_dir=args.data_dir,
        vector_store_dir=args.vector_store,
        force=args.force
    )
