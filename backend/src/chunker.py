"""
chunker.py
Raw text ko chhote-chhote chunks mein todta hai.
LangChain ka RecursiveCharacterTextSplitter use karta hai.
Chunk size: 500 tokens, overlap: 50 tokens
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text: str, source_name: str = "unknown") -> list[Document]:
    """
    Ek bade text string ko Document chunks mein todo.

    Args:
        text: raw extracted text (PDF se aaya hua)
        source_name: PDF ka naam (metadata ke liye)

    Returns:
        List of LangChain Document objects with metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "। ", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": source_name}]
    )

    return chunks


def chunk_all_docs(docs: dict) -> list[Document]:
    """
    Multiple PDFs ke extracted text ko ek saath chunk karo.

    Args:
        docs: {filename: text} dictionary (ocr_loader se aata hai)

    Returns:
        Saare PDFs ke chunks ek list mein
    """
    all_chunks = []

    for filename, text in docs.items():
        if not text.strip():
            print(f"  [warn] Skipping empty: {filename}")
            continue

        chunks = chunk_text(text, source_name=filename)
        all_chunks.extend(chunks)
        print(f"  [ok] {filename}: {len(chunks)} chunks bane")

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


def preview_chunks(chunks: list[Document], n: int = 3):
    """Debug ke liye — pehle n chunks print karo."""
    print(f"\n── First {n} chunks preview ──")
    for i, chunk in enumerate(chunks[:n]):
        print(f"\n[Chunk {i+1}] Source: {chunk.metadata.get('source')}")
        print(f"Length: {len(chunk.page_content)} chars")
        print(chunk.page_content[:200])
        print("...")


if __name__ == "__main__":
    sample = """
    GLA University B.Tech Admission 2024

    Eligibility: 10+2 with Physics, Chemistry, Mathematics.
    Minimum 60% marks required.

    Fee Structure:
    B. Tech CSE (Core) | 2.0L | 2.08L | 2.16L | 2.24L
    B. Tech CSE (AIML) | 2.35L | 2.43L | 2.51L | 2.59L
    B. Tech CSE (Honors) | 2.40L | 2.45L | 2.50L | 2.55L
    """

    chunks = chunk_text(sample, source_name="test.pdf")
    preview_chunks(chunks)
