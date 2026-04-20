"""
retriever.py
User ke query ke liye FAISS index mein se
sabse relevant chunks dhundta hai.
"""

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from embedder import load_index


TOP_K = 4


class GlaRetriever:
    """
    GLA chatbot ka retrieval engine.
    Ek baar banao, baar baar query karo.
    """

    def __init__(self, vector_store_dir: str = "vector_store"):
        print("Loading retriever...")
        self.vectorstore: FAISS = load_index(vector_store_dir)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K}
        )
        print("  [ok] Retriever ready")

    def get_relevant_chunks(self, query: str) -> list[Document]:
        """
        Query ke liye top-K relevant chunks wapas karo.
        """
        if not query.strip():
            return []

        docs = self.retriever.invoke(query)
        return docs

    def get_chunks_with_scores(self, query: str) -> list[tuple[Document, float]]:
        """
        Chunks ke saath similarity score bhi chahiye toh ye use karo.
        """
        return self.vectorstore.similarity_search_with_score(query, k=TOP_K)

    def format_context(self, docs: list[Document]) -> str:
        """
        Retrieved chunks ko ek clean context string mein combine karo.
        """
        if not docs:
            return "No relevant information found."

        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(
                f"[Source {i}: {source}]\n{doc.page_content.strip()}"
            )

        return "\n\n---\n\n".join(parts)

    def retrieve_and_format(self, query: str) -> str:
        """
        Shortcut: query do, formatted context string wapas lo.
        """
        docs = self.get_relevant_chunks(query)
        return self.format_context(docs)


if __name__ == "__main__":
    retriever = GlaRetriever()

    test_queries = [
        "B.Tech admission ke liye eligibility kya hai?",
        "GLA University mein fees kitni hai?",
        "Hostel facility hai kya?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        docs = retriever.get_relevant_chunks(query)
        for i, doc in enumerate(docs, 1):
            print(f"Chunk {i} [{doc.metadata.get('source')}]:")
            print(doc.page_content[:200])
            print()
