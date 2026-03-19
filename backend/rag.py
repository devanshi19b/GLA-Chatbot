from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
PDF_PATH = PROJECT_ROOT / "data" / "brochure.pdf"
INDEX_DIR = BASE_DIR / "faiss_index"
FALLBACK_ANSWER = "Information not available in the brochure"

load_dotenv(BASE_DIR / ".env")


class BrochureRAG:
    def __init__(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is missing. Add it to backend/.env before starting the API.")

        if not PDF_PATH.exists():
            raise FileNotFoundError(f"Brochure PDF not found at {PDF_PATH}")

        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.top_k = int(os.getenv("TOP_K", "4"))
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4")
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.vector_store = self._load_or_build_vector_store()
        self.llm = ChatOpenAI(model=self.chat_model, temperature=0)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a brochure-grounded assistant for GLA University. "
                        "Answer ONLY from the supplied brochure context. "
                        "Do not use outside knowledge, do not infer facts that are not clearly supported, "
                        "and keep the answer short and accurate. "
                        f"If the context does not contain the answer, reply exactly with: {FALLBACK_ANSWER}"
                    ),
                ),
                (
                    "human",
                    (
                        "Question:\n{question}\n\n"
                        "Brochure context:\n{context}\n\n"
                        "Return only the final answer."
                    ),
                ),
            ]
        )

    def _load_or_build_vector_store(self) -> FAISS:
        if INDEX_DIR.exists():
            return FAISS.load_local(
                str(INDEX_DIR),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

        loader = PyPDFLoader(str(PDF_PATH))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        vector_store = FAISS.from_documents(chunks, self.embeddings)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(INDEX_DIR))
        return vector_store

    def _retrieve_documents(self, question: str) -> list[tuple[Any, float | None]]:
        try:
            matches = self.vector_store.similarity_search_with_relevance_scores(
                question,
                k=self.top_k,
            )
        except Exception:
            matches = [
                (document, None)
                for document in self.vector_store.similarity_search(question, k=self.top_k)
            ]

        cleaned_matches: list[tuple[Any, float | None]] = []
        for document, score in matches:
            if document.page_content.strip():
                cleaned_matches.append((document, score))
        return cleaned_matches

    def _build_context(self, matches: list[tuple[Any, float | None]]) -> tuple[str, list[str]]:
        context_parts: list[str] = []
        sources: list[str] = []

        for index, (document, _) in enumerate(matches, start=1):
            page_number = document.metadata.get("page")
            page_label = f"Page {page_number + 1}" if isinstance(page_number, int) else "Page unknown"
            sources.append(page_label)
            context_parts.append(f"[Chunk {index} | {page_label}]\n{document.page_content.strip()}")

        unique_sources = list(dict.fromkeys(sources))
        return "\n\n".join(context_parts), unique_sources

    def ask(self, question: str) -> dict[str, Any]:
        normalized_question = question.strip()
        if not normalized_question:
            return {"answer": FALLBACK_ANSWER, "sources": []}

        matches = self._retrieve_documents(normalized_question)
        if not matches:
            return {"answer": FALLBACK_ANSWER, "sources": []}

        context, sources = self._build_context(matches)
        chain = self.prompt | self.llm
        response = chain.invoke({"question": normalized_question, "context": context})
        answer = getattr(response, "content", "").strip() or FALLBACK_ANSWER

        if answer != FALLBACK_ANSWER and FALLBACK_ANSWER.lower() in answer.lower():
            answer = FALLBACK_ANSWER

        return {"answer": answer, "sources": sources}
