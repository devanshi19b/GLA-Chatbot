from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = BASE_DIR / "faiss_index"
INDEX_MANIFEST_PATH = BASE_DIR / "faiss_index_manifest.json"
FALLBACK_ANSWER = "Information not available in the brochure"

load_dotenv(BASE_DIR / ".env")


class BrochureRAG:
    def __init__(self) -> None:
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key is missing. Add GEMINI_API_KEY or GOOGLE_API_KEY to backend/.env before starting the API."
            )

        self.chunk_size = int(os.getenv("CHUNK_SIZE", "2500"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
        self.top_k = int(os.getenv("TOP_K", "4"))
        self.chat_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
        self.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
        self.pdf_paths = self._get_pdf_paths()

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.embedding_model,
            google_api_key=self.api_key,
        )
        self.vector_store = self._load_or_build_vector_store()
        self.llm = ChatGoogleGenerativeAI(
            model=self.chat_model,
            temperature=0,
            google_api_key=self.api_key,
        )
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

    def _get_pdf_paths(self) -> list[Path]:
        pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
        if not pdf_paths:
            raise FileNotFoundError(f"No brochure PDFs found in {DATA_DIR}")
        return pdf_paths

    def _build_index_manifest(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "files": [
                {
                    "name": pdf_path.name,
                    "size": pdf_path.stat().st_size,
                    "mtime_ns": pdf_path.stat().st_mtime_ns,
                }
                for pdf_path in self.pdf_paths
            ],
        }

    def _index_is_current(self, manifest: dict[str, Any]) -> bool:
        if not INDEX_DIR.exists() or not INDEX_MANIFEST_PATH.exists():
            return False

        try:
            saved_manifest = json.loads(INDEX_MANIFEST_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            return False

        return saved_manifest == manifest

    def _load_documents(self) -> list[Any]:
        documents: list[Any] = []

        for pdf_path in self.pdf_paths:
            loaded_documents = PyPDFLoader(str(pdf_path)).load()
            for document in loaded_documents:
                document.metadata["source_file"] = pdf_path.name
                if document.page_content.strip():
                    documents.append(document)

        return documents

    def _load_or_build_vector_store(self) -> FAISS:
        manifest = self._build_index_manifest()
        if self._index_is_current(manifest):
            return FAISS.load_local(
                str(INDEX_DIR),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

        documents = self._load_documents()
        if not documents:
            raise ValueError(
                "No extractable brochure text found in data/. Add text-based PDFs or OCR the scanned brochures first."
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        vector_store = FAISS.from_documents(chunks, self.embeddings)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(INDEX_DIR))
        INDEX_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
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
            source_file = document.metadata.get("source_file", "unknown.pdf")
            source_label = f"{source_file} | {page_label}"
            sources.append(source_label)
            context_parts.append(f"[Chunk {index} | {source_label}]\n{document.page_content.strip()}")

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
