from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from .ingestion import DATA_DIR, ensure_data_directories
except ImportError:
    from ingestion import DATA_DIR, ensure_data_directories

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
INDEX_DIR = BASE_DIR / "faiss_index"
INDEX_MANIFEST_PATH = BASE_DIR / "faiss_index_manifest.json"
FALLBACK_ANSWER = "Information not available in the indexed sources"
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}
QUERY_NOISE_TOKENS = {
    "according",
    "assistant",
    "chatbot",
    "count",
    "currently",
    "gla",
    "mentioned",
    "official",
    "site",
    "student",
    "tell",
    "university",
    "web",
    "website",
}
NOISE_PATTERNS = (
    "information brochure",
    "and many more",
    "nurturing excellence",
    "inspiring futures",
)

load_dotenv(BASE_DIR / ".env")


class BrochureRAG:
    def __init__(self) -> None:
        ensure_data_directories()
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key is missing. Add GROQ_API_KEY to backend/.env before starting the API."
            )

        self.chunk_size = int(os.getenv("CHUNK_SIZE", "2500"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
        self.top_k = int(os.getenv("TOP_K", "4"))
        self.chat_model = os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")
        self.embedding_model = os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.source_paths = self._get_source_paths()

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
        )
        self.vector_store = self._load_or_build_vector_store()
        self.llm = ChatGroq(
            model=self.chat_model,
            temperature=0,
            groq_api_key=self.api_key,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a source-grounded assistant for GLA University. "
                        "Answer ONLY from the supplied indexed source context. "
                        "Do not use outside knowledge, do not infer facts that are not clearly supported, "
                        "and keep the answer short and accurate. "
                        f"If the context does not contain the answer, reply exactly with: {FALLBACK_ANSWER}"
                    ),
                ),
                (
                    "human",
                    (
                        "Question:\n{question}\n\n"
                        "Indexed source context:\n{context}\n\n"
                        "Return only the final answer."
                    ),
                ),
            ]
        )

    def _get_source_paths(self) -> list[Path]:
        source_paths = [
            path
            for path in sorted(DATA_DIR.rglob("*"))
            if path.is_file() and path.suffix.lower() in {".json", ".md", ".pdf", ".txt"}
        ]
        if not source_paths:
            raise FileNotFoundError(f"No supported source files found in {DATA_DIR}")
        return source_paths

    def _build_index_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "files": [
                {
                    "name": str(source_path.relative_to(DATA_DIR)),
                    "size": source_path.stat().st_size,
                    "mtime_ns": source_path.stat().st_mtime_ns,
                }
                for source_path in self.source_paths
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

        for source_path in self.source_paths:
            suffix = source_path.suffix.lower()

            if suffix == ".pdf":
                loaded_documents = PyPDFLoader(str(source_path)).load()
                for document in loaded_documents:
                    document.metadata["source_file"] = source_path.name
                    document.metadata["source_path"] = str(source_path.relative_to(DATA_DIR))
                    if document.page_content.strip():
                        documents.append(document)
                continue

            if suffix == ".json":
                try:
                    payload = json.loads(source_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue

                content = str(payload.get("content", "")).strip()
                if not content:
                    continue

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source_file": source_path.name,
                            "source_path": str(source_path.relative_to(DATA_DIR)),
                            "source_url": payload.get("url"),
                            "title": payload.get("title"),
                            "page": None,
                        },
                    )
                )
                continue

            try:
                content = source_path.read_text(encoding="utf-8")
            except OSError:
                continue

            if content.strip():
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source_file": source_path.name,
                            "source_path": str(source_path.relative_to(DATA_DIR)),
                            "page": None,
                        },
                    )
                )

        return documents

    def _format_source_label(self, metadata: dict[str, Any]) -> str:
        source_url = metadata.get("source_url")
        if source_url:
            return str(source_url)

        source_file = metadata.get("source_file", "unknown-source")
        page_number = metadata.get("page")
        if isinstance(page_number, int):
            return f"{source_file} | Page {page_number + 1}"

        return str(source_file)

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()

    def _tokenize(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2 and token not in STOPWORDS
        }

    def _token_variants(self, token: str) -> set[str]:
        variants = {token}

        if token.endswith("ies") and len(token) > 4:
            variants.add(f"{token[:-3]}y")
        if token.endswith("es") and len(token) > 4:
            variants.add(token[:-2])
        if token.endswith("s") and len(token) > 3:
            variants.add(token[:-1])

        return {variant for variant in variants if len(variant) > 2}

    def _looks_like_noise(self, text: str) -> bool:
        normalized_text = self._normalize_text(text)
        lowercase_text = normalized_text.lower()

        if len(normalized_text) < 80:
            return True

        unique_tokens = self._tokenize(normalized_text)
        if len(unique_tokens) < 8:
            return True

        uppercase_letters = sum(1 for character in normalized_text if character.isalpha() and character.isupper())
        total_letters = sum(1 for character in normalized_text if character.isalpha())
        uppercase_ratio = uppercase_letters / total_letters if total_letters else 0

        contains_noise_phrase = any(pattern in lowercase_text for pattern in NOISE_PATTERNS)
        return contains_noise_phrase and uppercase_ratio > 0.45 and len(unique_tokens) < 35

    def _question_prefers_website(self, question: str) -> bool:
        normalized_question = self._normalize_text(question).lower()
        return any(keyword in normalized_question for keyword in {"official", "website", "site", "web"})

    def _score_match(self, question: str, document: Any, score: float | None) -> float:
        normalized_question = self._normalize_text(question)
        normalized_content = self._normalize_text(document.page_content)
        title_text = self._normalize_text(str(document.metadata.get("title", "")))
        prefers_website = self._question_prefers_website(question)

        question_tokens = self._tokenize(normalized_question)
        content_tokens = self._tokenize(normalized_content)
        title_tokens = self._tokenize(title_text)

        overlap_count = len(question_tokens & content_tokens)
        overlap_ratio = overlap_count / len(question_tokens) if question_tokens else 0.0
        title_overlap_count = len(question_tokens & title_tokens)
        title_overlap_ratio = title_overlap_count / len(question_tokens) if question_tokens else 0.0

        semantic_score = max(float(score), 0.0) if score is not None else 0.0
        noise_penalty = 0.35 if self._looks_like_noise(normalized_content) else 0.0
        exact_phrase_bonus = 0.15 if normalized_question.lower() in normalized_content.lower() else 0.0
        website_bonus = 0.0
        website_penalty = 0.0
        if document.metadata.get("source_url") and prefers_website:
            website_bonus = 0.38
        if prefers_website and not document.metadata.get("source_url"):
            website_penalty = 0.18

        return (
            semantic_score
            + (overlap_ratio * 0.8)
            + (title_overlap_ratio * 0.35)
            + exact_phrase_bonus
            + website_bonus
            - website_penalty
            - noise_penalty
        )

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
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index

        vector_store = FAISS.from_documents(chunks, self.embeddings)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(INDEX_DIR))
        INDEX_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
        return vector_store

    def _retrieve_documents(self, question: str) -> list[tuple[Any, float | None]]:
        candidate_count = max(self.top_k * 4, 8)

        try:
            matches = self.vector_store.similarity_search_with_relevance_scores(
                question,
                k=candidate_count,
            )
        except Exception:
            matches = [
                (document, None)
                for document in self.vector_store.similarity_search(question, k=candidate_count)
            ]

        rescored_matches: list[tuple[Any, float | None, float]] = []
        for document, score in matches:
            if document.page_content.strip():
                rescored_matches.append((document, score, self._score_match(question, document, score)))

        rescored_matches.sort(key=lambda item: item[2], reverse=True)

        cleaned_matches: list[tuple[Any, float | None]] = []
        seen_sources: set[tuple[str, Any]] = set()

        for document, score, _ in rescored_matches:
            source_key = (
                document.metadata.get("source_url") or document.metadata.get("source_file", "unknown-source"),
                document.metadata.get("page"),
                document.metadata.get("chunk_index"),
            )
            if source_key in seen_sources:
                continue

            seen_sources.add(source_key)
            cleaned_matches.append((document, score))

            if len(cleaned_matches) >= self.top_k:
                break

        return cleaned_matches

    def _retrieve_website_documents(self, question: str) -> list[tuple[Any, float | None]]:
        docstore_entries = getattr(getattr(self.vector_store, "docstore", None), "_dict", {})
        if not docstore_entries:
            return []

        normalized_question = self._normalize_text(question)
        lowercase_question = normalized_question.lower()
        question_tokens = self._tokenize(normalized_question)
        subject_tokens = {
            token
            for token in question_tokens
            if token not in QUERY_NOISE_TOKENS and token not in {"many", "number"}
        }
        subject_variants = {
            variant
            for token in subject_tokens
            for variant in self._token_variants(token)
        }
        wants_number = any(phrase in lowercase_question for phrase in {"how many", "number of", "count of"})

        rescored_matches: list[tuple[Any, float]] = []
        for document in docstore_entries.values():
            if not document.metadata.get("source_url"):
                continue

            content = self._normalize_text(document.page_content)
            content_tokens = self._tokenize(content)
            if not content_tokens:
                continue

            title_tokens = self._tokenize(str(document.metadata.get("title", "")))
            overlap_count = len(question_tokens & content_tokens)
            subject_overlap = len(subject_variants & content_tokens)
            title_overlap = len(subject_variants & title_tokens)

            score = (
                (overlap_count * 0.35)
                + (subject_overlap * 1.25)
                + (title_overlap * 0.5)
                + (0.24 if wants_number and re.search(r"\d", content) else 0.0)
                + (0.16 if any(variant in content.lower() for variant in subject_variants) else 0.0)
            )
            rescored_matches.append((document, score))

        rescored_matches.sort(key=lambda item: item[1], reverse=True)
        return [(document, None) for document, score in rescored_matches[: self.top_k] if score > 0]

    def _build_context(self, matches: list[tuple[Any, float | None]]) -> tuple[str, list[str]]:
        context_parts: list[str] = []
        sources: list[str] = []

        for index, (document, _) in enumerate(matches, start=1):
            source_label = self._format_source_label(document.metadata)
            sources.append(source_label)
            context_parts.append(f"[Chunk {index} | {source_label}]\n{document.page_content.strip()}")

        unique_sources = list(dict.fromkeys(sources))
        return "\n\n".join(context_parts), unique_sources

    def _extract_direct_answer(self, question: str, matches: list[tuple[Any, float | None]]) -> str | None:
        normalized_question = self._normalize_text(question)
        lowercase_question = normalized_question.lower()
        question_tokens = self._tokenize(normalized_question)
        subject_tokens = {
            token
            for token in question_tokens
            if token not in QUERY_NOISE_TOKENS and token not in {"many", "number"}
        }
        subject_variants = {
            variant
            for token in subject_tokens
            for variant in self._token_variants(token)
        }

        if "title" in lowercase_question:
            for document, _ in matches:
                title = str(document.metadata.get("title", "")).strip()
                if title:
                    return title

        best_candidate: str | None = None
        best_score = 0.0
        wants_number = any(phrase in lowercase_question for phrase in {"how many", "number of", "count of"})

        for document, _ in matches:
            segments = re.split(r"(?<=[.!?])\s+|\n+", document.page_content)
            for segment in segments:
                candidate = self._normalize_text(segment)
                if len(candidate) < 20:
                    continue

                candidate_tokens = self._tokenize(candidate)
                if not candidate_tokens:
                    continue

                overlap_ratio = len(question_tokens & candidate_tokens) / len(question_tokens) if question_tokens else 0.0
                subject_overlap = len(subject_variants & candidate_tokens)
                subject_ratio = subject_overlap / len(subject_variants) if subject_variants else 0.0
                digit_bonus = 0.24 if wants_number and re.search(r"\d", candidate) else 0.0
                numeric_subject_bonus = (
                    0.4
                    if wants_number and subject_overlap and re.search(r"\d", candidate)
                    else 0.0
                )
                concise_bonus = 0.08 if len(candidate) <= 110 else 0.0
                intro_penalty = (
                    0.22
                    if candidate.lower().startswith(
                        ("about gla", "located in", "we nurture", "gla has", "copyright")
                    )
                    else 0.0
                )
                website_bonus = 0.12 if document.metadata.get("source_url") else 0.0
                score = (
                    (overlap_ratio * 0.55)
                    + (subject_ratio * 0.95)
                    + digit_bonus
                    + numeric_subject_bonus
                    + concise_bonus
                    + website_bonus
                    - intro_penalty
                )

                if score > best_score:
                    best_score = score
                    best_candidate = candidate

        if wants_number and best_score >= 0.42:
            return best_candidate

        if best_score >= 0.55:
            return best_candidate

        return None

    def ask(self, question: str) -> dict[str, Any]:
        normalized_question = question.strip()
        if not normalized_question:
            return {"answer": FALLBACK_ANSWER, "sources": []}

        if self._question_prefers_website(normalized_question):
            website_matches = self._retrieve_website_documents(normalized_question)
            if website_matches:
                matches = website_matches
            else:
                matches = self._retrieve_documents(normalized_question)
        else:
            matches = self._retrieve_documents(normalized_question)
        if not matches:
            return {"answer": FALLBACK_ANSWER, "sources": []}

        context, sources = self._build_context(matches)
        direct_answer = self._extract_direct_answer(normalized_question, matches)
        if direct_answer:
            return {"answer": direct_answer, "sources": sources}

        chain = self.prompt | self.llm
        response = chain.invoke({"question": normalized_question, "context": context})
        answer = getattr(response, "content", "").strip() or FALLBACK_ANSWER

        if answer != FALLBACK_ANSWER and FALLBACK_ANSWER.lower() in answer.lower():
            answer = FALLBACK_ANSWER

        return {"answer": answer, "sources": sources}
