from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
    from .ingestion import (
        build_allowed_domains,
        crawl_site,
        is_supported_upload,
        save_uploaded_bytes,
    )
    from .rag import BrochureRAG, FALLBACK_ANSWER
except ImportError:
    from ingestion import build_allowed_domains, crawl_site, is_supported_upload, save_uploaded_bytes
    from rag import BrochureRAG, FALLBACK_ANSWER

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's question about the indexed GLA sources")


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)


class CrawlRequest(BaseModel):
    url: str = Field(..., min_length=1, description="Website URL to crawl and index")
    max_pages: int = Field(default=12, ge=1, le=50, description="Maximum number of pages to crawl")


@lru_cache(maxsize=1)
def get_rag_service() -> BrochureRAG:
    return BrochureRAG()


def refresh_rag_service() -> BrochureRAG:
    get_rag_service.cache_clear()
    service = get_rag_service()
    app.state.startup_error = None
    return service


def get_official_site_url() -> str:
    return os.getenv("OFFICIAL_SITE_URL", "https://www.gla.ac.in/info/common/").strip()


def get_official_site_max_pages() -> int:
    try:
        return max(1, min(50, int(os.getenv("OFFICIAL_SITE_MAX_PAGES", "12"))))
    except ValueError:
        return 12


def sync_official_site(max_pages: int | None = None) -> dict[str, Any]:
    official_url = get_official_site_url()
    allowed_domains = build_allowed_domains(
        official_url,
        os.getenv("CRAWL_ALLOWED_DOMAINS"),
    )
    crawl_result = crawl_site(
        start_url=official_url,
        max_pages=max_pages or get_official_site_max_pages(),
        allowed_domains=allowed_domains,
    )
    refresh_rag_service()
    return crawl_result


def get_allowed_origins() -> list[str]:
    configured_origins = os.getenv("CORS_ORIGINS", "")
    if configured_origins.strip():
        return [origin.strip() for origin in configured_origins.split(",") if origin.strip()]

    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]


app = FastAPI(
    title="GLA University Knowledge Chatbot",
    version="1.1.0",
    description="A FastAPI RAG service that answers strictly from indexed GLA source content.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def warm_up_index() -> None:
    try:
        get_rag_service()
        app.state.startup_error = None
    except Exception as exc:
        app.state.startup_error = str(exc)


@app.get("/health")
def health_check() -> dict[str, str]:
    if getattr(app.state, "startup_error", None):
        return {"status": "error", "detail": app.state.startup_error}
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "GLA University Knowledge Chatbot API is running.",
        "health": "/health",
        "docs": "/docs",
        "chat": "/chat",
        "upload": "/ingest/upload",
        "crawl": "/ingest/web",
        "official_sync": "/ingest/official",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if getattr(app.state, "startup_error", None):
        raise HTTPException(status_code=500, detail=app.state.startup_error)

    question = request.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = get_rag_service().ask(question)
        return ChatResponse(
            answer=result.get("answer", FALLBACK_ANSWER),
            sources=result.get("sources", []),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ingest/upload")
async def upload_sources(files: list[UploadFile] = File(...)) -> dict[str, object]:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one file.")

    saved_files: list[str] = []

    for upload in files:
        filename = upload.filename or ""
        if not is_supported_upload(filename):
            raise HTTPException(
                status_code=400,
                detail="Only PDF, TXT, and MD files are supported for upload.",
            )

        content = await upload.read()
        await upload.close()
        if not content:
            continue

        destination = save_uploaded_bytes(filename, content)
        saved_files.append(str(destination.name))

    if not saved_files:
        raise HTTPException(status_code=400, detail="The uploaded files were empty.")

    try:
        refresh_rag_service()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload saved, but reindex failed: {exc}") from exc

    return {
        "message": f"Indexed {len(saved_files)} uploaded file(s).",
        "saved_files": saved_files,
    }


@app.post("/ingest/web")
def ingest_website(request: CrawlRequest) -> dict[str, object]:
    crawl_url = request.url.strip()
    if not crawl_url:
        raise HTTPException(status_code=400, detail="Website URL cannot be empty.")

    allowed_domains = build_allowed_domains(
        crawl_url,
        os.getenv("CRAWL_ALLOWED_DOMAINS"),
    )

    try:
        crawl_result = crawl_site(
            start_url=crawl_url,
            max_pages=request.max_pages,
            allowed_domains=allowed_domains,
        )
        refresh_rag_service()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "message": f"Indexed {crawl_result['pages_indexed']} page(s) from the website crawl.",
        **crawl_result,
    }


@app.post("/ingest/official")
def ingest_official_site() -> dict[str, Any]:
    try:
        crawl_result = sync_official_site()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "message": f"Indexed {crawl_result['pages_indexed']} page(s) from the official GLA website.",
        **crawl_result,
    }
