from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .rag import BrochureRAG, FALLBACK_ANSWER
except ImportError:
    from rag import BrochureRAG, FALLBACK_ANSWER


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's question about the brochure")


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)


@lru_cache(maxsize=1)
def get_rag_service() -> BrochureRAG:
    return BrochureRAG()


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
    title="GLA University Brochure Chatbot",
    version="1.0.0",
    description="A FastAPI RAG service that answers strictly from the university brochure.",
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
        "message": "GLA University Brochure Chatbot API is running.",
        "health": "/health",
        "docs": "/docs",
        "chat": "/chat",
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
