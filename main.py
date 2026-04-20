from pathlib import Path
import sys
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
APP_DIR = BASE_DIR / "gla_chatbot"
SRC_DIR = APP_DIR / "src"
VECTOR_STORE_DIR = APP_DIR / "vector_store"
ENV_FILE = APP_DIR / ".env"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(dotenv_path=ENV_FILE)

from chatbot import GLAChatbot  # noqa: E402
from embedder import index_exists  # noqa: E402


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


app = FastAPI(title="GLA Chatbot Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_chatbot: GLAChatbot | None = None


def get_chatbot() -> GLAChatbot:
    global _chatbot

    if _chatbot is None:
        if not index_exists(str(VECTOR_STORE_DIR)):
            raise RuntimeError(
                f"Vector store not found at '{VECTOR_STORE_DIR}'. Run ingestion first."
            )
        _chatbot = GLAChatbot(vector_store_dir=str(VECTOR_STORE_DIR))

    return _chatbot


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "vector_store_ready": index_exists(str(VECTOR_STORE_DIR)),
        "env_loaded": ENV_FILE.exists(),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        result = get_chatbot().ask_with_sources(message)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat request failed: {exc}") from exc

    return ChatResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
    )


@app.post("/reset")
def reset() -> dict[str, str]:
    try:
        get_chatbot().reset_history()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reset failed: {exc}") from exc

    return {"status": "reset"}
