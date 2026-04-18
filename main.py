import os
import json
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google.cloud import storage

from src.retriever import get_embedding_model, DOMAINS
from src.chatbot import RAGChatbot

load_dotenv()

# Global state — embedding model and chatbot sessions
embedding_model = None
sessions: dict[str, RAGChatbot] = {}


def download_chroma_from_gcs():
    """Download ChromaDB from GCS to local container filesystem at startup."""
    bucket_name = "ml-portfolio-rag-store-493708"
    gcs_prefix = "chroma_db"

    print(f"Downloading ChromaDB from GCS bucket: {bucket_name}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=gcs_prefix)
    for blob in blobs:
        local_path = Path(blob.name)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        print(f"Downloaded: {blob.name}")

    print("ChromaDB download complete.")


def log_query_to_gcs(session_id: str, domain: str, question: str, answer: str, confident: bool):
    """Append a query log entry to GCS as a JSON line."""
    try:
        client = storage.Client()
        bucket = client.bucket("ml-portfolio-rag-store-493708")

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "domain": domain,
            "question": question,
            "answer": answer,
            "confident": confident
        }

        filename = f"query_logs/{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
        blob = bucket.blob(filename)

        try:
            existing = blob.download_as_text()
        except Exception:
            existing = ""

        updated = existing + json.dumps(log_entry) + "\n"
        blob.upload_from_string(updated)
        print(f"Query logged to GCS: {filename}")

    except Exception as e:
        print(f"ERROR logging to GCS: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model
    download_chroma_from_gcs()
    print("Loading embedding model...")
    embedding_model = get_embedding_model()
    print("Embedding model loaded. API ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Multi-Domain RAG Chatbot",
    description="Query financial, regulatory and legal documents using natural language",
    version="1.0.0",
    lifespan=lifespan
)


# --- Request/Response Models ---

class ChatRequest(BaseModel):
    session_id: str
    domain: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    confident: bool
    sources: list
    session_id: str
    domain: str


class ResetRequest(BaseModel):
    session_id: str


# --- Endpoints ---

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "llama-3.1-8b-instant",
        "embedding_model": "all-MiniLM-L6-v2",
        "available_domains": list(DOMAINS.keys())
    }


@app.get("/domains")
def get_domains():
    return {"domains": DOMAINS}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if request.domain not in DOMAINS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid domain '{request.domain}'. Available: {list(DOMAINS.keys())}"
        )

    session_key = f"{request.session_id}:{request.domain}"
    if session_key not in sessions:
        sessions[session_key] = RAGChatbot(
            domain=request.domain,
            embedding_model=embedding_model
        )

    bot = sessions[session_key]
    result = bot.chat(request.question)

    log_query_to_gcs(
        session_id=request.session_id,
        domain=request.domain,
        question=request.question,
        answer=result["answer"],
        confident=result["confident"]
    )

    return ChatResponse(
        answer=result["answer"],
        confident=result["confident"],
        sources=result["sources"],
        session_id=request.session_id,
        domain=request.domain
    )


@app.post("/reset")
def reset_session(request: ResetRequest):
    """Clear all chatbot sessions for a given session_id."""
    cleared = []
    for domain in DOMAINS:
        key = f"{request.session_id}:{domain}"
        if key in sessions:
            sessions[key].reset()
            cleared.append(domain)
    return {"cleared": cleared, "session_id": request.session_id}

@app.get("/test-log")
def test_log():
    try:
        log_query_to_gcs(
            session_id="debug",
            domain="rbi",
            question="test question",
            answer="test answer",
            confident=True
        )
        return {"status": "log attempted, check GCS"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}