import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from src.retriever import get_embedding_model, DOMAINS
from src.chatbot import RAGChatbot

load_dotenv()

# Global state — embedding model and chatbot sessions
embedding_model = None
sessions: dict[str, RAGChatbot] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load embedding model once at startup."""
    global embedding_model
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

    # Create session if it doesn't exist
    session_key = f"{request.session_id}:{request.domain}"
    if session_key not in sessions:
        sessions[session_key] = RAGChatbot(
            domain=request.domain,
            embedding_model=embedding_model
        )

    bot = sessions[session_key]
    result = bot.chat(request.question)

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