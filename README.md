# DocuMind — Multi-Domain RAG Chatbot

A production-grade Retrieval Augmented Generation (RAG) system that enables natural language querying across financial, regulatory, and legal documents. Built to demonstrate real-world ML engineering patterns including hybrid search, multi-turn memory, confidence thresholding, and containerised deployment.

---

## 🎯 Business Problem

Analysts and compliance teams spend significant time manually searching through dense documents — annual reports, regulatory filings, legal acts — to answer specific questions. This system reduces that from 30+ minutes of manual PDF searching to under 30 seconds of natural language querying.

---

## 📚 Supported Domains

| Domain | Document | Use Case |
|--------|----------|----------|
| `zomato` | Zomato Annual Report 2023 | Financial analysis, business performance, strategy |
| `rbi` | RBI Monetary Policy Report Oct 2024 | Inflation outlook, interest rates, economic indicators |
| `dpdp` | Digital Personal Data Protection Act 2023 | Data privacy compliance, legal obligations, penalties |

---

## 🏗️ Architecture
User Query
↓
Streamlit Frontend (app.py)
↓ HTTP POST /chat
FastAPI Backend (main.py)
↓
RAGChatbot (src/chatbot.py)
↓
Hybrid Retriever (src/retriever.py)
├── Vector Search (ChromaDB + all-MiniLM-L6-v2)
└── BM25 Keyword Search (rank-bm25)
↓ Reciprocal Rank Fusion
Top-k Chunks + Confidence Check
↓ (if confident)
LLM Generation (Groq — LLaMA 3.1 8B)
↓
Answer with Page Citations

---

## ☁️ Live Deployment

**Live API:** https://multi-domain-rag-api-332613889772.asia-south1.run.app

```bash
# Health check
curl https://multi-domain-rag-api-332613889772.asia-south1.run.app/health

# Example query
curl -X POST https://multi-domain-rag-api-332613889772.asia-south1.run.app/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RBI inflation forecast?", "domain": "rbi", "session_id": "demo"}'
```

### GCP Architecture

| GCP Service | Purpose |
|-------------|---------|
| Artifact Registry | Stores Docker images |
| Cloud Run | Serverless container hosting — scales to zero when idle |
| GCS (Cloud Storage) | Stores ChromaDB vector store and query logs |
| Secret Manager | Stores GROQ_API_KEY securely |
| BigQuery | Analyses query logs |

### Startup Flow
On every Cold Start, Cloud Run:
1. Pulls image from Artifact Registry
2. Downloads ChromaDB from GCS
3. Loads HuggingFace embedding model
4. Starts serving requests

---

## 🔄 GCP → AWS Equivalents

| GCP Service | AWS Equivalent | Purpose |
|-------------|----------------|---------|
| Cloud Run | AWS App Runner / ECS Fargate | Serverless container hosting |
| Artifact Registry | Amazon ECR | Docker image storage |
| Cloud Storage (GCS) | Amazon S3 | Object storage |
| BigQuery | Amazon Redshift / Athena | Data warehouse / analytics |
| Secret Manager | AWS Secrets Manager | Secret storage |
| Cloud Logging | Amazon CloudWatch | Log management |

### Key Design Decisions

**Why Hybrid Search?**
Pure vector search misses exact technical terms (section numbers, specific metrics). Pure BM25 misses semantic similarity ("data breach" vs "personal data incident"). Hybrid search with RRF fusion gets the best of both.

**Why Reciprocal Rank Fusion?**
Vector search returns cosine distances, BM25 returns TF-IDF scores — incompatible units. RRF normalises both into ranks and combines them, giving a fair fusion score without unit conversion.

**Why a Confidence Threshold?**
ChromaDB always returns something, even for irrelevant queries. Without a threshold, the LLM hallucinates answers from unrelated chunks. Distance > 1.3 triggers a "I don't have enough information" response instead.

**Why Separate Collections per Domain?**
Each domain has its own ChromaDB collection. This prevents cross-domain retrieval contamination and allows independent updates — add a new document by ingesting one collection without touching others.

**Why Session-based Memory?**
Each `session_id:domain` pair gets its own `RAGChatbot` instance with independent conversation history. Multi-turn context is maintained by passing the full `chat_history` list to the LLM on every call.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | LLaMA 3.1 8B via Groq API |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace, local) |
| Vector Store | ChromaDB (persistent) |
| Keyword Search | BM25 (rank-bm25) |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Framework | LangChain LCEL |
| API | FastAPI + Pydantic |
| Frontend | Streamlit |
| Containerisation | Docker + docker-compose |

---

## 🚀 Running Locally

### Prerequisites
- Python 3.11
- Groq API key (free at console.groq.com)

### Setup

```bash
git clone https://github.com/adityapatil371/multi-domain-rag
cd multi-domain-rag
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your Groq API key to `.env`:
GROQ_API_KEY=your_key_here

Add your PDFs to `data/`:
data/
├── Zomato Annual Report 2023.pdf
├── RBI Monetary Policy Report October 2024.pdf
└── Digital Personal Data Protection Act 2023 India Gazette.pdf

### Ingest Documents

```bash
python src/ingest.py
```

This embeds all three documents into persistent ChromaDB collections. Run once — results are saved to `chroma_db/`.

### Start the API

```bash
uvicorn main:app --port 8000
```

### Start the Frontend

```bash
streamlit run app.py --server.port 8501
```

Open `http://localhost:8501`

---

## 🐳 Running with Docker

```bash
docker-compose up --build
```

- API: `http://localhost:8000`
- Frontend: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

---

## 📡 API Reference

### `GET /health`
Returns API status and available domains.

### `GET /domains`
Returns domain list with descriptions.

### `POST /chat`
```json
{
  "session_id": "user123",
  "domain": "dpdp",
  "question": "What are the penalties for a data breach?"
}
```
Response:
```json
{
  "answer": "According to the Act, penalties for data breach...",
  "confident": true,
  "sources": [20],
  "session_id": "user123",
  "domain": "dpdp"
}
```

### `POST /reset`
Clears conversation history for a session.

---

## 💬 Example Queries

**DPDP Act:**
- "What are the rights of a Data Principal?"
- "What are the penalties for a data breach?"
- "Who is a Significant Data Fiduciary and what are their obligations?"

**Zomato Annual Report:**
- "What was Zomato's revenue in FY2024?"
- "What is Zomato's quick commerce strategy?"
- "How did food delivery performance change year over year?"

**RBI Monetary Policy:**
- "What is RBI's inflation forecast for 2024-25?"
- "What are the key risks to India's economic outlook?"
- "How has the repo rate changed recently?"

---

## ⚠️ Known Limitations

- **Retrieval sensitivity** — query phrasing affects retrieval quality. "Gross Order Value" may not retrieve the same chunks as "GOV" or "order value metrics"
- **PDF extraction artifacts** — some PDFs produce spacing artifacts during text extraction that slightly affect chunk quality
- **Context window** — very long answers may get truncated by the LLM's context window
- **Local embeddings** — `all-MiniLM-L6-v2` is fast but smaller models like `bge-large` would improve retrieval quality at the cost of speed

---

## 🔧 What I'd Improve

- **Re-ranking** — add a cross-encoder re-ranker after hybrid retrieval for better precision
- **Query expansion** — use the LLM to generate alternative phrasings of the query before retrieval
- **Streaming responses** — stream LLM output token by token for better UX
- **Persistent sessions** — store conversation history in Redis so sessions survive API restarts
- **Evaluation pipeline** — build a RAGAS evaluation set to measure retrieval precision and answer faithfulness

---

## 📁 Project Structure
multi-domain-rag/
├── src/
│   ├── init.py
│   ├── ingest.py        # PDF loading, chunking, embedding
│   ├── retriever.py     # Hybrid search with RRF fusion
│   └── chatbot.py       # LLM chain with multi-turn memory
├── data/                # PDF documents (gitignored)
├── chroma_db/           # Persistent vector store (gitignored)
├── main.py              # FastAPI application
├── app.py               # Streamlit frontend
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .gitignore