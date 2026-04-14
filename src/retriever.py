from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

DOMAINS = {
    "zomato": "Zomato Annual Report 2023 — business performance, financials, strategy",
    "rbi": "RBI Monetary Policy Report October 2024 — inflation, interest rates, economy",
    "dpdp": "Digital Personal Data Protection Act 2023 — data privacy law in India"
}

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

def load_vectorstore(domain: str, embedding_model) -> Chroma:
    """Load an existing ChromaDB collection from disk."""
    collection_path = str(CHROMA_DIR / domain)
    return Chroma(
        persist_directory=collection_path,
        embedding_function=embedding_model,
        collection_name=domain
    )

def hybrid_search(query: str, domain: str, embedding_model, n_results: int = 5) -> list[dict]:
    """
    Combine vector search and BM25 keyword search using Reciprocal Rank Fusion.
    Returns top chunks ranked by combined relevance.
    """
    # Load vectorstore for this domain
    vectorstore = load_vectorstore(domain, embedding_model)

    # Get all documents for BM25
    all_docs = vectorstore.get()
    documents = all_docs["documents"]
    metadatas = all_docs["metadatas"]

    if not documents:
        return []

    # --- VECTOR SEARCH ---
    vector_results = vectorstore.similarity_search_with_score(query, k=10)
    # Returns list of (Document, distance) tuples — lower distance = more relevant

    # --- BM25 KEYWORD SEARCH ---
    corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(corpus)
    tokenised_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenised_query)
    bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]

    # --- RECIPROCAL RANK FUSION ---
    # RRF score = sum of 1/(rank + k) across both systems
    # k=60 is the standard constant that prevents top ranks from dominating
    k = 60
    rrf_scores = {}

    # Score from vector search
    for rank, (doc, distance) in enumerate(vector_results):
        doc_id = doc.page_content[:100]  # use first 100 chars as unique key
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": 0,
                "distance": distance
            }
        rrf_scores[doc_id]["score"] += 1 / (rank + k)

    # Score from BM25
    for rank, idx in enumerate(bm25_ranked):
        doc_id = documents[idx][:100]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "content": documents[idx],
                "metadata": metadatas[idx],
                "score": 0,
                "distance": 1.0  # unknown distance for BM25-only results
            }
        rrf_scores[doc_id]["score"] += 1 / (rank + k)

    # Sort by combined RRF score — higher is better
    ranked = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:n_results]

def retrieve(query: str, domain: str, embedding_model, confidence_threshold: float = 1.3) -> dict:
    """
    Main retrieval function. Returns chunks if confident, flags if not.
    """
    if domain not in DOMAINS:
        return {
            "confident": False,
            "reason": f"Unknown domain '{domain}'. Available: {list(DOMAINS.keys())}",
            "chunks": []
        }

    results = hybrid_search(query, domain, embedding_model, n_results=5)

    if not results:
        return {"confident": False, "reason": "No results found.", "chunks": []}

    # Use top result's distance as confidence signal
    top_distance = results[0].get("distance", 1.0)

    if top_distance > confidence_threshold:
        return {
            "confident": False,
            "reason": f"Top result distance {top_distance:.3f} exceeds threshold {confidence_threshold}.",
            "chunks": results
        }

    return {
        "confident": True,
        "reason": None,
        "chunks": results
    }

