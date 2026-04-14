import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DOMAINS = {
    "zomato": {
        "file": "Zomato Annual Report 2023.pdf",
        "description": "Zomato Annual Report 2023 — business performance, financials, strategy"
    },
    "rbi": {
        "file": "RBI Monetary Policy Report October 2024.pdf",
        "description": "RBI Monetary Policy Report October 2024 — inflation, interest rates, economy"
    },
    "dpdp": {
        "file": "Digital Personal Data Protection Act 2023 India Gazette.pdf",
        "description": "Digital Personal Data Protection Act 2023 — data privacy law in India"
    }
}

DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

def get_embedding_model():
    """Load the sentence transformer embedding model."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

def load_and_split(pdf_path: Path) -> list:
    """Load a PDF and split into chunks."""
    print(f"  Loading {pdf_path.name}...")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    print(f"  Loaded {len(pages)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_documents(pages)
    print(f"  Split into {len(chunks)} chunks")
    return chunks

def ingest_domain(domain_name: str, embedding_model):
    """Ingest one domain's PDF into its own ChromaDB collection."""
    domain = DOMAINS[domain_name]
    pdf_path = DATA_DIR / domain["file"]

    if not pdf_path.exists():
        print(f"  WARNING: {pdf_path} not found. Skipping.")
        return

    print(f"\nIngesting domain: {domain_name}")
    chunks = load_and_split(pdf_path)

    for chunk in chunks:
        chunk.metadata["domain"] = domain_name
        chunk.metadata["description"] = domain["description"]

    collection_path = str(CHROMA_DIR / domain_name)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=collection_path,
        collection_name=domain_name
    )
    print(f"  Stored {vectorstore._collection.count()} chunks in ChromaDB")
    print(f"  Collection saved to {collection_path}")

def ingest_all():
    """Ingest all domains."""
    print("Loading embedding model...")
    embedding_model = get_embedding_model()
    print("Embedding model loaded.")

    for domain_name in DOMAINS:
        ingest_domain(domain_name, embedding_model)

    print("\nAll domains ingested successfully.")

if __name__ == "__main__":
    ingest_all()