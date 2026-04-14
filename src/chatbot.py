import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

DOMAIN_CONTEXT = {
    "zomato": "Zomato's Annual Report 2023 covering business performance, financials, and strategy",
    "rbi": "RBI Monetary Policy Report October 2024 covering inflation, interest rates, and economic outlook",
    "dpdp": "Digital Personal Data Protection Act 2023 covering data privacy law in India"
}

def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0.1
    )

def format_chunks(chunks: list[dict]) -> str:
    """Format retrieved chunks into a single context string."""
    formatted = []
    for i, chunk in enumerate(chunks):
        page = chunk["metadata"].get("page", "unknown")
        formatted.append(f"[Source {i+1} | Page {page}]\n{chunk['content']}")
    return "\n\n".join(formatted)

def build_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful assistant that answers questions based strictly on the provided document context.

Document: {domain_context}

Rules:
- Answer only based on the context provided below
- If the context does not contain enough information, say so clearly
- Always mention which page your answer comes from
- Be concise and precise
- Do not make up information

Context:
{context}"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

class RAGChatbot:
    def __init__(self, domain: str, embedding_model, confidence_threshold: float = 1.3):
        from src.retriever import retrieve

        self.domain = domain
        self.embedding_model = embedding_model
        self.confidence_threshold = confidence_threshold
        self.llm = get_llm()
        self.prompt = build_prompt()
        self.chain = self.prompt | self.llm
        self.chat_history = []
        self.retrieve = retrieve

    def chat(self, question: str) -> dict:
        """Process one turn of conversation. Returns answer and metadata."""

        retrieval = self.retrieve(
            query=question,
            domain=self.domain,
            embedding_model=self.embedding_model,
            confidence_threshold=self.confidence_threshold
        )

        if not retrieval["confident"]:
            response = "I don't have enough information in this document to answer that confidently."
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=response))
            return {
                "answer": response,
                "confident": False,
                "sources": []
            }

        context = format_chunks(retrieval["chunks"])

        response = self.chain.invoke({
            "domain_context": DOMAIN_CONTEXT[self.domain],
            "context": context,
            "chat_history": self.chat_history,
            "question": question
        })

        answer = response.content

        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        sources = list(set([
            chunk["metadata"].get("page", "unknown")
            for chunk in retrieval["chunks"]
        ]))

        return {
            "answer": answer,
            "confident": True,
            "sources": sorted(sources)
        }

    def reset(self):
        """Clear conversation history."""
        self.chat_history = []

    