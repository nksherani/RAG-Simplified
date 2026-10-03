import os
import numpy as np
from openai import OpenAI

# Initialize the OpenAI client (picks up OPENAI_API_KEY from environment variables)
client = OpenAI()


# ==========================================
# 1. KNOWLEDGE BASE (Documents)
# ==========================================
documents = [
    "Antigravity is an advanced agentic IDE created to streamline software engineering workflows.",
    "Python 3.12 introduced improved error messages and significant performance enhancements.",
    "Retrieval-Augmented Generation (RAG) combines search retrieval with LLMs to provide grounded answers.",
    "Cosine similarity measures the cosine of the angle between two non-zero vectors in an inner product space.",
    "Vector databases like Chroma, Pinecone, and FAISS store embeddings for fast similarity search."
]


# ==========================================
# 2. EMBEDDING GENERATION
# ==========================================
def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Generate an embedding vector for a given text using OpenAI SDK."""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding


# ==========================================
# 3. VECTOR SIMILARITY (Cosine Similarity)
# ==========================================
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec_a = np.array(a)
    vec_b = np.array(b)
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


# ==========================================
# 4. RETRIEVAL STEP
# ==========================================
def retrieve_relevant_docs(query: str, doc_embeddings: list[list[float]], top_k: int = 2) -> list[str]:
    """Find the top-k most relevant documents for a query."""
    query_embedding = get_embedding(query)
    
    # Calculate similarity score with all documents
    scores = [
        (doc, cosine_similarity(query_embedding, doc_emb))
        for doc, doc_emb in zip(documents, doc_embeddings)
    ]
    
    # Sort documents by descending similarity score
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return the text of the top-k documents
    return [doc for doc, score in scores[:top_k]]


# ==========================================
# 5. AUGMENTED GENERATION STEP
# ==========================================
def answer_query_with_rag(query: str, doc_embeddings: list[list[float]]) -> str:
    """Retrieve relevant context and generate an answer using OpenAI Chat Completion."""
    # Step A: Retrieve relevant context
    retrieved_docs = retrieve_relevant_docs(query, doc_embeddings, top_k=2)
    context_str = "\n".join([f"- {doc}" for doc in retrieved_docs])
    
    # Step B: Construct prompt with retrieved context (Augmentation)
    system_prompt = (
        "You are a helpful assistant. Use the following context to answer the user's question. "
        "If the answer is not in the context, state that you do not know based on the provided context."
    )
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"
    
    # Step C: Generation
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    
    return response.choices[0].message.content


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Indexing documents into embeddings...")
    # Precompute embeddings for our knowledge base
    doc_embeddings = [get_embedding(doc) for doc in documents]
    print(f"Indexed {len(doc_embeddings)} documents.\n")

    user_query = "What is RAG and how does it work?"
    print(f"User Query: '{user_query}'\n")

    answer = answer_query_with_rag(user_query, doc_embeddings)
    print("--- RAG Response ---")
    print(answer)
