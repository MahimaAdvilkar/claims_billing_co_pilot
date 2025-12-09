from typing import List, Dict
from langchain_ollama import OllamaEmbeddings
from .vectordb import get_collection

# -----------------------------------------
# Embedding Model
# -----------------------------------------
EMBED_MODEL = "nomic-embed-text"
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# -----------------------------------------
# VectorDB Collection
# -----------------------------------------
collection = get_collection()

# -----------------------------------------
# Simple Retrieve Function
# -----------------------------------------
def retrieve(query: str, k: int = 5) -> List[Dict]:
    """
    Retrieve k most relevant results from the vector database
    using Ollama embeddings.

    Returns a list of dictionaries:
    {
        "id": ...,          # chunk id
        "text": ...,        # content
        "metadata": ...,    # file/row info
        "score": ...        # distance score
    }
    """
    try:
        # Generate embedding for the query
        query_vec = embeddings.embed_query(query)

        # Query the vector database
        result = collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(result["ids"][0])):
            hits.append(
                {
                    "id": result["ids"][0][i],
                    "text": result["documents"][0][i],
                    "metadata": result["metadatas"][0][i],
                    "score": float(result["distances"][0][i]),
                }
            )

        # Sort by ascending distance score (lower = more similar)
        hits.sort(key=lambda x: x["score"])
        return hits

    except Exception as e:
        print(f"❌ RAG Retrieval Error: {e}")
        return []
