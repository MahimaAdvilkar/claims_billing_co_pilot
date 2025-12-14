import os
import chromadb
from dotenv import load_dotenv

load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")

_client = chromadb.PersistentClient(path=CHROMA_DIR)

def get_collection(name: str = "claims_corpus"):
    return _client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
