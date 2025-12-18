from typing import List
from pathlib import Path
from langchain_community.embeddings import OllamaEmbeddings
from .parse_pdfs import parse_pdf
from .parse_csvs import parse_csv
from ..rag.vectordb import get_collection

EMBED_MODEL = "nomic-embed-text"

emb = OllamaEmbeddings(model=EMBED_MODEL)
coll = get_collection()

def upsert_documents(file_paths: List[str]):
    docs = []
    for fp in file_paths:
        if fp.lower().endswith(".pdf"):
            docs.extend(parse_pdf(fp))
        elif fp.lower().endswith(".csv"):
            docs.extend(parse_csv(fp))
        else:
            continue
    if not docs:
        return 0

    texts = [d["text"] for d in docs]
    metadatas = [d["metadata"] for d in docs]
    ids = [f"{m['source']}::{m.get('page', m.get('row', 0))}::{i}" for i, m in enumerate(metadatas)]

    vectors = emb.embed_documents(texts)
    coll.add(ids=ids, embeddings=vectors, metadatas=metadatas, documents=texts)
    return len(texts)

if __name__ == "__main__":
    raw = Path("data/raw").glob("*")
    n = upsert_documents([str(p) for p in raw])
    print(f"Upserted {n} chunks")
