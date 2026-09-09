import sys

import chromadb
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "bartenders_guide"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3


def search(query: str, k: int = TOP_K) -> None:
    model = SentenceTransformer(EMBED_MODEL)
    query_vec = model.encode(query).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
        include=["documents", "distances"],
    )

    docs = results["documents"][0]
    distances = results["distances"][0]

    print(f'Query: "{query}"\n')
    for rank, (doc, dist) in enumerate(zip(docs, distances), 1):
        score = 1 - dist  # cosine distance → similarity
        print(f"--- Result {rank}  (similarity: {score:.3f}) ---")
        print(doc[:600] + ("..." if len(doc) > 600 else ""))
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query.py \"your question here\"")
        sys.exit(1)
    search(" ".join(sys.argv[1:]))
