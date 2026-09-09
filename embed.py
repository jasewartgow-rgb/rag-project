import chromadb
from sentence_transformers import SentenceTransformer

from ingest import extract_text, split_into_chunks

SOURCE_FILE = "Full text of _The bar-tenders' guide __.html"
COLLECTION_NAME = "bartenders_guide"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def main():
    print("Loading chunks...")
    text = extract_text(SOURCE_FILE)
    chunks = split_into_chunks(text)
    print(f"  {len(chunks)} chunks ready\n")

    print(f"Loading embedding model ({EMBED_MODEL})...")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Embedding {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    embeddings = model.encode(
        chunks,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
    ).tolist()
    print()

    print(f"Storing in Chroma at {CHROMA_DIR!r}...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Start fresh each run so re-running doesn't accumulate duplicates
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
    )

    stored = collection.count()
    print(f"\nDone — {stored} chunks embedded and stored in '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
