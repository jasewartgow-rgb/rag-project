import sys

import anthropic
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "bartenders_guide"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

SYSTEM_PROMPT = """You are an expert on 19th-century cocktails and bartending, \
with access to Jerry Thomas's 1862 "Bar-Tender's Guide." Answer the user's \
question using only the provided excerpts. If the excerpts don't contain \
enough information to answer, say so. Quote or paraphrase the source text \
where helpful. Note that the text is OCR'd and may contain garbled characters."""


def retrieve(query: str, k: int = TOP_K) -> list[str]:
    model = SentenceTransformer(EMBED_MODEL)
    query_vec = model.encode(query).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
        include=["documents"],
    )
    return results["documents"][0]


def answer(question: str) -> None:
    load_dotenv()

    print("Retrieving relevant excerpts...")
    chunks = retrieve(question)

    context = "\n\n---\n\n".join(
        f"Excerpt {i + 1}:\n{chunk}" for i, chunk in enumerate(chunks)
    )

    client = anthropic.Anthropic()

    print(f'\nQuestion: "{question}"\n')
    print("Answer:")

    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Here are relevant excerpts from the book:\n\n{context}\n\nQuestion: {question}",
            }
        ],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rag.py \"your question here\"")
        sys.exit(1)
    answer(" ".join(sys.argv[1:]))
