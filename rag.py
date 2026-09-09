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


class RAG:
    def __init__(self):
        load_dotenv()
        print(f"Loading embedding model ({EMBED_MODEL})...")
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        chroma = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = chroma.get_collection(COLLECTION_NAME)
        self.claude = anthropic.Anthropic()
        print(f"Ready — {self.collection.count()} chunks indexed.\n")

    def retrieve(self, query: str) -> list[str]:
        vec = self.embed_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[vec],
            n_results=TOP_K,
            include=["documents"],
        )
        return results["documents"][0]

    def ask(self, question: str) -> None:
        chunks = self.retrieve(question)
        context = "\n\n---\n\n".join(
            f"Excerpt {i + 1}:\n{chunk}" for i, chunk in enumerate(chunks)
        )

        print("\nAnswer:")
        with self.claude.messages.stream(
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
        print("\n")

    def chat(self) -> None:
        print("Ask anything about Jerry Thomas's 1862 Bar-Tender's Guide.")
        print("Type 'quit' or press Ctrl+C to exit.\n")
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodnight.")
                break
            if not question:
                continue
            if question.lower() in {"quit", "exit", "q"}:
                print("Goodnight.")
                break
            self.ask(question)


if __name__ == "__main__":
    rag = RAG()
    if len(sys.argv) > 1:
        rag.ask(" ".join(sys.argv[1:]))
    else:
        rag.chat()
