# Bartender's Guide RAG

A retrieval-augmented generation (RAG) system that lets you ask questions about Jerry Thomas's 1862 *Bar-Tender's Guide* — the first cocktail book ever published in the United States. Ask it for a recipe, a historical technique, or a punch for a party of twenty, and it retrieves the relevant passages and has Claude answer in plain English.

## What it does

The source document is an OCR'd scan of the original 1862 text, with all its period spelling, garbled characters, and archaic measurements intact. Rather than asking Claude to answer from memory (which would produce anachronistic answers), this system grounds every response in the actual text. If a drink isn't in the book — like a margarita — it says so.

## Tech stack

| Layer | Tool |
|---|---|
| Language | Python 3 |
| LLM | [Claude](https://anthropic.com) via the Anthropic API (`claude-opus-4-7`) |
| Embeddings | [`sentence-transformers`](https://www.sbert.net/) — `all-MiniLM-L6-v2` (runs locally, no API key needed) |
| Vector store | [ChromaDB](https://www.trychroma.com/) (local, persistent) |
| HTML parsing | Python stdlib `html.parser` |

## How the pipeline works

```
ingest.py   →   embed.py   →   rag.py
─────────       ────────       ──────
Parse HTML      Embed each     On each question:
Strip IA        chunk with     1. Embed the query
header          MiniLM         2. Retrieve top-3 chunks
Split into      Store in       3. Pass chunks + question
~400-word       ChromaDB          to Claude
chunks                         4. Stream the answer
```

Run `ingest.py` and `embed.py` once to build the vector store. After that, only `rag.py` is needed.

## Setup

**1. Clone and create a virtual environment**

```bash
git clone https://github.com/jasewartgow/rag-project.git
cd rag-project
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install anthropic python-dotenv sentence-transformers chromadb
```

**3. Add your Anthropic API key**

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_key_here
```

**4. Build the vector store**

```bash
python3 ingest.py   # preview the chunks (optional)
python3 embed.py    # embed and store in ChromaDB
```

This downloads the `all-MiniLM-L6-v2` model on first run (~90 MB) and writes the vector store to `./chroma_db/`.

**5. Ask questions**

Interactive chat loop:

```bash
python3 rag.py
```

Single question:

```bash
python3 rag.py "how do you make a mint julep"
```

## Example Q&As

### A recipe it knows

**Q: Baltimore egg nogg recipe**

> Here is the Baltimore Egg Nogg recipe from Jerry Thomas's 1862 *Bar-Tender's Guide* (recipe No. 84), designed to serve a party of fifteen:
>
> **Ingredients:**
> - Yolks of 16 eggs
> - 12 tablespoonfuls of pulverized loaf-sugar
> - Two-thirds of a nutmeg, grated
> - ½ pint of good brandy or Jamaica rum
> - 2 wine-glasses of Madeira wine
> - Whites of the 16 eggs, beaten to a stiff froth
> - 6 pints of good rich milk
>
> **Notes from Thomas:** "There is no heat used. Egg Nogg made in this manner is digestible, and will not cause headache. It makes an excellent drink for debilitated persons, and a nourishing diet for consumptives."
>
> What distinguishes the Baltimore version from the standard egg nogg is the use of **Madeira wine** alongside the spirits, and the separation and whipping of the whites into a stiff froth — a more refined, punch-bowl style preparation suited to a genteel Southern gathering.

---

### A recipe outside the book

**Q: How do you make a margarita?**

> The provided excerpts do not contain a recipe for a margarita. This isn't surprising, as the margarita is a 20th-century cocktail and would not have appeared in an 1862 bartending manual.
>
> If you're interested in a citrus-forward drink from this era, the **Brandy Crusta** (recipe #116) is often cited by cocktail historians as a spiritual ancestor of the margarita, featuring spirit, citrus juice, curaçao, and a sugared rim:
>
> *"Crusta is made the same as a fancy cocktail, with a little lemon juice and a small lump of ice added. First, mix the ingredients in a small tumbler, then take a fancy red wine-glass, rub a sliced lemon around the rim of the same, and dip it in pulverized white sugar, so that the sugar will adhere to the edge of the glass..."*
