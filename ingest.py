import re
from html.parser import HTMLParser
from pathlib import Path

SOURCE_FILE = "Full text of _The bar-tenders' guide __.html"
CHUNK_TARGET = 400  # words, aim for middle of 300-500 range
CHUNK_MIN = 300
CHUNK_MAX = 500


class TextExtractor(HTMLParser):
    """Strip HTML tags; skip script/style content."""

    def __init__(self):
        super().__init__()
        self._skip = False
        self._parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self._skip = True
        elif tag in ("p", "div", "br", "h1", "h2", "h3", "h4"):
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self):
        return "".join(self._parts)


def extract_text(path: str) -> str:
    raw = Path(path).read_text(encoding="utf-8")
    parser = TextExtractor()
    parser.feed(raw)
    text = parser.get_text()

    # Rejoin OCR line-break hyphens: "mix-\ning" → "mixing"
    text = re.sub(r"-\s*\n\s*", "", text)
    # Collapse runs of whitespace/newlines to single spaces or paragraph breaks
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Drop Internet Archive header — everything before the digitization notice
    marker = "Digitized by the Internet Archive"
    idx = text.find(marker)
    if idx != -1:
        text = text[idx:]

    return text.strip()


def split_into_chunks(text: str) -> list[str]:
    """
    Split on paragraph boundaries (blank lines), then merge small paragraphs
    and split oversized ones so every chunk lands in the 300-500 word range.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    chunks = []
    current_words: list[str] = []

    for para in paragraphs:
        para_words = para.split()

        # If adding this paragraph keeps us under the max, accumulate it
        if len(current_words) + len(para_words) <= CHUNK_MAX:
            current_words.extend(para_words)
        else:
            # Flush current buffer if it's large enough
            if len(current_words) >= CHUNK_MIN:
                chunks.append(" ".join(current_words))
                current_words = list(para_words)
            else:
                # Buffer too small — add paragraph and flush regardless
                current_words.extend(para_words)
                chunks.append(" ".join(current_words))
                current_words = []

            # If the paragraph itself is oversized, hard-split it
            while len(current_words) > CHUNK_MAX:
                chunks.append(" ".join(current_words[:CHUNK_TARGET]))
                current_words = current_words[CHUNK_TARGET:]

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def main():
    print(f"Reading: {SOURCE_FILE}")
    text = extract_text(SOURCE_FILE)
    print(f"Extracted {len(text.split()):,} words from source\n")

    chunks = split_into_chunks(text)
    print(f"Created {len(chunks)} chunks\n")
    print("=" * 60)

    for i, chunk in enumerate(chunks[:2], 1):
        word_count = len(chunk.split())
        print(f"CHUNK {i}  ({word_count} words)")
        print("-" * 60)
        print(chunk)
        print("=" * 60)


if __name__ == "__main__":
    main()
