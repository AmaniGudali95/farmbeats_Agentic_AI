#!/usr/bin/env python3
"""
FarmBeats RAG CLI
=================
A retrieval-augmented generation system for the FarmBeats paper.
Ask questions in plain English and get answers grounded in the research.

Usage:
    python rag_cli.py                    # interactive chat
    python rag_cli.py --ingest           # (re)build the vector index
    python rag_cli.py --query "..."      # single query and exit
    python rag_cli.py --show-sources     # print retrieved chunks with answers
"""

import os
import sys
import json
import time
import textwrap
import argparse
from pathlib import Path

# ── Dependency check ──────────────────────────────────────────────────────────
MISSING = []
try:
    import anthropic
except ImportError:
    MISSING.append("anthropic")

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    MISSING.append("chromadb")

try:
    import pypdf
except ImportError:
    MISSING.append("pypdf")

if MISSING:
    print("\n[setup] Missing packages. Run:\n")
    print(f"  pip install {' '.join(MISSING)}\n")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

COLLECTION_NAME  = "farmbeats"
CHUNK_SIZE       = 400      # words per chunk
CHUNK_OVERLAP    = 60       # word overlap between consecutive chunks
TOP_K            = 5        # number of chunks to retrieve per query
MODEL            = "claude-opus-4-6"
EMBED_MODEL      = "text-embedding-3-small"  # used via Anthropic or fallback
DB_PATH          = str(Path(__file__).parent / "data" / "chroma_db")
DOCS_DIR         = Path(__file__).parent / "docs"

PAPER_URL = (
    "https://www.microsoft.com/en-us/research/wp-content/uploads/"
    "2022/09/Democratizing_Data-Driven_Agriculture_Using_Affordable_Hardware.pdf"
)

SYSTEM_PROMPT = """You are an expert agricultural AI assistant with deep knowledge \
of the Microsoft FarmBeats research program. You help farmers and researchers \
understand data-driven agriculture, IoT sensor systems, drone/satellite imaging, \
affordable connectivity, and AI-driven farm insights.

When answering:
- Base your answer on the retrieved passages from the FarmBeats research paper
- Be specific and cite concrete details (numbers, techniques, components)
- Explain technical concepts in plain language a farmer can understand
- If the retrieved passages don't contain enough information, say so clearly
- Keep answers practical and actionable where possible

You will receive retrieved passages from the paper before each question."""

WELCOME = """
╔══════════════════════════════════════════════════════════╗
║        FarmBeats RAG CLI  —  Powered by Claude           ║
║   Ask questions about the FarmBeats research paper       ║
╚══════════════════════════════════════════════════════════╝

Type your question and press Enter.
Commands:  /sources   toggle showing retrieved passages
           /clear     clear conversation history
           /stats     show index statistics
           /help      show this message
           /quit      exit

"""

# ── Colour helpers (works in most terminals) ──────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BLUE   = "\033[34m"
RED    = "\033[31m"

def c(color, text): return f"{color}{text}{RESET}"
def wrap(text, width=80, indent=""):
    return textwrap.fill(text, width=width, subsequent_indent=indent)


# ── PDF loading ───────────────────────────────────────────────────────────────

def load_pdf(path: Path) -> str:
    """Extract all text from a PDF file."""
    reader = pypdf.PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append(f"[Page {i+1}]\n{text.strip()}")
    return "\n\n".join(pages)


def load_text(path: Path) -> str:
    """Load a plain text file."""
    return path.read_text(encoding="utf-8", errors="replace")


def load_all_docs(docs_dir: Path) -> list[dict]:
    """Load all PDFs and .txt files from the docs directory."""
    supported = {".pdf", ".txt", ".md"}
    docs = []
    for f in sorted(docs_dir.iterdir()):
        if f.suffix.lower() not in supported:
            continue
        print(f"  Loading {f.name} ...", end=" ", flush=True)
        try:
            if f.suffix.lower() == ".pdf":
                text = load_pdf(f)
            else:
                text = load_text(f)
            docs.append({"name": f.name, "text": text})
            print(c(GREEN, "ok"))
        except Exception as e:
            print(c(RED, f"failed: {e}"))
    return docs


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split text into overlapping word-window chunks.
    Returns list of {id, text, source, chunk_index, word_start}.
    """
    words = text.split()
    chunks = []
    step   = chunk_size - overlap
    idx    = 0

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if len(chunk_words) < 20:          # skip tiny trailing fragments
            break
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "id":          f"{source}__chunk_{idx}",
            "text":        chunk_text,
            "source":      source,
            "chunk_index": idx,
            "word_start":  start,
        })
        idx += 1

    return chunks


# ── Embedding (Anthropic) ─────────────────────────────────────────────────────

class AnthropicEmbedder:
    """
    Thin wrapper around the Anthropic embeddings API.
    Falls back to a simple TF-IDF-style hash embedding if the API key
    is missing so the demo still runs without credentials.
    """

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else None
        self._dim  = 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return a list of float vectors, one per input text."""
        if self.client:
            return self._embed_anthropic(texts)
        return self._embed_fallback(texts)

    def _embed_anthropic(self, texts: list[str]) -> list[list[float]]:
        """
        Use Anthropic's embedding endpoint (batched in groups of 96
        to stay within the API limit).
        """
        all_vectors = []
        batch_size  = 96
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp  = self.client.beta.messages.batches  # placeholder
            # Actual embeddings call — requires anthropic>=0.30
            try:
                result = self.client.embeddings.create(
                    model=EMBED_MODEL, input=batch
                )
                all_vectors.extend([e.embedding for e in result.data])
            except Exception:
                # Fall back gracefully if embedding endpoint unavailable
                all_vectors.extend(self._embed_fallback(batch))
        return all_vectors

    def _embed_fallback(self, texts: list[str]) -> list[list[float]]:
        """
        Deterministic hash-based pseudo-embedding.
        Works without any API key — useful for testing the pipeline.
        Quality is much lower than a real embedding model.
        """
        import hashlib, math
        vectors = []
        dim = self._dim
        for text in texts:
            vec = [0.0] * dim
            words = text.lower().split()
            for word in set(words):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                for k in range(4):
                    idx = (h >> (k * 8)) % dim
                    vec[idx] += 1.0
            norm = math.sqrt(sum(x*x for x in vec)) or 1.0
            vectors.append([x / norm for x in vec])
        return vectors

    @property
    def dimension(self):
        return self._dim


# ── ChromaDB vector store ─────────────────────────────────────────────────────

class VectorStore:
    """Wraps ChromaDB for storing and querying chunk embeddings."""

    def __init__(self, db_path: str, embedder: AnthropicEmbedder):
        self.embedder = embedder
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.chroma  = chromadb.PersistentClient(path=db_path)
        # Use an embedding function ChromaDB can call internally
        self.collection = self.chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def is_empty(self) -> bool:
        return self.collection.count() == 0

    def count(self) -> int:
        return self.collection.count()

    def clear(self):
        self.chroma.delete_collection(COLLECTION_NAME)
        self.collection = self.chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict], batch_size: int = 64):
        """Embed and store a list of chunk dicts."""
        total = len(chunks)
        added = 0
        for i in range(0, total, batch_size):
            batch    = chunks[i : i + batch_size]
            texts    = [c["text"] for c in batch]
            vectors  = self.embedder.embed(texts)
            ids      = [c["id"] for c in batch]
            metas    = [{k: v for k, v in c.items() if k != "text"}
                        for c in batch]
            self.collection.add(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metas,
            )
            added += len(batch)
            pct = int(added / total * 40)
            bar = "█" * pct + "░" * (40 - pct)
            print(f"\r  [{bar}] {added}/{total}", end="", flush=True)
        print()

    def query(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """Return the top-k most relevant chunks for a query."""
        vector  = self.embedder.embed([question])[0]
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        passages = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            passages.append({
                "text":        doc,
                "source":      meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "score":       round(1 - dist, 3),   # cosine similarity
            })
        return passages


# ── Claude answer generation ──────────────────────────────────────────────────

class FarmBeatsRAG:
    """Orchestrates retrieval + Claude generation."""

    def __init__(self, store: VectorStore):
        self.store    = store
        self.history  = []   # list of {role, content} for multi-turn

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print(c(YELLOW,
                "Warning: ANTHROPIC_API_KEY not set. "
                "Set it to get Claude answers.\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...\n"
            ))
        self.claude = anthropic.Anthropic(api_key=api_key) if api_key else None

    def ask(self, question: str, top_k: int = TOP_K) -> dict:
        """
        Full RAG pipeline:
          1. Retrieve relevant chunks
          2. Build a context-enriched prompt
          3. Call Claude with conversation history
          4. Return answer + sources
        """
        # ── 1. Retrieve ──────────────────────────────────────────────
        passages = self.store.query(question, top_k=top_k)

        # ── 2. Build retrieval context block ─────────────────────────
        context_lines = []
        for i, p in enumerate(passages, 1):
            context_lines.append(
                f"[Passage {i} | source: {p['source']} | "
                f"relevance: {p['score']:.2f}]\n{p['text']}"
            )
        context_block = "\n\n---\n\n".join(context_lines)

        user_message = (
            f"Retrieved passages from the FarmBeats research paper:\n\n"
            f"{context_block}\n\n"
            f"---\n\n"
            f"Question: {question}"
        )

        # ── 3. Multi-turn history ─────────────────────────────────────
        self.history.append({"role": "user", "content": user_message})

        # ── 4. Call Claude ────────────────────────────────────────────
        if not self.claude:
            answer = (
                "[Claude not available — ANTHROPIC_API_KEY not set]\n\n"
                "Retrieved passages:\n" +
                "\n\n".join(p["text"][:300] + "..." for p in passages)
            )
        else:
            response = self.claude.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=self.history,
            )
            answer = response.content[0].text

        self.history.append({"role": "assistant", "content": answer})
        return {"answer": answer, "passages": passages}

    def clear_history(self):
        self.history = []


# ── Ingest pipeline ───────────────────────────────────────────────────────────

def ingest(store: VectorStore, force: bool = False):
    """Load all docs from the docs/ folder, chunk them, and index them."""
    if not store.is_empty() and not force:
        print(c(GREEN, f"Index already contains {store.count()} chunks. "
                       "Use --ingest --force to rebuild.\n"))
        return

    print(c(BOLD, "\nIngesting documents into vector store...\n"))

    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True)

    pdfs = list(DOCS_DIR.glob("*.pdf"))
    txts = list(DOCS_DIR.glob("*.txt")) + list(DOCS_DIR.glob("*.md"))

    if not pdfs and not txts:
        print(c(YELLOW,
            f"No documents found in {DOCS_DIR}/\n\n"
            "Download the FarmBeats paper and place it there:\n"
            f"  {PAPER_URL}\n\n"
            "Or create a text file:\n"
            f"  echo 'Your content here' > {DOCS_DIR}/notes.txt\n"
        ))
        sys.exit(0)

    docs   = load_all_docs(DOCS_DIR)
    chunks = []

    for doc in docs:
        doc_chunks = chunk_text(doc["text"], source=doc["name"])
        chunks.extend(doc_chunks)
        print(f"  {doc['name']}: {len(doc_chunks)} chunks")

    print(f"\n  Total: {len(chunks)} chunks to embed\n")

    if force and not store.is_empty():
        print("  Clearing existing index...")
        store.clear()

    print("  Embedding and indexing (this may take a minute)...")
    store.add_chunks(chunks)
    print(c(GREEN, f"\n  Done! {store.count()} chunks indexed.\n"))


# ── CLI display helpers ───────────────────────────────────────────────────────

def print_passages(passages: list[dict]):
    print(c(DIM, "\n" + "─" * 60))
    print(c(DIM, f"  Retrieved {len(passages)} passages:\n"))
    for i, p in enumerate(passages, 1):
        score_color = GREEN if p["score"] > 0.7 else YELLOW if p["score"] > 0.5 else DIM
        print(c(score_color, f"  [{i}] {p['source']}  score={p['score']:.3f}"))
        excerpt = p["text"][:220].replace("\n", " ")
        print(c(DIM, f"      {excerpt}...\n"))
    print(c(DIM, "─" * 60 + "\n"))


def print_answer(answer: str):
    print()
    print(c(CYAN, "Assistant"))
    print(c(DIM, "─" * 60))
    # Wrap long lines nicely
    for para in answer.split("\n"):
        if para.strip():
            print(wrap(para, width=78, indent="  "))
        else:
            print()
    print()


def spinner(msg: str):
    """Print a simple "thinking" indicator."""
    print(c(DIM, f"  {msg}"), end="\r", flush=True)


# ── Main CLI ──────────────────────────────────────────────────────────────────

def interactive_loop(rag: FarmBeatsRAG, show_sources: bool = False):
    show_src = show_sources

    print(WELCOME)
    print(c(DIM, f"  Index: {rag.store.count()} chunks  |  "
                 f"Model: {MODEL}  |  top_k={TOP_K}\n"))

    while True:
        try:
            raw = input(c(GREEN, "You > ")).strip()
        except (EOFError, KeyboardInterrupt):
            print(c(DIM, "\n\nGoodbye. Happy farming!\n"))
            break

        if not raw:
            continue

        # ── Commands ──────────────────────────────────────────────────
        if raw.lower() in ("/quit", "/exit", "/q"):
            print(c(DIM, "\nGoodbye. Happy farming!\n"))
            break

        if raw.lower() == "/sources":
            show_src = not show_src
            state = c(GREEN, "ON") if show_src else c(DIM, "OFF")
            print(c(DIM, f"  Source display {state}\n"))
            continue

        if raw.lower() == "/clear":
            rag.clear_history()
            print(c(DIM, "  Conversation history cleared.\n"))
            continue

        if raw.lower() == "/stats":
            print(c(DIM,
                f"\n  Collection : {COLLECTION_NAME}\n"
                f"  Chunks     : {rag.store.count()}\n"
                f"  Model      : {MODEL}\n"
                f"  top_k      : {TOP_K}\n"
                f"  DB path    : {DB_PATH}\n"
            ))
            continue

        if raw.lower() in ("/help", "/?"):
            print(WELCOME)
            continue

        # ── RAG query ─────────────────────────────────────────────────
        spinner("Retrieving relevant passages...")
        t0     = time.time()
        result = rag.ask(raw)
        elapsed = time.time() - t0

        if show_src:
            print_passages(result["passages"])

        print_answer(result["answer"])
        print(c(DIM, f"  ({elapsed:.1f}s | {len(result['passages'])} passages retrieved)\n"))


def main():
    parser = argparse.ArgumentParser(
        description="FarmBeats RAG CLI — ask questions about the FarmBeats paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ingest",      action="store_true",
                        help="(Re)build the vector index from docs/")
    parser.add_argument("--force",       action="store_true",
                        help="Force re-ingest even if index exists")
    parser.add_argument("--query",  "-q", type=str, default=None,
                        help="Run a single query and exit")
    parser.add_argument("--show-sources", action="store_true",
                        help="Show retrieved passages alongside answers")
    parser.add_argument("--top-k",       type=int, default=TOP_K,
                        help=f"Number of passages to retrieve (default {TOP_K})")
    args = parser.parse_args()

    # ── Initialise components ─────────────────────────────────────────
    embedder = AnthropicEmbedder()
    store    = VectorStore(db_path=DB_PATH, embedder=embedder)

    if args.ingest or (store.is_empty() and not args.query):
        ingest(store, force=args.force)
        if args.ingest:
            return   # --ingest flag means "just index, then exit"

    if store.is_empty():
        print(c(YELLOW,
            "\nNo documents indexed yet. Run:\n"
            "  python rag_cli.py --ingest\n\n"
            f"after placing the FarmBeats PDF in {DOCS_DIR}/\n"
        ))
        sys.exit(0)

    rag = FarmBeatsRAG(store)

    if args.query:
        # Single-shot mode
        result = rag.ask(args.query, top_k=args.top_k)
        if args.show_sources:
            print_passages(result["passages"])
        print_answer(result["answer"])
    else:
        # Interactive mode
        interactive_loop(rag, show_sources=args.show_sources)


if __name__ == "__main__":
    main()
