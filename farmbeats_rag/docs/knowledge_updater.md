# Knowledge Updater — Weekly Pipeline

## What it does

Keeps the ChromaDB knowledge base current without manual intervention:

```
1. Search Semantic Scholar for new agriculture papers
2. Check each paper for duplicates — skip if already indexed
3. Download full PDF text where available, fall back to abstract
4. Chunk and embed with BGE
5. Add to ChromaDB — immediately available to the RAG agent
```

---

## Running it

```bash
# Run once immediately
python knowledge_updater.py

# Run on weekly schedule (every Monday at 6am)
python knowledge_updater.py --schedule
```

`sys.argv[1] == "--schedule"` checks whether `--schedule` was passed on the command line. Same file, two modes.

---

## Search queries

```python
queries = [
    "precision agriculture IoT soil moisture sensors",
    "NDVI crop health satellite imagery farming",
    "FarmBeats data driven agriculture smallholder",
    "irrigation scheduling machine learning crops",
]
```

Four queries × 3 papers each = up to 12 new papers per weekly run.

Note: Semantic Scholar rate-limits unauthenticated requests. Getting a free API key at semanticscholar.org gives higher limits and more consistent results across all four queries.

---

## Deduplication

```python
def paper_already_indexed(store, paper_id):
    chunk_id = f"semantic_scholar__{hashlib.md5(paper_id.encode()).hexdigest()[:8]}__chunk_0"
    results = store.collection.get(ids=[chunk_id])
    return len(results["ids"]) > 0
```

**Why MD5 hash?**
Semantic Scholar paper IDs contain `/` and `.` which ChromaDB rejects in chunk IDs. MD5 converts any ID into a clean 8-character alphanumeric string. Deterministic — same paper always produces the same hash.

**Why check chunk_0?**
If a paper was indexed, chunk_0 always exists — it's the first chunk created. Checking one chunk is enough to know the entire paper was indexed. No need to check all chunks.

---

## Chunk size for abstracts

Abstract-only content: chunk_size=200, overlap=30
Full paper content: chunk_size=400, overlap=60

Reason: A 150-word abstract with 400-word chunks would produce one tiny incomplete chunk or nothing. 200-word chunks fit a short abstract meaningfully.

---

## First run results

```
Searching: 'irrigation scheduling machine learning crops'
Found 3 papers

Skipping 'A Machine Learning-Based Probabilistic...' — no content
Added 'Soil quality monitoring and evaluation system...' (1 chunk)
Added 'AGROGUIDE - IoT and Machine Learning based...' (1 chunk)

Added:   2 new papers
Skipped: 0 duplicates
Total chunks now: 9 (was 7)
```

Three of four queries returned 0 results — Semantic Scholar rate limiting unauthenticated requests. One query returned 3 papers, 2 were successfully indexed.

---

## Retrieval impact

After adding 2 new papers, the irrigation question now returns a new paper in position 3:

```
Q: What soil moisture triggers corn irrigation?
  [3] score=0.635 — "Title: Soil quality monitoring and evaluation system..."
```

As more papers are added weekly, retrieval quality improves automatically — especially for topics not well covered in the original FarmBeats document (e.g. NDVI thresholds, specific crop varieties).

---

## Connection to rag_cli.py

```python
from rag_cli import AgriEmbedder, VectorStore, DB_PATH, chunk_text
```

The updater uses the same embedder, same vector store, and same DB path as the RAG pipeline. New chunks added by the updater are immediately available to the agent without any restart.

This is the critical line that connects Week 1 (RAG pipeline) with the knowledge updater — same ChromaDB, same BGE embedder, same everything.
