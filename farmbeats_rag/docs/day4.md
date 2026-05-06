# Day 4 — ChromaDB Internals

## What ChromaDB stores per chunk

Four things stored for each chunk:

| Field | Example | Purpose |
|---|---|---|
| ID | `farmbeats__chunk_0` | Unique identifier, used for deduplication |
| Vector | `[0.23, -0.41, 0.87, ...]` | 768 numbers encoding meaning |
| Text | `"FarmBeats: Democratizing..."` | Original text returned with results |
| Metadata | `{"source": "farmbeats", "chunk_index": 0}` | Source tracking for citations |

---

## HNSW — why ChromaDB is fast

Brute force search: compare query vector against every stored vector. O(n) — time grows linearly with chunk count.

HNSW (Hierarchical Navigable Small World): builds a graph where each vector connects to its nearest neighbours. Search hops greedily toward the query vector. O(log n) — time grows logarithmically.

---

## Speed experiment

```
7 chunks:
  ChromaDB HNSW:  0.044s for 100 queries
  Brute force:    0.233s for 100 queries
  ChromaDB is 5.3x faster at 7 chunks

Extrapolated to 7 million chunks:
  Brute force:    ~4.47 hours per query
  ChromaDB HNSW:  ~4 seconds per query
```

At 7 chunks the difference is small in absolute terms. At scale it's the difference between a usable system and an unusable one.

---

## Wrong embedder experiment

Queried ChromaDB with a random 1536-dimensional vector instead of a real embedding:

```
score=0.026   "expensive to deploy in rural areas..."
score=0.009   "in rural areas. Drone-based imaging..."
score=-0.001  "components of the system on GitHub..."
```

**ChromaDB returned results with no error** — silent failure. The -0.001 score means the random vector pointed in nearly the opposite direction (180° angle) to that chunk.

**Rule:** Must use the same embedding model for both indexing and querying. ChromaDB cannot detect a mismatch.

---

## `metadata={"hnsw:space": "cosine"}`

Tells ChromaDB which distance metric to use for the HNSW index. Options:
- `cosine` — measures angle between vectors (best for text)
- `l2` — straight-line distance (penalises longer documents)
- `ip` — inner product (dot product without normalisation)

Setting this incorrectly produces wrong results with no error — another silent failure mode.

---

## Key rules

1. Delete `data/chroma_db/` and re-ingest when switching embedding models
2. Same embedder for indexing and querying — always
3. `hnsw:space: cosine` for text retrieval — always
4. Chunk 0 always exists if a document was indexed — use it for deduplication checks
