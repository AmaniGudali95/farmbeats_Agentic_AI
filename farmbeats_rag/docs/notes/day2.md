# Day 2 — Chunking Experiments

## Document
- File: `farmbeats_overview.txt`
- Total words: 2127
- Total chunks at size=400, overlap=60: **7 chunks**

---

## Exercise 1 — Chunk inspection

Chunk 0 starts with the paper header — "FarmBeats: Democratizing Data-Driven Agriculture..."
Chunk 6 is the short trailing fragment (87 words) — kept because above the 20-word minimum guard.

Key observation: header metadata ends up in chunk 0's embedding — minor noise for retrieval.

---

## Exercise 2 — Chunk size sweep

```
Size    Chunks    Avg words    Min    Max    Notes
100     26        97           27     100    Too granular — NDVI split across 4 chunks
200     13        194          123    200    Better but still fragmented
400     7         360          123    400    Sweet spot
600     5         505          127    600    Losing granularity
800     4         629          126    800    Too broad
1200    3         818          127    1200   Half the document per chunk
```

The Min column stays around 123-127 for sizes 200+ — that's the same trailing fragment regardless of chunk size.

**Chosen: 400 words**

---

## Exercise 3 — Overlap test

```
Overlap    Ratio    Chunks    Redundant words
0          0%       6         0
40         10%      6         200
80         20%      7         480
120        30%      8         840
200        50%      11        1927
```

At 50% overlap: 1927 redundant words stored — nearly the entire document a second time.

### Boundary comparison

**overlap=0:**
```
End of chunk 0:   "...more dire for smallholder farmers. An average farmer in Sub-Saharan Africa"
Start of chunk 1: "earns less than $2 per day. For these farmers..."
```
The sentence "An average farmer in Sub-Saharan Africa earns less than $2 per day" is split.
Subject in chunk 0, predicate in chunk 1. Neither chunk retrieves the complete fact.

**overlap=120:**
The chunks step differently — the same sentence appears completely in at least one chunk.

---

## Conclusions

- Chunk size 400 words is the sweet spot for this 2127-word document
- 60-word overlap (15%) catches boundary splits without excessive redundancy
- Overlap is not optional — a split sentence loses ~25% retrieval similarity
- The trailing fragment (chunk 6, 87 words) is kept — above the 20-word minimum

**Chosen settings: chunk_size=400, overlap=60**
