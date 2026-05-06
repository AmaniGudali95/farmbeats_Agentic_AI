# Day 3 — Embedding Experiments

## What is an embedding?

An embedding converts text into a list of numbers (a vector) so that similar meanings produce similar vectors. The position of that vector in high-dimensional space encodes the meaning of the text.

- Texts with similar meanings → nearby positions → high cosine similarity
- Texts with different meanings → distant positions → low cosine similarity

---

## Cosine similarity

Measures the angle between two vectors:
- Angle 0° → similarity 1.0 → identical meaning
- Angle 90° → similarity 0.0 → completely unrelated
- Angle 180° → similarity -1.0 → opposite meaning

Formula:
```
similarity = dot(A, B) / (|A| × |B|)
```

---

## Experiment results — fallback hash embedder

```python
sentences = [
    "corn needs water when soil is dry",
    "irrigate maize at 50 percent field capacity",
    "the weather forecast shows rain tomorrow"
]

corn vs maize:   0.035  ← nearly zero despite same meaning
corn vs weather: 0.000  ← zero
```

The fallback embedder only detects shared words — "corn" and "maize" share no words so similarity ≈ 0. This is the embedder used throughout Week 1 and Week 2 development because no Anthropic API key was available.

---

## Embedding model comparison

Tested four models on five sentence pairs:

```
                    Fallback    AgriScibert    SPECTER2    BGE
corn vs maize:      0.035       0.859          0.859       0.715
corn vs weather:    0.000       0.798          0.814       0.521
corn vs tractor:    0.000       0.748          0.736       0.517
corn vs stocks:     0.000       0.673          0.713       0.357

Discrimination gap (maize score - stocks score):
                    0.035       0.186          0.146       0.358
```

### Why BGE wins

AgriScibert and SPECTER2 were trained as masked language models — they learned what words appear together in text. They correctly identify corn=maize (0.859) but over-relate all agricultural topics because they appear in the same domain.

BGE (BAAI/bge-base-en-v1.5) was trained with contrastive learning specifically for retrieval — explicitly trained to push similar sentences together and dissimilar ones apart. This gives nearly double the discrimination gap (0.358 vs 0.186).

### Key insight

Domain-specific models trained without contrastive objectives over-relate topics within a domain. A model knowing "corn" and "weather" both appear in agricultural text will score them similarly — bad for retrieval. A model trained to distinguish similar from dissimilar retrieves correctly.

---

## BGE retrieval test results

After switching to BGE and re-indexing (9 chunks):

```
Q: What soil moisture triggers corn irrigation?
  [1] score=0.695 — "falls below a threshold...system recommends irrigation" ✓

Q: What NDVI value indicates healthy crops?
  [1] score=0.726 — "FarmBeats sensor system measures soil properties"
  Note: NDVI gap is knowledge base problem, not embedder problem

Q: How does FarmBeats work without internet?
  [3] score=0.734 — "internet unavailable, FarmBeats Hub stores locally" ✓
```

---

## Final choice

**BGE (BAAI/bge-base-en-v1.5)**
- Free and open source
- Runs completely offline after first download (~440MB)
- No API key needed
- Best discrimination gap: 0.358
- Requires `normalize_embeddings=True` for accurate cosine similarity
- 768-dimensional vectors

---

## Next step (Month 2)

Fine-tune BGE on 2000-5000 labelled agricultural sentence pairs.
Expected discrimination gap: 0.358 → 0.50+
Method: LoRA fine-tuning, runs on Mac M2/M3
