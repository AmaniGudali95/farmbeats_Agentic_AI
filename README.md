# FarmBeats Agentic AI Advisor

An end-to-end agentic AI system for precision agriculture, inspired by Microsoft's FarmBeats research program (IEEE Micro, 2022). The system combines retrieval-augmented generation over agricultural research with a ReAct agent that calls live data tools to give farmers specific, cited, actionable recommendations.

> **Inspired by:** Chandra et al., *"Democratizing Data-Driven Agriculture Using Affordable Hardware"*, IEEE Micro, January 2022

---

## The problem this solves

Smallholder farmers — who earn less than $2 a day and feed the majority of the world — cannot access data-driven agriculture tools. Existing solutions require expensive sensors, reliable internet, and technical expertise they don't have.

This system demonstrates how AI can bridge that gap: a farmer types a question in plain English and gets a specific, research-grounded recommendation based on their actual field conditions and real local weather.

```
Farmer asks: "Should I irrigate my corn today?"

System responds: "Irrigate today. Soil moisture is at 39% —
below the 50% field capacity threshold from FarmBeats research —
and no rain is forecast for the next 3 days."
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Farmer Interface                      │
│    Web UI · REST API · Daily Alerts · Feedback System   │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                   ReAct Agent Loop                       │
│                                                          │
│   Reason → Call Tool → Observe → Reason → ...           │
│                                                          │
│   Tools:                                                 │
│   ├── get_sensor_data()       IoT soil moisture + temp   │
│   ├── get_weather_forecast()  Real weather (GPS-based)   │
│   ├── get_ndvi_index()        Satellite crop health      │
│   └── search_farm_knowledge() RAG knowledge retrieval    │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                   RAG Pipeline                           │
│                                                          │
│   FarmBeats knowledge + Semantic Scholar weekly updates  │
│        ↓                                                 │
│   Chunked (400 words, 60-word overlap)                   │
│        ↓                                                 │
│   BGE embeddings (BAAI/bge-base-en-v1.5)                │
│        ↓                                                 │
│   ChromaDB vector store (cosine similarity / HNSW)       │
│        ↓                                                 │
│   Semantic retrieval at query time                       │
└─────────────────────────────────────────────────────────┘
```

---

## What was built — week by week

### Week 1 — RAG Pipeline

The FarmBeats research and agricultural documents are too long to include in every prompt. The RAG pipeline makes them searchable:

- **PDF ingestion** — extracts text from research papers using pypdf
- **Chunking** — splits text into 400-word windows with 60-word overlap
- **Embedding** — converts each chunk to a 768-dimensional vector using BGE
- **Vector storage** — stores vectors in ChromaDB with HNSW index
- **Retrieval** — at query time, embeds the question and finds the most semantically similar chunks

---

### Week 2 — ReAct Agent

```
Round 1: Claude reasons → calls get_sensor_data(field_a)
         Tool runs → soil_moisture: 39%

Round 2: Claude reasons → calls get_weather_forecast(days=3)
         Real Open-Meteo API called with farmer's GPS coordinates

Round 3: Claude reasons → calls search_farm_knowledge("corn irrigation")
         BGE embeds query → ChromaDB returns relevant passages

Round 4: Claude synthesises all data → cited recommendation
```

**Four tools:**
- `get_sensor_data()` — per-field soil moisture, temperature, humidity
- `get_weather_forecast()` — real weather via Open-Meteo API (GPS-based)
- `get_ndvi_index()` — crop health index (healthy above 0.5)
- `search_farm_knowledge()` — BGE semantic search over ChromaDB

**Reliability:** try/except on every tool, 5s timeout, 10-step limit, timestamped logging

**Daily alert system:**
- Soil moisture below 50% → irrigation alert (HIGH below 35%, MEDIUM 35-50%)
- Min temperature below 2°C in 3 days → frost alert
- NDVI below 0.5 → crop stress alert

---

### Week 3 — FastAPI + Farmer UI

```
POST /ask               → question + field_id + GPS → recommendation
GET  /alerts            → field-filtered alert status
GET  /sensor/{field_id} → live field conditions
GET  /weather?lat&lon   → location-based 3-day forecast
POST /feedback          → store farmer ratings (👍 👎)
GET  /ui                → farmer web interface
```

**UI features:**
- Field selector (A, B, C) with independent sensor data
- Plain English question input
- Real GPS-based weather forecast with rain/frost icons
- Field-filtered alerts — colour coded by severity
- Feedback system stored to `data/feedback.json`

---

### Knowledge Updater — weekly pipeline

```bash
python knowledge_updater.py            # run once
python knowledge_updater.py --schedule # run every Monday at 6am
```

- Searches Semantic Scholar for new agriculture papers (free, no key)
- Deduplicates using MD5 hash of paper ID
- Downloads full PDF text where available, falls back to abstract
- Chunks and embeds with BGE, adds to ChromaDB immediately
- Index grew from 7 → 9 chunks on first run

---

## Experiment results

### Experiment 1 — Chunk size sweep

Tested 6 chunk sizes on the FarmBeats document (2127 words):

```
Size    Chunks    Avg words    Notes
100     26        97           Too granular — NDVI explanation split across 4 chunks
200     13        194          Better but still fragmented
400     7         360          Sweet spot — balanced precision and context
600     5         505          Starts losing granularity
800     4         629          Too broad for precise retrieval
1200    3         818          Half the document per chunk — unusable
```

**Chosen:** 400 words, 60-word overlap (15%)

---

### Experiment 2 — The boundary problem

Splitting the sentence "An average farmer in Sub-Saharan Africa earns less than $2 per day" across a chunk boundary:

```
overlap=0:
  Chunk 0 ends:   "...An average farmer in Sub-Saharan Africa"
  Chunk 1 starts: "earns less than $2 per day..."
  → Subject and predicate in different chunks
  → Neither chunk retrieves correctly for "smallholder farmer income"

overlap=60:
  Complete sentence appears in at least one chunk
  → Retrieves correctly
```

**Finding:** Boundary splits reduced retrieval similarity by ~25%. Overlap is not optional.

---

### Experiment 3 — Embedding model comparison

Tested four embedding models on five sentence pairs:

```
                    Fallback    AgriScibert    SPECTER2    BGE
corn vs maize:      0.035       0.859          0.859       0.715
corn vs weather:    0.000       0.798          0.814       0.521
corn vs tractor:    0.000       0.748          0.736       0.517
corn vs stocks:     0.000       0.673          0.713       0.357

Discrimination gap (maize - stocks):
                    0.035       0.186          0.146       0.358
```

**Analysis:**
- **Fallback (hash-based):** Near-zero scores for everything — cannot distinguish meaning at all
- **AgriScibert:** Good at recognising corn=maize (0.859) but over-relates all agricultural topics — discrimination gap only 0.186
- **SPECTER2:** Similar problem to AgriScibert, worse discrimination gap (0.146)
- **BGE:** Best discrimination gap (0.358) — nearly double AgriScibert. Correctly identifies stocks as unrelated (0.357) while keeping corn/maize similarity high (0.715)

**Winner: BGE (BAAI/bge-base-en-v1.5)**

Key insight: AgriScibert and SPECTER2 were trained as masked language models — they learned what words appear together in agricultural text. BGE was trained with contrastive learning specifically for retrieval — explicitly trained to push similar sentences together and different sentences apart. This is why BGE discriminates better within a domain.

---

### Experiment 4 — ChromaDB vs brute force speed

```
7 chunks:
  ChromaDB HNSW:  0.044s for 100 queries
  Brute force:    0.233s for 100 queries
  ChromaDB is 5.3x faster

Extrapolated to 7 million chunks:
  Brute force:    ~4.47 hours per query
  ChromaDB HNSW:  ~4 seconds per query
```

**Finding:** Brute force is O(n) — time grows linearly with chunk count. HNSW is O(log n) — time grows logarithmically. The gap is negligible at 7 chunks but catastrophic at scale.

---

### Experiment 5 — Wrong embedder experiment

Queried ChromaDB with a random 1536-dimensional vector instead of a real embedding:

```
Results:
  score=0.026  "expensive to deploy in rural areas..."
  score=0.009  "in rural areas. Drone-based imaging..."
  score=-0.001 "components of the system on GitHub..."
```

**Finding:** ChromaDB returned results with no error — silent failure. A negative score means the random vector pointed in the opposite direction to that chunk in vector space (180° angle). Must use the same embedding model for both indexing and querying — ChromaDB cannot detect a mismatch.

---

### Experiment 6 — BGE retrieval quality

After switching to BGE and re-indexing:

```
Q: What soil moisture triggers corn irrigation?
  [1] score=0.695 — "falls below a threshold...system recommends irrigation" ✓

Q: What NDVI value indicates healthy crops?
  [1] score=0.726 — "FarmBeats sensor system measures soil properties" (partial)
  Note: NDVI-specific content not yet in knowledge base — will improve with weekly updates

Q: How does FarmBeats work without internet?
  [1] score=0.740 — "in rural areas. Drone-based imaging requires..."
  [2] score=0.736 — "farms lack reliable internet access. FarmBeats..."
  [3] score=0.734 — "internet unavailable, FarmBeats Hub stores locally" ✓
```

**Finding:** BGE retrieves the correct chunk first for irrigation and connectivity questions. NDVI retrieval is limited by knowledge base content, not embedder quality — will improve as weekly updater adds NDVI-specific papers.

---

## Key technical decisions and why

**Why BGE over AgriScibert?**
BGE's discrimination gap (0.358) is nearly double AgriScibert's (0.186). AgriScibert learned that all agricultural topics are related — correct conceptually but bad for retrieval. BGE was trained contrastively to distinguish between similar and dissimilar sentences regardless of domain.

**Why cosine similarity over L2 distance?**
Cosine measures the angle between vectors — ignores magnitude (text length), compares only direction (meaning). L2 penalises longer documents unfairly.

**Why 400-word chunks with 60-word overlap?**
Tested six sizes. 400 words balanced retrieval precision with context richness. 60-word overlap (15%) ensures key sentences appear completely in at least one chunk.

**Why one `run_tool()` dispatcher?**
The agent loop receives a tool name string from Claude. One dispatcher handles any name in one line. Adding a fifth tool requires only one `elif` — the agent loop never changes.

**Why messages grow each loop iteration?**
Claude has no memory between API calls. Complete history sent every call so Claude knows what tools it called and what results came back.

**Why GPS coordinates instead of hardcoded location?**
A farmer in Kenya and California get completely different weather. Browser Geolocation API provides accurate coordinates silently — no input from the farmer required.

**Why MD5 hash for deduplication?**
Semantic Scholar paper IDs contain `/` and `.` which ChromaDB rejects. MD5 produces a clean 8-character alphanumeric string. Deterministic — same paper always produces the same hash.

**Why `normalize_embeddings=True` for BGE?**
BGE is specifically designed for cosine similarity with normalised vectors. Without normalisation, dot product scores are not bounded between -1 and 1, making comparison unreliable.

---

## What this system cannot do yet

- **No real IoT sensors** — soil moisture simulated with realistic random variation per field
- **No real NDVI** — hardcoded at 0.67; production would use Sentinel-2 satellite API (free via Sentinel Hub)
- **No offline LLM** — requires Anthropic API for reasoning; production would use Ollama + Phi-3 Mini
- **No multilingual support yet** — English only; Claude responds natively in any language with one system prompt change
- **Abstract-only indexing for most papers** — full PDF download works where open access PDFs are available
- **No fine-tuned embedder** — BGE is general-purpose; fine-tuning on labelled agricultural sentence pairs could push discrimination gap above 0.5

---

## Connecting real data sources

**Real soil sensors (~$50):**
```python
# Raspberry Pi + capacitive moisture sensor
response = requests.get(f"http://raspberry-pi:5000/sensor/{field_id}")
# Or ThingSpeak free IoT platform
```

**Real NDVI (free):**
```
Sentinel Hub — sentinelhub.com
Sentinel-2 satellites: 10m resolution, updated every 5 days
Free research tier available
```

**Offline operation:**
```bash
brew install ollama
ollama pull phi3          # 2.4GB, runs on 4GB RAM
export USE_LOCAL_LLM=true
python week2_agent.py     # no internet, no API cost
```

**Multilingual:**
```python
run_agent(question="मिट्टी की नमी क्या है?", language="Hindi")
# BGE embeds in English, Claude responds in Hindi
```

**Fine-tuning BGE on agricultural data (Month 2 goal):**
```python
# Collect 2000-5000 labelled agricultural sentence pairs
# Fine-tune with LoRA — runs on Mac M2/M3
# Expected discrimination gap: 0.35 → 0.50+
```

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/AmaniGudali95/farmbeats_Agentic_AI
cd farmbeats_Agentic_AI/farmbeats_rag

# 2. Virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install
pip install anthropic chromadb pypdf fastapi uvicorn requests \
            schedule pydantic sentence-transformers torch peft

# 4. API key (optional — BGE embedder works without it)
export ANTHROPIC_API_KEY=sk-ant-...

# 5. Build knowledge base
python rag_cli.py --ingest

# 6. Update with latest research papers
python knowledge_updater.py

# 7. Start server
uvicorn api:app --reload

# 8. Open UI (allow location when prompted)
open http://127.0.0.1:8000/ui
```

---

## Project structure

```
farmbeats_rag/
├── rag_cli.py              Week 1 — RAG pipeline, ChromaDB, BGE embeddings
├── week2_agent.py          Week 2 — ReAct agent, tools, alerts, logging
├── api.py                  Week 3 — FastAPI endpoints, feedback storage
├── knowledge_updater.py    Ongoing — weekly Semantic Scholar updates
├── static/
│   └── index.html          Farmer UI with GPS weather
├── docs/
│   └── farmbeats_overview.txt  FarmBeats knowledge base
├── data/
│   ├── chroma_db/          ChromaDB vector store (generated)
│   └── feedback.json       Farmer feedback ratings (generated)
└── notes/
    ├── day2.md             Chunking experiment results
    ├── day3.md             Embedding model comparison
    ├── day4.md             ChromaDB speed test results
    └── week2_progress.md   Week 2 findings
```

---

## What I learned building this

**Chunk size is the most impactful RAG parameter.**
Tested 6 sizes. At 100 words the NDVI explanation split across 4 chunks. At 1200 only 3 chunks covered the entire document. 400 words balanced precision with context.

**The boundary problem is real and measurable.**
Splitting one sentence across a chunk boundary reduced retrieval similarity by ~25%. Overlap is not optional.

**Domain-specific models can over-fit the domain.**
AgriScibert knows corn and maize are related (0.859) but also thinks corn and weather forecasts are nearly as related (0.798). Trained on agricultural text without contrastive objectives, it learned "everything agricultural is similar." BGE's contrastive training gives better discrimination (gap: 0.358 vs 0.186).

**Silent failures are the most dangerous bugs.**
Wrong embedder → near-zero scores → wrong answers returned confidently with no error. Harder to find than a crash.

**The system prompt determines grounding more than anything else.**
"Answer using the retrieved passages" reduced hallucination more than tripling retrieved chunks.

**Brute force search does not scale.**
4.47 hours per query at 7M chunks. ChromaDB HNSW: 4 seconds. This is why vector databases exist.

**Agent loops need limits.**
Without a 10-step maximum, ambiguous questions loop indefinitely consuming tokens.

**JavaScript is case-sensitive and quote-type-sensitive.**
`userLat` and `userlat` are different variables. Single quotes `'${lat}'` send literal text — backticks evaluate it. Both bugs cause silent failures.

**Location matters for agriculture.**
Hardcoded coordinates give wrong weather for farmers in different countries. GPS coordinates silently provide real local forecasts with no input from the farmer.

**Labelled data beats raw scale.**
Fine-tuning BGE on 2000 labelled agricultural sentence pairs would likely outperform AgriScibert trained on millions of unlabelled documents. The right data matters more than volume.

---

## Next steps

- Fine-tune BGE on labelled agricultural sentence pairs (target: discrimination gap > 0.5)
- Sentinel Hub API for real NDVI from Sentinel-2 satellites
- Ollama + Phi-3 Mini for fully offline operation
- Multilingual support — language selector in UI
- Outcome tracking — follow up a week after recommendations
- JWT authentication and rate limiting for production
- Raspberry Pi deployment — test fully offline on edge hardware
- Voice input using Web Speech API for low-literacy farmers

---

## References

Chandra, R., Swaminathan, M., Chakraborty, T., Ding, J., Kapetanovic, Z., Kumar, P., & Vasisht, D. (2022). Democratizing Data-Driven Agriculture Using Affordable Hardware. *IEEE Micro*, 42(1), 69–77.

---

*Built as a learning project to understand agentic AI systems end to end — from PDF ingestion and vector embeddings through ReAct reasoning loops to a deployed farmer interface with real GPS-based weather, a weekly knowledge update pipeline, and a systematic embedding model evaluation.*
