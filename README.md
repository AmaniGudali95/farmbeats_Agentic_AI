# FarmBeats Agentic AI Advisor

An end-to-end agentic AI system for precision agriculture, inspired by Microsoft's FarmBeats research program (IEEE Micro, 2022). The system combines retrieval-augmented generation over agricultural research with a ReAct agent that calls live data tools to give farmers specific, cited, actionable recommendations.

> **Inspired by:** Chandra et al., *"Democratizing Data-Driven Agriculture Using Affordable Hardware"*, IEEE Micro, January 2022

---

## The problem this solves

Smallholder farmers — who earn less than $2 a day and feed the majority of the world — cannot access data-driven agriculture tools. Existing solutions require expensive sensors, reliable internet, and technical expertise they don't have.

This system demonstrates how AI can bridge that gap: a farmer types a question in plain English and gets a specific, research-grounded recommendation based on their actual field conditions and real local weather.

```
Farmer asks: "Should I irrigate my corn today?"

System responds: "Irrigate today. Soil moisture is at 43% —
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
│   59 chunks from multiple sources:                       │
│   · FarmBeats agricultural knowledge                     │
│   · FAO irrigation and crop water guides                 │
│   · USDA soil health documentation                       │
│   · UMN Extension corn growing guide                     │
│   · Peer-reviewed papers (Semantic Scholar)              │
│        ↓                                                 │
│   Chunked (400 words, 60-word overlap)                   │
│        ↓                                                 │
│   BGE embeddings (BAAI/bge-base-en-v1.5)                │
│   Runs locally — no API key, no cost, works offline      │
│        ↓                                                 │
│   ChromaDB vector store (cosine similarity / HNSW)       │
└─────────────────────────────────────────────────────────┘
```

---

## What was built — week by week

### Week 1 — RAG Pipeline

- **PDF ingestion** — extracts text from research papers using pypdf
- **Chunking** — splits text into 400-word windows with 60-word overlap
- **Embedding** — converts each chunk to a 768-dimensional vector using BGE
- **Vector storage** — stores vectors in ChromaDB with HNSW index
- **Retrieval** — embeds the question at query time, finds most similar chunks

---

### Week 2 — ReAct Agent

```
Round 1: Claude reasons → calls get_sensor_data(field_a)
         Returns: soil_moisture=43%, temperature=22°C

Round 2: Claude reasons → calls get_weather_forecast(days=7)
         Real Open-Meteo API → no rain for 7 days, temps rising to 29.9°C

Round 3: Claude reasons → calls search_farm_knowledge("corn irrigation")
         BGE retrieves → FarmBeats + UMN Extension passages

Round 4: Claude synthesises → cited recommendation with specific advice
```

**Real Claude output (first live run):**
> "Yes — irrigate today. Soil moisture at 43% is below the 50-70% optimal range for week 6 corn. No rain for 7 days. Temperatures rising to 29.9°C will increase water demand. Irrigate to 60-65% field capacity today. Monitor again around May 10-11."

**Four tools:**
- `get_sensor_data()` — per-field soil moisture, temperature, humidity with realistic random variation
- `get_weather_forecast()` — real weather via Open-Meteo API using farmer's GPS coordinates
- `get_ndvi_index()` — crop health index (healthy above 0.5)
- `search_farm_knowledge()` — BGE semantic search over 59-chunk knowledge base

**Reliability:** try/except on every tool, 5s timeout, 10-step limit, timestamped logging

**Daily alert system:**
- Soil moisture below 50% → irrigation alert (HIGH below 35%, MEDIUM 35-50%)
- Min temperature below 2°C in next 3 days → frost alert
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
- Real GPS-based weather — browser gets coordinates, fetches actual local forecast
- 3-day forecast with rain (🌧️) and frost (🌨️) icons
- Field-filtered alerts — amber for medium, red for high
- Feedback system stored to `data/feedback.json`

---

### Knowledge Updater — self-updating pipeline

Keeps the knowledge base current from two sources automatically:

**Source 1 — Semantic Scholar (academic papers)**
```bash
python knowledge_updater.py            # run once
python knowledge_updater.py --schedule # run every Monday at 6am
```

**Source 2 — Trusted agricultural websites**
Web scraper fetches content from curated sources:
- FAO irrigation management guides
- FAO crop water requirements
- USDA soil health documentation
- University Extension corn growing guides

Deduplication using MD5 hash of source ID — never indexes the same content twice.

**Knowledge base growth:**
```
Start (Week 1):    7 chunks   — FarmBeats overview only
After updater:    59 chunks   — 8.4x growth

Breakdown:
  FarmBeats overview:              7 chunks
  FAO irrigation management:      15 chunks
  FAO crop water requirements:    19 chunks
  USDA soil health:                7 chunks
  UMN corn growing guide:          2 chunks
  Semantic Scholar papers (6):     9 chunks
```

---

## Embedding model progression

### Stage 1 — Fallback hash embedder

Used during all Week 1 and Week 2 development (no API key available):

```
corn vs maize:   0.035  ← nearly zero despite same meaning
corn vs weather: 0.000
```

Cannot distinguish meaning — only detects shared words.

### Stage 2 — Anthropic embeddings

Supported in code but **never actually used** — no API key was available during development.

### Stage 3 — BGE (current, selected after systematic comparison)

```
BAAI/bge-base-en-v1.5
  Free, open source, runs offline after first download (~440MB)
  Trained with contrastive learning for retrieval
  Best discrimination gap: 0.358
  normalize_embeddings=True required
  768-dimensional vectors
```

---

## Experiment results

### Experiment 1 — Chunk size sweep

```
Size    Chunks    Avg words    Notes
100     26        97           Too granular — NDVI split across 4 chunks
200     13        194          Better but fragmented
400     7         360          Sweet spot
600     5         505          Losing granularity
800     4         629          Too broad
1200    3         818          Half the document per chunk
```

**Chosen: 400 words, 60-word overlap (15%)**

---

### Experiment 2 — The boundary problem

```
overlap=0:
  Chunk 0 ends:   "...An average farmer in Sub-Saharan Africa"
  Chunk 1 starts: "earns less than $2 per day..."
  → Sentence split — neither chunk retrieves the complete fact
  → ~25% retrieval similarity drop

overlap=60:
  Complete sentence appears in at least one chunk → retrieves correctly
```

---

### Experiment 3 — Embedding model comparison

```
                    Fallback    AgriScibert    SPECTER2    BGE
corn vs maize:      0.035       0.859          0.859       0.715
corn vs weather:    0.000       0.798          0.814       0.521
corn vs tractor:    0.000       0.748          0.736       0.517
corn vs stocks:     0.000       0.673          0.713       0.357

Discrimination gap (maize - stocks):
                    0.035       0.186          0.146       0.358
```

**Key finding:** AgriScibert and SPECTER2 were trained as masked language models — they learned what words appear together in agricultural text, causing them to over-relate all agricultural topics (gap: 0.186 and 0.146). BGE was trained with contrastive learning specifically for retrieval, giving nearly double the discrimination gap (0.358).

**Winner: BGE**

---

### Experiment 4 — ChromaDB vs brute force

```
7 chunks:
  ChromaDB HNSW:  0.044s for 100 queries
  Brute force:    0.233s for 100 queries  →  5.3x faster

Extrapolated to 7 million chunks:
  Brute force:    ~4.47 hours per query
  ChromaDB HNSW:  ~4 seconds per query
```

---

### Experiment 5 — Wrong embedder silent failure

Random vector query → scores near zero, one negative (-0.001). No error thrown. Must use same embedding model for indexing and querying — ChromaDB cannot detect a mismatch.

---

### Experiment 6 — BGE retrieval quality (59 chunks)

```
Q: What soil moisture triggers corn irrigation?
  [1] score=0.695 — FarmBeats irrigation threshold ✓
  [2] score=0.695 — UMN Extension corn guide ✓  ← tied for first
  [3] score=0.690 — IoT Irrigation System paper ✓

Q: What NDVI value indicates healthy crops?
  [2] score=0.692 — Satellite imagery + deep learning paper
  [3] score=0.684 — Landsat-8 NDVI classification paper

Q: How does FarmBeats work without internet?
  [3] score=0.734 — "FarmBeats Hub stores locally and syncs when connected" ✓
```

Two independent sources (FarmBeats research + UMN Extension) now tied for first on the irrigation question — demonstrating multi-source retrieval working correctly.

---

## Key technical decisions and why

**Why BGE over AgriScibert?**
BGE's discrimination gap (0.358) is nearly double AgriScibert's (0.186). Contrastive training teaches the model to distinguish similar from dissimilar sentences — masked language modelling does not.

**Why not Anthropic embeddings?**
BGE runs completely offline, is free forever, and was available without an API key during development. For production at scale, local embeddings are essential for cost control.

**Why 400-word chunks with 60-word overlap?**
Tested six sizes. 400 words balanced retrieval precision with context. 60-word overlap ensures key sentences appear completely in at least one chunk.

**Why web scraping alongside Semantic Scholar?**
Academic papers cover general principles. USDA and FAO guides cover specific thresholds and practical advice — exactly what farmers need. Both sources are necessary for comprehensive retrieval.

**Why MD5 hash for deduplication?**
Source IDs contain special characters ChromaDB rejects. MD5 produces a clean 8-character alphanumeric string. Deterministic — same source always produces the same hash.

**Why GPS coordinates instead of hardcoded location?**
A farmer in Kenya and California get completely different weather. Browser Geolocation API provides accurate coordinates silently.

**Why messages grow each loop iteration?**
Claude has no memory between API calls. Complete history sent every call so Claude knows what tools it called and what came back.

---

## What this system cannot do yet

- **No real IoT sensors** — soil moisture simulated per field
- **No real NDVI** — hardcoded at 0.67; production would use Sentinel-2 API
- **No offline LLM** — requires Anthropic API; production would use Ollama + Phi-3 Mini
- **No multilingual support** — English only; Claude responds natively in any language with one system prompt change
- **No fine-tuned embedder** — BGE general-purpose; fine-tuning on labelled agricultural pairs could push gap above 0.5

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
            schedule pydantic sentence-transformers torch peft \
            beautifulsoup4

# 4. API key (optional — BGE embedder works without it)
export ANTHROPIC_API_KEY=sk-ant-...

# 5. Build knowledge base
python rag_cli.py --ingest

# 6. Expand with FAO, USDA, and latest research papers
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
├── knowledge_updater.py    Ongoing — Semantic Scholar + web scraper
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
    ├── week2_progress.md   Week 2 findings + real Claude output
    ├── week3.md            FastAPI + UI decisions
    └── knowledge_updater.md  Weekly pipeline findings
```

---

## What I learned building this

**The embedding model used during development was always the fallback.**
No Anthropic API key was available, so all Week 1 and Week 2 development used the hash-based fallback. The comparison experiment revealed the full gap — 0.035 vs 0.715 for corn/maize.

**Domain-specific models can over-fit the domain.**
AgriScibert correctly identifies corn=maize (0.859) but treats corn and weather as nearly as similar (0.798). Contrastive training — not domain data — is what produces good retrieval discrimination.

**Multi-source knowledge bases retrieve better.**
After adding FAO, USDA, and UMN Extension content, two independent sources tied for first place on the irrigation question. Single-source knowledge bases have blind spots that multi-source retrieval fills.

**Silent failures are the most dangerous bugs.**
Wrong embedder, wrong quote type in JavaScript, wrong variable case — all caused incorrect behaviour with no error. Harder to find than crashes.

**Brute force search does not scale.**
4.47 hours per query at 7M chunks. ChromaDB HNSW: 4 seconds. This is why vector databases exist.

**Labelled data beats raw scale.**
Fine-tuning BGE on 2000 labelled agricultural pairs would likely outperform AgriScibert trained on millions of unlabelled documents.

---

## Next steps

- Fine-tune BGE on labelled agricultural sentence pairs (target: gap > 0.5)
- Sentinel Hub API for real NDVI from Sentinel-2 satellites
- Ollama + Phi-3 Mini for fully offline operation on edge hardware
- Multilingual support — language selector in UI
- JWT authentication and rate limiting for production
- Outcome tracking — follow up a week after recommendations
- Voice input using Web Speech API for low-literacy farmers

---

## References

Chandra, R., Swaminathan, M., Chakraborty, T., Ding, J., Kapetanovic, Z., Kumar, P., & Vasisht, D. (2022). Democratizing Data-Driven Agriculture Using Affordable Hardware. *IEEE Micro*, 42(1), 69–77.

---

*Built as a learning project to understand agentic AI systems end to end — RAG pipeline with BGE embeddings, ReAct agent with live tools, FastAPI farmer interface, and a self-updating knowledge base that grew from 7 to 59 chunks across FarmBeats research, FAO guides, USDA documentation, and peer-reviewed papers.*
