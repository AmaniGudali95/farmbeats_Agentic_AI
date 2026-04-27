# farmbeats_Agentic_AI

# FarmBeats Agentic AI Advisor

An end-to-end agentic AI system for precision agriculture, inspired by Microsoft's FarmBeats research program (IEEE Micro, 2022). The system combines retrieval-augmented generation over agricultural research with a ReAct agent that calls live data tools to give farmers specific, cited, actionable recommendations.

> **Inspired by:** Chandra et al., *"Democratizing Data-Driven Agriculture Using Affordable Hardware"*, IEEE Micro, January 2022

---

## The problem this solves

Smallholder farmers — who earn less than $2 a day and feed the majority of the world — cannot access data-driven agriculture tools. Existing solutions require expensive sensors, reliable internet, and technical expertise they don't have.

This system demonstrates how AI can bridge that gap: a farmer types a question in plain English and gets a specific, research-grounded recommendation based on their actual field conditions.

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
│         Web UI  ·  REST API  ·  Daily Alerts            │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                   ReAct Agent Loop                       │
│                                                          │
│   Reason → Call Tool → Observe → Reason → ...           │
│                                                          │
│   Tools:                                                 │
│   ├── get_sensor_data()      IoT soil moisture + temp    │
│   ├── get_weather_forecast() Open-Meteo 7-day forecast   │
│   ├── get_ndvi_index()       Satellite crop health       │
│   └── search_farm_knowledge() RAG knowledge retrieval    │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                   RAG Pipeline                           │
│                                                          │
│   FarmBeats paper + new research papers                  │
│        ↓                                                 │
│   Chunked (400 words, 60-word overlap)                   │
│        ↓                                                 │
│   Embedded (Anthropic text-embedding-3-small)            │
│        ↓                                                 │
│   ChromaDB vector store (cosine similarity)              │
│        ↓                                                 │
│   Semantic retrieval at query time                       │
└─────────────────────────────────────────────────────────┘
```

---

## What was built — week by week

### Week 1 — RAG Pipeline

The FarmBeats paper and agricultural research documents are too long to include in every prompt. The RAG pipeline makes them searchable:

- **PDF ingestion** — extracts text from research papers using pypdf
- **Chunking** — splits text into 400-word windows with 60-word overlap to prevent key facts being split across boundaries
- **Embedding** — converts each chunk to a 1536-dimensional vector using the Anthropic embeddings API
- **Vector storage** — stores vectors in ChromaDB with HNSW index for fast cosine similarity search
- **Retrieval** — at query time, embeds the question and finds the most semantically similar chunks

**Key experiment:** Splitting a key sentence ("corn should be irrigated at 50% field capacity") across a chunk boundary reduced its retrieval similarity score by 25% — proving why overlap is essential, not optional.

**Key numbers from experiments:**
- Chunk size 400 words: 7 chunks from the FarmBeats paper
- Brute force search at 7M chunks: ~4.47 hours per query
- ChromaDB HNSW at 7M chunks: ~4 seconds per query — 5.3x faster even at 7 chunks

---

### Week 2 — ReAct Agent

A single RAG pipeline can only answer questions from research. A farmer asking "should I irrigate today?" needs live data — not just what the paper says about thresholds.

The ReAct (Reason + Act) agent loop:

```
Round 1: Claude reasons → calls get_sensor_data(field_a)
         You run the tool → soil_moisture: 39%
         Result sent back to Claude

Round 2: Claude reasons → calls get_weather_forecast(days=3)
         You run the tool → no rain for 3 days
         Result sent back to Claude

Round 3: Claude reasons → calls search_farm_knowledge("corn irrigation threshold")
         RAG pipeline runs → returns FarmBeats research passages
         Result sent back to Claude

Round 4: Claude has all data → produces final cited recommendation
```

**Four tools implemented:**
- `get_sensor_data()` — per-field soil moisture, temperature, humidity with realistic random variation
- `get_weather_forecast()` — real weather data via Open-Meteo API (free, no key needed)
- `get_ndvi_index()` — crop health index (0-1, healthy above 0.5)
- `search_farm_knowledge()` — semantic search over ChromaDB knowledge base

**Reliability features:**
- `try/except` error handling on every tool
- 5-second timeout on external API calls
- 10-step maximum loop limit with helpful fallback message
- Timestamped logging of every tool call and result

**Daily alert system:**
Runs every morning across all fields. Checks three conditions:
- Soil moisture below 50% field capacity → irrigation alert
- Minimum temperature below 2°C in next 3 days → frost alert
- NDVI below 0.5 → crop stress alert

Only sends alerts when action is required. Severity levels: HIGH (below 35% moisture) and MEDIUM (35-50%).

---

### Week 3 — FastAPI + Farmer UI

The agent becomes accessible to farmers through a REST API and web interface:

**Endpoints:**
```
POST /ask              → question + field_id → recommendation + sensor data
GET  /alerts           → field-filtered alert status
GET  /sensor/{field_id}→ live field conditions
GET  /weather          → 3-day forecast
GET  /ui               → farmer web interface
```

**Farmer interface features:**
- Field selector (A, B, C) — each with independent sensor data
- Plain English question input
- Recommendation with reasoning
- Live sensor dashboard (moisture, temperature, humidity)
- Field-filtered alerts (amber for medium, red for high)
- Feedback system (👍 👎) — stores ratings to `data/feedback.json`

**Technology:** FastAPI + uvicorn + Pydantic validation + vanilla HTML/CSS/JavaScript

---

### Knowledge updater (ongoing)

Weekly pipeline that keeps the knowledge base current:

- Searches Semantic Scholar API for new agriculture papers (free, no key)
- Deduplicates using MD5 hash of paper ID — never indexes the same paper twice
- Chunks and embeds abstracts using the same pipeline as Week 1
- Adds to the same ChromaDB index — immediately available to the agent
- Runs on a schedule: every Monday at 6am

```bash
# Run once
python knowledge_updater.py

# Run on weekly schedule
python knowledge_updater.py --schedule
```

---

## Key technical decisions and why

**Why cosine similarity over L2 distance?**
Cosine similarity measures the angle between vectors — it ignores vector magnitude (which correlates with text length) and compares only direction (which represents meaning). L2 distance penalises longer documents unfairly.

**Why 400-word chunks with 60-word overlap?**
Tested chunk sizes from 100 to 1200 words. Size 100 was too granular — the NDVI explanation split across 4 chunks. Size 800 retrieved noisily. 400 words balanced precision and context. 60-word overlap (15%) ensures key sentences appear completely in at least one chunk.

**Why one `run_tool()` dispatcher instead of separate functions?**
The agent loop receives a tool name string from Claude (`block.name`). A single dispatcher handles any tool name in one line: `result = run_tool(block.name, block.input)`. Adding a fifth tool requires only adding an `elif` to `run_tool` — the agent loop never changes.

**Why messages grow each loop iteration?**
Claude has no memory between API calls. Every call sends the complete conversation history so Claude knows what tools it called and what results came back. Without this, Claude would have no context for the tool results it receives.

**Why put retrieved passages before the question in the prompt?**
Claude reads sequentially. Putting evidence first means Claude is primed with facts before forming an answer. Evidence after the question risks Claude starting to answer from training knowledge before reaching the retrieved research.

---

## What this system cannot do yet

Being honest about limitations:

- **No real IoT sensors** — soil moisture is simulated with realistic random variation per field
- **No real NDVI** — hardcoded at 0.67; production would use Sentinel-2 satellite API
- **Fallback embedder quality** — without an API key, hash-based embeddings give corn vs maize a similarity of 0.035 instead of ~0.85 with real embeddings
- **No fine-tuning** — uses a general-purpose LLM; a model fine-tuned on agronomic data would give better recommendations
- **No offline mode** — requires internet for Claude API and weather data; production deployment in low-connectivity areas would need local model (Ollama + Phi-3) and cached weather

---

## Connecting real data sources

**For real soil sensors:**
```python
# Replace mock data in get_sensor_data() with:
response = requests.get(
    f"http://your-raspberry-pi:5000/sensor/{field_id}"
)
# A $50 capacitive sensor + Raspberry Pi gives real readings
```

**For real NDVI:**
```python
# Sentinel Hub API — free tier available
# European Space Agency Sentinel-2 satellites
# 10m resolution, updated every 5 days, covers entire Earth
```

**For offline operation:**
```python
# Replace Claude with local Llama 3 or Phi-3 via Ollama
# No internet required, no API cost
# Lower reasoning quality but fully offline
export USE_LOCAL_LLM=true
python week2_agent.py
```

---

## Quick start

**Requirements:** Python 3.12, pip

```bash
# 1. Clone the repository
git clone https://github.com/yourname/farmbeats-agent
cd farmbeats-agent

# 2. Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install anthropic chromadb pypdf fastapi uvicorn requests schedule pydantic

# 4. Set API key (optional — fallback embedder works without it)
export ANTHROPIC_API_KEY=sk-ant-...

# 5. Build the knowledge base
python rag_cli.py --ingest

# 6. Start the API server
uvicorn api:app --reload

# 7. Open the farmer UI
open http://127.0.0.1:8000/ui
```

---

## Project structure

```
farmbeats-agent/
├── rag_cli.py              Week 1 — RAG pipeline, ChromaDB, embeddings
├── week2_agent.py          Week 2 — ReAct agent, tools, alerts, logging
├── api.py                  Week 3 — FastAPI endpoints
├── knowledge_updater.py    Ongoing — weekly knowledge base updates
├── static/
│   └── index.html          Farmer web interface
├── docs/
│   └── farmbeats_overview.txt  FarmBeats paper knowledge base
├── data/
│   ├── chroma_db/          ChromaDB vector store (generated)
│   └── feedback.json       Farmer feedback ratings (generated)
└── notes/
    ├── day2.md             Chunking experiment results
    ├── day3.md             Embedding experiment results
    ├── day4.md             ChromaDB speed test results
    └── week2_progress.md   Week 2 findings
```

---

## What I learned building this

**Chunk size is the most impactful RAG parameter.**
Tested 6 sizes from 100 to 1200 words. At size 100 the NDVI explanation split across 4 chunks. At size 1200 only 3 chunks covered the entire paper. 400 words balanced retrieval precision with context richness.

**The boundary problem is real and measurable.**
Splitting the sentence "corn should be irrigated at 50% field capacity" across a chunk boundary reduced its retrieval similarity score by 25% — from 0.847 to 0.631. Overlap is not optional.

**Silent failures are the most dangerous bugs.**
When ChromaDB is queried with a random vector (wrong embedder), it returns results with scores near zero and no error. The system confidently returns wrong answers. This is harder to debug than a crash.

**The system prompt determines grounding more than anything else.**
"Answer using the retrieved passages" reduced hallucination more than tripling the number of retrieved chunks. Claude's instruction-following is the primary defence against confident wrong answers.

**Brute force search does not scale.**
At 7 chunks, brute force cosine search takes 0.233 seconds for 100 queries. Extrapolating to 7 million chunks: 4.47 hours per query. ChromaDB's HNSW index answers the same query in ~4 seconds. This is why vector databases exist.

**Agent loops need limits.**
Without a 10-step maximum, a poorly worded question can cause Claude to loop indefinitely calling tools, consuming tokens and never returning an answer. The limit plus a helpful fallback message is the minimum viable reliability feature.

---

## Next steps

- Connect Sentinel Hub API for real NDVI data
- Add Ollama + Phi-3 for fully offline operation in low-connectivity areas
- Fine-tune a small model on agronomic decision data
- Add outcome tracking — follow up with farmers a week after recommendations
- Expand knowledge base with crop-specific guides (wheat, potato, millet)
- Add multilingual support — Claude responds in farmer's language natively

---

## References

Chandra, R., Swaminathan, M., Chakraborty, T., Ding, J., Kapetanovic, Z., Kumar, P., & Vasisht, D. (2022). Democratizing Data-Driven Agriculture Using Affordable Hardware. *IEEE Micro*, 42(1), 69–77.

---

*Built as a learning project to understand agentic AI systems end to end — from PDF ingestion and vector embeddings through ReAct reasoning loops to a deployed farmer interface.*
