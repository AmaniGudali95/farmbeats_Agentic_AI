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
│   FarmBeats paper + Semantic Scholar weekly updates      │
│        ↓                                                 │
│   Chunked (400 words, 60-word overlap)                   │
│        ↓                                                 │
│   Embedded (Anthropic text-embedding-3-small)            │
│        ↓                                                 │
│   ChromaDB vector store (cosine similarity / HNSW)       │
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
- Fallback embedder similarity for corn vs maize: 0.035 (real embeddings: ~0.85)
- Brute force search at 7M chunks: ~4.47 hours per query
- ChromaDB HNSW at 7M chunks: ~4 seconds per query — 5.3x faster even at 7 chunks

---

### Week 2 — ReAct Agent

A single RAG pipeline can only answer questions from research. A farmer asking "should I irrigate today?" needs live data — not just what the paper says about thresholds.

The ReAct (Reason + Act) agent loop:

```
Round 1: Claude reasons → calls get_sensor_data(field_a)
         Tool runs → soil_moisture: 39%
         Result sent back to Claude

Round 2: Claude reasons → calls get_weather_forecast(days=3)
         Real Open-Meteo API called with farmer's GPS coordinates
         Result sent back to Claude

Round 3: Claude reasons → calls search_farm_knowledge("corn irrigation threshold")
         RAG pipeline runs → returns FarmBeats research passages
         Result sent back to Claude

Round 4: Claude has all data → produces final cited recommendation
```

**Four tools implemented:**
- `get_sensor_data()` — per-field soil moisture, temperature, humidity with realistic random variation
- `get_weather_forecast()` — real weather data via Open-Meteo API using farmer's actual GPS location
- `get_ndvi_index()` — crop health index (0-1, healthy above 0.5)
- `search_farm_knowledge()` — semantic search over ChromaDB knowledge base

**Reliability features:**
- `try/except` error handling on every tool
- 5-second timeout on all external API calls
- 10-step maximum loop limit with helpful fallback message
- Timestamped logging of every tool call, result, and step

**Daily alert system** checks three conditions across all fields:
- Soil moisture below 50% field capacity → irrigation alert (HIGH below 35%, MEDIUM 35-50%)
- Minimum temperature below 2°C in next 3 days → frost alert
- NDVI below 0.5 → crop stress alert

---

### Week 3 — FastAPI + Farmer UI

**Endpoints:**
```
POST /ask               → question + field_id + GPS coords → recommendation
GET  /alerts            → field-filtered alert status
GET  /sensor/{field_id} → live field conditions
GET  /weather?lat&lon   → location-based 3-day forecast
POST /feedback          → store farmer ratings
GET  /ui                → farmer web interface
```

**Farmer interface features:**
- Field selector (A, B, C) — each with independent sensor data and alerts
- Plain English question input
- Real-time recommendation with cited reasoning
- Live sensor dashboard (moisture, temperature, humidity)
- **Real location-based weather** — browser GPS → actual local forecast
- 3-day weather with rain (🌧️) and frost (🌨️) icons
- Field-filtered alerts — Field B shows "all clear", Field A and C show irrigation warnings
- Feedback system (👍 👎) — ratings stored to `data/feedback.json`

---

### Knowledge Updater — weekly pipeline

Keeps the knowledge base current without manual intervention:

- Searches Semantic Scholar API for new agriculture papers (free, no key)
- Four search queries: precision agriculture, NDVI, FarmBeats, irrigation ML
- Deduplicates using MD5 hash of paper ID — never indexes the same paper twice
- Chunks and embeds abstracts using the same Week 1 pipeline
- Adds to the same ChromaDB index — immediately available to the agent

```bash
python knowledge_updater.py            # run once
python knowledge_updater.py --schedule # run every Monday at 6am
```

---

### Feedback Learning System

**Level 1 — Explicit ratings (implemented)**
Farmers rate 👍 or 👎 after every recommendation. Stored with question, field, and recommendation for analysis.

**Level 2 — Outcome tracking (planned)**
Follow up a week later to check if crop health improved.

**Level 3 — Farmer observations as knowledge (planned)**
Farmer-contributed field observations added directly to ChromaDB. System learns from real outcomes, not just published research.

---

### Location-based weather

```javascript
// Browser gets real GPS coordinates
navigator.geolocation.getCurrentPosition(function(position) {
    userLat = position.coords.latitude;
    userLon = position.coords.longitude;
    loadWeather(userLat, userLon);
});
```

```python
# FastAPI passes coordinates to Open-Meteo
@app.get("/weather")
def weather(lat: float = 37.7749, lon: float = -122.4194):
    data = run_tool("get_weather_forecast", {"days": 3, "lat": lat, "lon": lon})
    return data
```

`"timezone": "auto"` in the Open-Meteo call automatically detects the correct timezone — a farmer in India gets IST, a farmer in Kenya gets EAT.

---

## Key technical decisions and why

**Why cosine similarity over L2 distance?**
Cosine measures the angle between vectors — ignores magnitude (text length), compares only direction (meaning). L2 penalises longer documents unfairly.

**Why 400-word chunks with 60-word overlap?**
Tested 100 to 1200 words. Size 100 split the NDVI explanation across 4 chunks. Size 800 retrieved noisily. 400 words balanced precision and context. 15% overlap ensures key sentences appear completely in at least one chunk.

**Why one `run_tool()` dispatcher?**
The agent loop receives a tool name string from Claude. One dispatcher handles any name in one line: `result = run_tool(block.name, block.input)`. Adding a fifth tool requires only an `elif` — the agent loop never changes.

**Why messages grow each loop iteration?**
Claude has no memory between API calls. Every call sends complete conversation history so Claude knows what tools it called and what results came back.

**Why GPS instead of hardcoded location?**
A farmer in Kenya and a farmer in California get completely different weather. The browser Geolocation API provides accurate coordinates silently — the farmer never types their location.

**Why MD5 hash for deduplication?**
Semantic Scholar paper IDs contain `/` and `.` which ChromaDB rejects in chunk IDs. MD5 converts any ID into a clean 8-character alphanumeric string. Deterministic — same paper always produces the same hash.

---

## What this system cannot do yet

- **No real IoT sensors** — soil moisture simulated with realistic random variation per field
- **No real NDVI** — hardcoded at 0.67; production would use Sentinel-2 satellite API
- **Fallback embedder** — without API key, hash embeddings give corn vs maize similarity 0.035 vs ~0.85 real
- **No offline mode** — needs internet for Claude API and weather; production needs Ollama + Phi-3
- **No multilingual support yet** — English only; Claude responds natively in any language with one system prompt change
- **Abstract-only indexing** — knowledge updater indexes abstracts only, not full paper text

---

## Connecting real data sources

**Real soil sensors (~$50 hardware):**
```python
response = requests.get(f"http://raspberry-pi:5000/sensor/{field_id}")
# Capacitive soil moisture sensor + Raspberry Pi
# Or ThingSpeak free IoT platform — no server needed
```

**Real NDVI (free):**
```
Sentinel Hub — sentinelhub.com
European Space Agency Sentinel-2 satellites
10m resolution, updated every 5 days, covers entire Earth
Free research tier available
```

**Offline operation:**
```bash
brew install ollama
ollama pull phi3          # runs on 4GB RAM
export USE_LOCAL_LLM=true
python week2_agent.py     # no internet, no API cost
```

**Multilingual (one line change):**
```python
run_agent(question="मिट्टी की नमी क्या है?", language="Hindi")
# Claude reads English research, responds in Hindi
```

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/yourname/farmbeats-agent
cd farmbeats-agent

# 2. Virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install
pip install anthropic chromadb pypdf fastapi uvicorn requests schedule pydantic

# 4. API key (optional — fallback embedder works without it)
export ANTHROPIC_API_KEY=sk-ant-...

# 5. Build knowledge base
python rag_cli.py --ingest

# 6. Start server
uvicorn api:app --reload

# 7. Open UI (allow location when prompted)
open http://127.0.0.1:8000/ui

# 8. Update knowledge base with latest research
python knowledge_updater.py
```

---

## Project structure

```
farmbeats-agent/
├── rag_cli.py              Week 1 — RAG pipeline, ChromaDB, embeddings
├── week2_agent.py          Week 2 — ReAct agent, tools, alerts, logging
├── api.py                  Week 3 — FastAPI endpoints, feedback storage
├── knowledge_updater.py    Ongoing — weekly Semantic Scholar updates
├── static/
│   └── index.html          Farmer UI with GPS weather
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
Tested 6 sizes from 100 to 1200 words. 400 words balanced retrieval precision with context richness.

**The boundary problem is real and measurable.**
Splitting one key sentence across a chunk boundary reduced retrieval similarity by 25% — from 0.847 to 0.631. Overlap is not optional.

**Silent failures are the most dangerous bugs.**
Wrong embedder → near-zero scores → wrong answers returned confidently with no error. Harder to find than a crash.

**The system prompt determines grounding more than anything else.**
"Answer using the retrieved passages" reduced hallucination more than tripling the number of retrieved chunks.

**Brute force search does not scale.**
4.47 hours per query at 7M chunks. ChromaDB HNSW: 4 seconds. This is why vector databases exist.

**Agent loops need limits.**
Without a 10-step maximum, ambiguous questions loop indefinitely consuming tokens and money.

**JavaScript is case-sensitive and quote-type-sensitive.**
`userLat` and `userlat` are different variables. Single quotes `'${lat}'` send literal text — backticks `` `${lat}` `` evaluate it. Both bugs cause silent failures.

**Location matters for agriculture.**
Hardcoded coordinates give wrong weather for farmers in different countries. GPS coordinates silently provide real local forecasts with no input from the farmer.

---

## Next steps

- Sentinel Hub API for real NDVI from Sentinel-2 satellites
- Ollama + Phi-3 for fully offline operation
- Multilingual support — language selector in UI
- Outcome tracking — follow up a week after recommendations
- Crop-specific guides (wheat, potato, millet, sorghum)
- Voice input for low-literacy farmers using Web Speech API
- sending alerts to an email or phone.

---

## References

Chandra, R., Swaminathan, M., Chakraborty, T., Ding, J., Kapetanovic, Z., Kumar, P., & Vasisht, D. (2022). Democratizing Data-Driven Agriculture Using Affordable Hardware. *IEEE Micro*, 42(1), 69–77.

---

*Built as a learning project to understand agentic AI systems end to end — from PDF ingestion and vector embeddings through ReAct reasoning loops to a deployed farmer interface with real GPS-based weather and a weekly knowledge update pipeline.*
