# Week 2 — ReAct Agent Progress

## What the ReAct agent does

Instead of one RAG call, the agent reasons across multiple steps:

```
Round 1: Claude reasons → calls get_sensor_data(field_a)
         Returns: soil_moisture=43%, temperature=22°C

Round 2: Claude reasons → calls get_weather_forecast(days=7)
         Returns: no rain for 7 days, temperatures rising

Round 3: Claude reasons → calls search_farm_knowledge("corn irrigation threshold")
         BGE retrieves: "irrigate below 50% field capacity..."

Round 4: Claude synthesises → final cited recommendation
```

---

## Four tools

```
get_sensor_data(field_id)
  Returns: soil_moisture, temperature, humidity
  Currently: simulated with realistic random variation per field
  Production: IoT sensor API or Raspberry Pi

get_weather_forecast(days, lat, lon)
  Returns: date, rain_mm, min_temp_c, max_temp_c per day
  Currently: real Open-Meteo API — free, no key, GPS-based
  Production: same API, coordinates from farmer's phone

get_ndvi_index(field_id)
  Returns: ndvi value (0-1), status, last_updated
  Currently: hardcoded 0.67
  Production: Sentinel Hub free satellite API

search_farm_knowledge(query)
  Returns: top 3 relevant passages from ChromaDB
  Currently: BGE embeddings, 9 chunks
  Production: grows weekly via knowledge_updater.py
```

---

## Field baselines (simulated sensors)

```
field_a: base moisture ~42% → MEDIUM alert (below 50%)
field_b: base moisture ~58% → all clear (above 50%)
field_c: base moisture ~31% → HIGH alert (below 35%)
```

Random variation of ±3% applied each call to simulate sensor noise.

---

## Alert thresholds

```
Irrigation:
  moisture < 50%  → MEDIUM alert
  moisture < 35%  → HIGH alert

Frost:
  min_temp_c < 2°C in next 3 days → HIGH alert

Crop stress:
  ndvi < 0.5 → HIGH alert
```

---

## Error handling

Every tool wrapped in try/except:
```python
try:
    # tool code
except Exception as e:
    return {"error": f"Tool name failed: {str(e)}"}
```

Claude reads the error dictionary and adjusts its reasoning — "I couldn't get sensor data, here's what I can tell you from research alone."

---

## Logging

Every tool call timestamped:
```
[10:23:03] Step 0 | Agent started | question: My corn is in week 6...
[10:23:08] Step 1 | Tool called: get_sensor_data | {'field_id': 'field_a'}
[10:23:08] Step 1 | Tool result: get_sensor_data | {'soil_moisture': 43...}
[10:23:31] Step 2 | Final answer produced
```

Three most important things to log for debugging:
1. What question triggered the agent
2. What each tool returned
3. Which step produced the final answer

---

## Real Claude agent output (first run with API key)

Question: "My corn is in week 6 of growth. Should I irrigate today?"

Claude called three tools in Step 1:
- get_sensor_data → soil_moisture: 43%
- get_weather_forecast → no rain for 7 days, temps rising to 29.9°C
- search_farm_knowledge → FarmBeats irrigation passages

Claude's answer:
> "Yes — irrigate today. Soil moisture at 43% is below the 50-70% optimal range for week 6 corn. No rain for 7 days. Temperatures rising to 29.9°C will increase water demand. Irrigate to 60-65% field capacity today. Monitor again around May 10-11 — you may need a second irrigation."

Compare to simulated reasoning:
> "Irrigate today. Soil at 41%, below 50% threshold, no rain for 3 days."

Real Claude: specific target range, growth stage context, temperature trend reasoning, predicted second irrigation date.
Simulated: correct but generic.

---

## Key concepts

**Why messages grow each loop iteration:**
Claude has no memory between API calls. Every call sends complete conversation history so Claude knows what tools it called and what results came back.

**Why tool_use_id matters:**
Each tool call has a unique ID. Results must reference this ID so Claude knows which result belongs to which tool call. Multiple tools can be called in one step.

**Why 10-step limit:**
Without a limit, a poorly worded question could cause Claude to loop indefinitely. 10 steps is generous — most questions resolve in 2-3 steps.

**Why json.dumps() for tool results:**
The Anthropic API requires tool results as strings, not Python dictionaries. json.dumps() converts the dict to a JSON string the API accepts.

**Why single run_tool() dispatcher:**
The agent loop receives a tool name string from Claude. One dispatcher handles any name in one line: `result = run_tool(block.name, block.input)`. Adding a fifth tool requires only one elif — the agent loop never changes.
