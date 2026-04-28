import anthropic
import json
import os
import sys
import requests
import time
from datetime import datetime
import random
sys.path.insert(0, ".")

FARM_TOOLS = [
    {
        "name": "get_sensor_data",
        "description": "Get current soil moisture and temperature readings from IoT sensors on the farm. Use this when you need to know current field conditions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "field_id": {
                    "type": "string",
                    "description": "The field identifier, e.g. 'field_a'"
                }
            },
            "required": ["field_id"]
        }
    },
    {
        "name": "get_weather_forecast",
        "description": "Get 7-day weather forecast for the farm. Use this when you need to plan field activities or check rain and frost risk.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of forecast days, 1-7"
                }
            },
            "required": ["days"]
        }
    },
    {
        "name": "get_ndvi_index",
        "description": "Get the latest NDVI crop health index from satellite imagery. Values above 0.5 indicate healthy crops. Use when farmer asks about crop health.",
        "input_schema": {
            "type": "object",
            "properties": {
                "field_id": {
                    "type": "string",
                    "description": "The field identifier"
                }
            },
            "required": ["field_id"]
        }
    },
    {
        "name": "search_farm_knowledge",
        "description": "Search the FarmBeats research paper for agronomic advice, thresholds, and best practices. Use this to ground answers in research.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language question about farming practices"
                }
            },
            "required": ["query"]
        }
    }
]

def run_tool(name, inputs):
    if name == "get_sensor_data":
        try:
            base_moisture = {
                "field_a": 42,
                "field_b": 58,
                "field_c": 31
            }
            moisture = base_moisture.get(
                inputs["field_id"],
                random.randint(30, 70)
            )
        # Add small random variation to simulate sensor noise
            moisture = moisture + random.randint(-3, 3)

            return {
                "field_id": inputs["field_id"],
                "soil_moisture": moisture,
                "temperature": random.randint(22, 35),
                "humidity": random.randint(40, 80),
                "unit": "percent field capacity"
            }
        except Exception as e:
            return {"error": f"Sensor data failed:{str(e)}"}

    elif name == "get_weather_forecast":
        try:
            days = inputs.get("days", 3)
            lat=inputs.get("lat", 37.7749)
            lon=inputs.get("lon", -122.4194)
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": [
                    "precipitation_sum",
                    "temperature_2m_min",
                    "temperature_2m_max",
                    "weathercode"
                ],
                "forecast_days": days,
                "timezone": "auto"
        }
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
    # Format into clean list Claude can reason about
            daily = data["daily"]
            forecast = []
            for i in range(len(daily["time"])):
                forecast.append({
                "date":       daily["time"][i],
                "rain_mm":    daily["precipitation_sum"][i],
                "min_temp_c": daily["temperature_2m_min"][i],
                "max_temp_c": daily["temperature_2m_max"][i],
            })

            return {"forecast": forecast, "location": f"{lat:.2f}, {lon:.2f}"}
        except Exception as e:
            return {"error": f"Weather forecast failed:{str(e)}"}

    elif name == "get_ndvi_index":
        try:
            return {
            "field_id": inputs["field_id"],
            "ndvi": 0.67,
            "status": "healthy",
            "last_updated": "2026-04-17"
            }
        except Exception as e:
            return {"error": f"NDVI data failed:{str(e)}"}

    elif name == "search_farm_knowledge":
        try:
            from rag_cli import AnthropicEmbedder, VectorStore, DB_PATH
            embedder = AnthropicEmbedder()
            store = VectorStore(db_path=DB_PATH, embedder=embedder)
            passages = store.query(inputs["query"], top_k=3)
            return {"passages": [p["text"][:300] for p in passages]}
        except Exception as e:
            return {"error": f"Knowledge search failed:{str(e)}"}
    else:
        return {"error": f"Unknown tool: {name}"}


SYSTEM_PROMPT = """You are an AI farming advisor built on the FarmBeats 
research system. You help farmers make data-driven decisions.

Always:
- Check live sensor data before making irrigation recommendations
- Check weather forecast before advising on field activities
- Search the knowledge base for research to back your advice
- Give answers in plain language a farmer understands
- Cite your data sources in your answer
"""


def log(step, event, data=None):
    timestamp = datetime.now().strftime("%H:%M:%S")
    if data:
        print(f"[{timestamp}] Step {step} | {event} | {str(data)[:150]}")
    else:
        print(f"[{timestamp}] Step {step} | {event}")

def run_agent(question, field_id="field_a"):
    log(0, "Agent started", f"question: {question}")
    start_time = time.time()
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY", "")
    )

    messages = [{"role": "user", "content": question}]

    print(f"\nFarmer asks: {question}")
    print("─" * 60)

    for step in range(10):
        log(step + 1, "Calling Claude", f"messages in history: {len(messages)}")
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=FARM_TOOLS,
            messages=messages
        )
        log(step + 1, f"Claude responded", f"stop_reason: {response.stop_reason}")
        print(f"\nStep {step + 1} — stop_reason: {response.stop_reason}")

        if response.stop_reason == "end_turn":
            log(step + 1, "Final answer produced")
            answer = next(
                b.text for b in response.content if b.type == "text"
            )
            print(f"\nFinal answer:\n{answer}")
            return answer

        if response.stop_reason == "tool_use":
            messages.append({
                "role": "assistant",
                "content": response.content
            })

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    log(step + 1, f"Tool called: {block.name}", block.input)
                    print(f"  → calling {block.name}({block.input})")
                    result = run_tool(block.name, block.input)
                    log(step + 1, f"Tool result: {block.name}", result)
                    print(f"  ← result: {json.dumps(result)[:100]}...")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })

            messages.append({
                "role": "user",
                "content": tool_results
            })
    elapsed = time.time() - start_time
    log(0, f"Agent finished in {elapsed:.1f}s")
    print("WARNING: Agent reached step limit after 10 steps")
    return (
        "I was unable to complete my analysis within the allowed steps. "
        "Based on what I gathered: check your soil moisture levels and "
        "consult the weather forecast before making irrigation decisions. "
        "Please try asking a more specific question."
    )
def check_field_alerts(field_id="field_a"):
    alerts = []

    # Check 1 — soil moisture
    sensor = run_tool("get_sensor_data", {"field_id": field_id})
    if "error" not in sensor:
        moisture = sensor["soil_moisture"]
        if moisture < 50:
            alerts.append({
                "type": "irrigation",
                "severity": "high" if moisture < 35 else "medium",
                "message": f"Soil moisture at {moisture}% — below 50% threshold. Irrigation needed.",
                "field": field_id
            })

    # Check 2 — frost risk
    weather = run_tool("get_weather_forecast", {"days": 3})
    if "error" not in weather:
        for day in weather["forecast"]:
            if day["min_temp_c"] < 2:
                alerts.append({
                    "type": "frost",
                    "severity": "high",
                    "message": f"Frost risk on {day['date']} — minimum {day['min_temp_c']}°C. Protect crops.",
                    "field": field_id
                })

    # Check 3 — crop health
    ndvi = run_tool("get_ndvi_index", {"field_id": field_id})
    if "error" not in ndvi:
        if ndvi["ndvi"] < 0.5:
            alerts.append({
                "type": "crop_stress",
                "severity": "high",
                "message": f"NDVI at {ndvi['ndvi']} — below 0.5 threshold. Crop stress detected.",
                "field": field_id
            })

    return alerts


def daily_alert_check(fields=["field_a", "field_b", "field_c"]):
    print(f"\n{'='*60}")
    print(f"Daily Alert Check — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    all_clear = True

    for field in fields:
        alerts = check_field_alerts(field)

        if alerts:
            all_clear = False
            print(f"\n⚠️  ALERTS for {field}:")
            for alert in alerts:
                print(f"   [{alert['severity'].upper()}] {alert['message']}")
        else:
            print(f"\n✓  {field} — all clear")

    if all_clear:
        print("\nAll fields clear. No action needed today.")

    return all_clear

def simulate_agent_reasoning(question, field_id="field_a"):
    """
    Simulates what the agent would do without calling Claude.
    Shows the full reasoning trace so you understand the flow.
    """
    print(f"\n{'='*60}")
    print(f"SIMULATED AGENT TRACE")
    print(f"Question: {question}")
    print(f"{'='*60}")

    # Step 1 — get sensor data
    print("\nStep 1 — Claude decides: I need current field conditions")
    sensor = run_tool("get_sensor_data", {"field_id": field_id})
    print(f"  → called get_sensor_data({field_id})")
    print(f"  ← soil_moisture: {sensor['soil_moisture']}%")

    # Step 2 — get weather
    print("\nStep 2 — Claude decides: I need the weather forecast")
    weather = run_tool("get_weather_forecast", {"days": 3})
    print(f"  → called get_weather_forecast(days=3)")
    for day in weather["forecast"]:
        print(f"  ← {day['date']}: rain={day['rain_mm']}mm  "
              f"min={day['min_temp_c']}°C")

    # Step 3 — search knowledge base
    print("\nStep 3 — Claude decides: I need research on irrigation thresholds")
    knowledge = run_tool("search_farm_knowledge",
                         {"query": "corn irrigation soil moisture threshold"})
    print(f"  → called search_farm_knowledge")
    print(f"  ← retrieved {len(knowledge['passages'])} passages")
    print(f"  ← passage 1: {knowledge['passages'][0][:120]}...")

    # Step 4 — synthesise
    print("\nStep 4 — Claude synthesises all data:")
    moisture = sensor["soil_moisture"]
    rain_coming = any(d["rain_mm"] > 5 for d in weather["forecast"])
    frost_risk = any(d["min_temp_c"] < 2 for d in weather["forecast"])

    print(f"  soil moisture: {moisture}% "
          f"({'below' if moisture < 50 else 'above'} 50% threshold)")
    print(f"  rain coming:   {'yes' if rain_coming else 'no'}")
    print(f"  frost risk:    {'yes' if frost_risk else 'no'}")

    # Step 5 — recommendation
    print("\nStep 5 — Claude produces final recommendation:")
    if moisture < 50 and not rain_coming:
        print(f"  RECOMMENDATION: Irrigate today.")
        print(f"  REASON: Soil at {moisture}% (below 50% threshold) "
              f"and no rain forecast for 3 days.")
    elif moisture < 50 and rain_coming:
        print(f"  RECOMMENDATION: Hold off — rain is coming.")
        print(f"  REASON: Soil at {moisture}% but rain forecast. "
              f"Check moisture again after rain.")
    else:
        print(f"  RECOMMENDATION: No irrigation needed.")
        print(f"  REASON: Soil at {moisture}% — above 50% threshold.")

    if frost_risk:
        print(f"  ⚠️  FROST WARNING: Protect crops — sub-2°C temps forecast.")

#run_agent("My corn is in week 6 of growth. Should I irrigate today?")
# Test tools directly — no API key needed
print("Testing tools directly:\n")

print("Sensor data:")
print(run_tool("get_sensor_data", {"field_id": "field_a"}))

print("\nWeather forecast:")
print(run_tool("get_weather_forecast", {"days": 3}))

print("\nNDVI index:")
print(run_tool("get_ndvi_index", {"field_id": "field_a"}))

print("\nKnowledge search:")
print(run_tool("search_farm_knowledge",
               {"query": "soil moisture threshold corn irrigation"}))

daily_alert_check()

simulate_agent_reasoning(
    "My corn is in week 6. It hasn't rained in 10 days. Should I irrigate?",
    field_id="field_a"
)

simulate_agent_reasoning(
    "Is my crop healthy? Should I be worried?",
    field_id="field_c"
)

