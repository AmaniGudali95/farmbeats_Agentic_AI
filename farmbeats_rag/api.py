from fastapi import FastAPI
from pydantic import BaseModel
import sys
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json
from datetime import datetime
from pathlib import Path
sys.path.insert(0, ".")
from week2_agent import (
    run_tool,
    daily_alert_check,
    simulate_agent_reasoning,
    check_field_alerts
)

FEEDBACK_FILE = Path("data/feedback.json")

def save_feedback(question, field_id, recommendation, rating):
    FEEDBACK_FILE.parent.mkdir(exist_ok=True)

    # Load existing feedback
    if FEEDBACK_FILE.exists():
        feedback = json.loads(FEEDBACK_FILE.read_text())
    else:
        feedback = []

    # Add new entry
    feedback.append({
        "timestamp":      datetime.now().isoformat(),
        "question":       question,
        "field_id":       field_id,
        "recommendation": recommendation,
        "rating":         rating  # "helpful" or "not_helpful"
    })

    FEEDBACK_FILE.write_text(json.dumps(feedback, indent=2))


app = FastAPI(title="FarmBeats AI Advisor")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
def ui():
    return FileResponse("static/index.html")

class Question(BaseModel):
    question: str
    field_id: str = "field_a"
    lat:      float = 37.7749    # default San Francisco
    lon:      float = -122.4194

@app.get("/")
def root():
    return {"message": "FarmBeats AI Advisor is running"}

@app.post("/ask")
def ask(body: Question):
    # Get all the data the agent would use
    sensor = run_tool("get_sensor_data", {"field_id": body.field_id})
    weather = run_tool("get_weather_forecast", {"days": 3, "lat": body.lat, "lon": body.lon})
    knowledge = run_tool("search_farm_knowledge", {"query": body.question})

    # Build a recommendation
    moisture = sensor.get("soil_moisture", 0)
    rain_coming = any(
        d["rain_mm"] > 5 for d in weather.get("forecast", [])
    )

    if moisture < 50 and not rain_coming:
        recommendation = (
            f"Irrigate today. Soil moisture is at {moisture}% "
            f"(below the 50% threshold) and no rain is forecast "
            f"for the next 3 days."
        )
    elif moisture < 50 and rain_coming:
        recommendation = (
            f"Hold off — rain is coming. Soil is at {moisture}% "
            f"but rain is forecast. Check again after rainfall."
        )
    else:
        recommendation = (
            f"No irrigation needed. Soil moisture is at {moisture}% "
            f"which is above the 50% threshold."
        )

    return {
        "question": body.question,
        "field_id": body.field_id,
        "recommendation": recommendation,
        "sensor_data": sensor,
        "weather_summary": {
            "days_checked": len(weather.get("forecast", [])),
            "rain_coming": rain_coming
        },
        "research_passages": len(knowledge.get("passages", [])),
        "status": "success"
    }

@app.get("/alerts")
def alerts():
    all_alerts = []
    fields = ["field_a", "field_b", "field_c"]

    for field in fields:
        field_alerts = check_field_alerts(field)
        all_alerts.extend(field_alerts)

    return {
        "all_clear": len(all_alerts) == 0,
        "alerts": all_alerts,
        "fields_checked": fields
    }

@app.get("/sensor/{field_id}")
def sensor(field_id: str):
    data = run_tool("get_sensor_data", {"field_id": field_id})
    return data

@app.get("/weather")
def weather(lat: float = 37.7749, lon: float = -122.4194):
    data = run_tool("get_weather_forecast", {"days": 3, "lat": lat, "lon": lon})
    return data


class Feedback(BaseModel):
    question:       str
    field_id:       str
    recommendation: str
    rating:         str  # "helpful" or "not_helpful"

@app.post("/feedback")
def feedback(body: Feedback):
    save_feedback(
        body.question,
        body.field_id,
        body.recommendation,
        body.rating
    )
    return {"status": "feedback saved"}