from http.client import HTTPException

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
import sys
import psutil
import time
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
    check_field_alerts, run_agent
)
from auth import (
    UserRegister, UserLogin, TokenResponse,
    hash_password, verify_password, create_token,
    get_current_user, USERS, oauth2_scheme
)

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

FEEDBACK_FILE = Path("data/feedback.json")
ESTIMATED_WATTS_PER_CPU_AT_FULL_LOAD=15.0
ESTIMATED_IDLE_WATTS_PER_CPU_CORE=3.0
_process_start_time  = time.time()

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
limiter= Limiter(key_func=get_remote_address)
app.state.limiter=limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
def ui():
    return FileResponse("static/index.html")

class Question(BaseModel):
    question: str
    field_id: str = "field_a"
    lat:      float = 37.7749    # default San Francisco
    lon:      float = -122.4194
    language: str = "English"

@app.get("/")
def root():
    return {"message": "FarmBeats AI Advisor is running"}

@app.post("/ask")
@limiter.limit("10/minute")
def ask(request: Request, body: Question, current_user=Depends(get_current_user)):
    if body.field_id not in current_user["fields"]:
        raise HTTPException(
            status_code=403,
            detail=f"You don't have access to {body.field_id}"
        )

    recommendation = run_agent(
        question=body.question,
        field_id=body.field_id,
        lat=body.lat,
        lon=body.lon,
        language=body.language
    )

    return {
        "question":       body.question,
        "field_id":       body.field_id,
        "language":       body.language,
        "recommendation": recommendation,
        "status":         "success"
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

@app.post("/register", response_model=TokenResponse)
def register(body: UserRegister):
    if body.email in USERS:
        raise HTTPException(status_code=400, detail="Email already registered")

    USERS[body.email]={
        "email": body.email,
        "name": body.name,
        "hashed_password": hash_password(body.password),
        "fields": body.fields
    }

    token=create_token(body.email)
    return TokenResponse(access_token=token,
                         token_type="bearer",
                         name=body.name,
                         fields=body.fields)

# @app.post("/login", response_model=TokenResponse)
# def login(form_data: OAuth2PasswordRequestForm=Depends()):
#     user=USERS.get(form_data.username)
#
#     if not user or not verify_password(form_data.password, user["hashed_password"]):
#         raise HTTPException(status_code=401, detail="Incorrect email or password")
#
#     token=create_token(form_data.username)
#     return TokenResponse(
#         access_token=token,
#         token_type="bearer",
#         name=user["name"],
#         fields=user["fields"]
#     )

@app.get("/me")
def me(current_user=Depends(get_current_user)):
    return {
        "email": current_user["email"],
        "name": current_user["name"],
        "fields": current_user["fields"]
    }

@app.post("/login", response_model=TokenResponse)
def login(body: UserLogin):
    user = USERS.get(body.email)
    if not user or not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password"
        )
    token = create_token(body.email)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        name=user["name"],
        fields=user["fields"]
    )

@app.get("/sustainability/energy")
def get_estimated_energy():
    cpu_utilization=psutil.cpu_percent(interval=0.5) / 100.0
    available_cores=psutil.cpu_count()
    watts_per_core = ESTIMATED_IDLE_WATTS_PER_CPU_CORE + (
        cpu_utilization * (ESTIMATED_WATTS_PER_CPU_AT_FULL_LOAD - ESTIMATED_IDLE_WATTS_PER_CPU_CORE))
    total_watts = watts_per_core * available_cores

    uptime_hours = (time.time() - _process_start_time) / 3600.0
    estimated_energy_kwh = (total_watts * uptime_hours) / 1000.0

    return {"estimatedEnergyKwh": round(estimated_energy_kwh, 4)}


