# Week 3 — FastAPI + Farmer UI

## What FastAPI does

Turns Python functions into REST API endpoints with minimal code:

```python
@app.post("/ask")
def ask(body: Question):
    # your code here
    return {"recommendation": "..."}
```

The `@app.post("/ask")` decorator tells FastAPI: when someone sends a POST request to /ask, run this function and return its result as JSON.

---

## Endpoints

```
POST /ask
  Input:  question, field_id, lat, lon
  Output: recommendation, sensor_data, weather_summary, research_passages

GET /alerts
  Input:  none
  Output: all_clear, alerts list, fields_checked

GET /sensor/{field_id}
  Input:  field_id in URL path
  Output: soil_moisture, temperature, humidity

GET /weather?lat=X&lon=Y
  Input:  lat, lon as query parameters
  Output: 3-day forecast

POST /feedback
  Input:  question, field_id, recommendation, rating
  Output: status confirmation

GET /ui
  Input:  none
  Output: serves static/index.html
```

---

## Pydantic BaseModel

Validates incoming request data automatically:

```python
class Question(BaseModel):
    question: str
    field_id: str = "field_a"
    lat:      float = 37.7749
    lon:      float = -122.4194
```

If a farmer sends a request without a question field, FastAPI returns a 422 error automatically — no manual validation needed.

GET vs POST:
- GET — retrieving data that exists (alerts, sensor readings, weather)
- POST — sending data to be processed (question, feedback)

---

## GPS-based weather

Browser gets real coordinates silently:

```javascript
navigator.geolocation.getCurrentPosition(function(position) {
    userLat = position.coords.latitude;
    userLon = position.coords.longitude;
    loadWeather(userLat, userLon);
});
```

Coordinates passed to FastAPI → passed to Open-Meteo → real local forecast.

`"timezone": "auto"` in Open-Meteo call detects correct timezone from coordinates automatically. Farmer in India gets IST, farmer in Kenya gets EAT.

---

## Field-filtered alerts

Alert filtering happens in JavaScript, not the API:

```javascript
const fieldAlerts = data.alerts.filter(alert =>
    alert.field === selectedField
);
```

Field B (58% moisture) shows "All clear." Field A and C show irrigation warnings. Switching fields updates alerts immediately without a page reload.

---

## Three bugs fixed in the UI

1. `lon=userLon` → should be `lon: userLon` (= is assignment, : is object key-value)
2. `userlat` → should be `userLat` (JavaScript is case-sensitive)
3. `'http://...?lat=${lat}'` → should use backticks `` `http://...?lat=${lat}` `` (single quotes don't evaluate ${} expressions)

All three caused silent failures — no crash, just wrong behaviour.

---

## Feedback system

```
Farmer clicks 👍 or 👎
      ↓
POST /feedback called with question + recommendation + rating
      ↓
Saved to data/feedback.json
      ↓
"Thanks for your feedback ✓" shown
```

Future: feed positive ratings back into ChromaDB as farmer-contributed knowledge.

---

## Key decisions

**Why Option B for alerts (filter by field)?**
A farmer asking about Field B shouldn't be distracted by Field A and C problems. Field-specific view matches how a farmer thinks — they're standing in one field, asking about that field.

**Why uvicorn separately from FastAPI?**
FastAPI is the framework that defines your endpoints. Uvicorn is the ASGI server that actually runs them and handles HTTP connections. They're separate tools with separate jobs — like a recipe vs a kitchen.

**Why `--reload` flag?**
Uvicorn watches your files for changes and restarts automatically. Saves time during development — no need to stop and restart the server manually after every code change.
