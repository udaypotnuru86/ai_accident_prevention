from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
INDIA_DATA_PATH = BASE / "data" / "india_accidents.csv"
US_DATA_PATH = BASE / "data" / "US_Accidents_March23.csv"
DEMO_DATA_PATH = BASE / "data" / "demo_accidents.csv"
DATA_PATH = INDIA_DATA_PATH if INDIA_DATA_PATH.exists() else (US_DATA_PATH if US_DATA_PATH.exists() else DEMO_DATA_PATH)
MODEL_PATH = BASE / "models" / "severity_model.joblib"

app = FastAPI(
    title="India AI Accident Prediction & Prevention System",
    description="Intelligent road safety intelligence: severity classification, corridor analytics, and live risk mitigation."
)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

class PredictionInput(BaseModel):
    latitude: float = Field(28.6139, ge=-90.0, le=90.0)
    longitude: float = Field(77.2090, ge=-180.0, le=180.0)
    hour: int = Field(19, ge=0, le=23)
    month: int = Field(7, ge=1, le=12)
    temperature_f: float = 84.0
    humidity: float = Field(75.0, ge=0.0, le=100.0)
    visibility_mi: float = Field(4.0, ge=0.0)
    weather_condition: str = "Rain"
    road_type: str = "Expressway"
    vehicle_type: str = "Four-Wheeler / Car"

_cached_df = None

def load_data():
    global _cached_df
    if _cached_df is not None:
        return _cached_df
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    if "Start_Time" in df.columns:
        df["dt"] = pd.to_datetime(df["Start_Time"], errors="coerce")
        df["hour"] = df["dt"].dt.hour
        df["month"] = df["dt"].dt.month
    _cached_df = df
    return df

@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")

@app.get("/api/summary")
def summary():
    df = load_data()
    if df.empty:
        return {
            "total_records": 0,
            "mean_severity": 0,
            "high_severity_pct": 0,
            "peak_hour": 19,
            "hotspots": [],
            "cities": []
        }
    
    group_cols = ["City", "Start_Lat", "Start_Lng"]
    if "Corridor" in df.columns:
        group_cols.append("Corridor")
    if "Road_Type" in df.columns:
        group_cols.append("Road_Type")
    if "Primary_Cause" in df.columns:
        group_cols.append("Primary_Cause")

    hotspots = (df.groupby(group_cols, dropna=True)
                  .agg(count=("Severity", "size"), severity=("Severity", "mean"))
                  .reset_index()
                  .sort_values("count", ascending=False)
                  .head(150))

    total = len(df)
    high_sev = len(df[df["Severity"].isin([3, 4])])
    high_sev_pct = round((high_sev / total * 100), 1) if total > 0 else 0
    mean_sev = round(float(df["Severity"].mean()), 2)
    peak_hr = int(df["hour"].mode()[0]) if "hour" in df.columns and not df["hour"].dropna().empty else 19

    city_coords = (
        df.groupby("City")
          .agg(lat=("Start_Lat", "mean"), lng=("Start_Lng", "mean"), count=("Severity", "size"))
          .reset_index()
          .sort_values("count", ascending=False)
          .to_dict(orient="records")
    )

    hotspot_list = []
    for _, r in hotspots.iterrows():
        hotspot_list.append({
            "city": str(r.City),
            "corridor": str(getattr(r, "Corridor", r.City)),
            "road_type": str(getattr(r, "Road_Type", "Road")),
            "primary_cause": str(getattr(r, "Primary_Cause", "High Traffic Flow")),
            "lat": float(r.Start_Lat),
            "lng": float(r.Start_Lng),
            "count": int(r["count"]),
            "severity": round(float(r.severity), 2)
        })

    return {
        "total_records": total,
        "mean_severity": mean_sev,
        "high_severity_pct": high_sev_pct,
        "peak_hour": peak_hr,
        "hotspots": hotspot_list,
        "cities": city_coords
    }

@app.get("/api/analytics")
def analytics():
    df = load_data()
    if df.empty:
        return {}

    hourly_counts = [0] * 24
    if "hour" in df.columns:
        counts = df["hour"].value_counts().to_dict()
        for h in range(24):
            hourly_counts[h] = int(counts.get(h, 0))

    monthly_counts = [0] * 12
    if "month" in df.columns:
        counts = df["month"].value_counts().to_dict()
        for m in range(1, 13):
            monthly_counts[m - 1] = int(counts.get(m, 0))

    sev_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for k, v in df["Severity"].value_counts().to_dict().items():
        if int(k) in sev_counts:
            sev_counts[int(k)] = int(v)

    weather_counts = {}
    if "Weather_Condition" in df.columns:
        for k, v in df["Weather_Condition"].value_counts().head(6).to_dict().items():
            weather_counts[str(k)] = int(v)

    road_counts = {}
    if "Road_Type" in df.columns:
        for k, v in df["Road_Type"].value_counts().to_dict().items():
            road_counts[str(k)] = int(v)

    corridors = []
    if "Corridor" in df.columns:
        top_corr = (
            df.groupby("Corridor")
              .agg(
                  count=("Severity", "size"),
                  severity=("Severity", "mean"),
                  city=("City", "first"),
                  road_type=("Road_Type", "first"),
                  primary_cause=("Primary_Cause", "first")
              )
              .reset_index()
              .sort_values(["severity", "count"], ascending=False)
              .head(8)
        )
        for _, r in top_corr.iterrows():
            corridors.append({
                "name": str(r.Corridor),
                "city": str(r.city),
                "road_type": str(r.road_type),
                "count": int(r["count"]),
                "avg_severity": round(float(r.severity), 2),
                "primary_cause": str(r.primary_cause)
            })

    return {
        "hourly": hourly_counts,
        "monthly": monthly_counts,
        "severity": sev_counts,
        "weather": weather_counts,
        "road_type": road_counts,
        "top_corridors": corridors
    }

@app.post("/api/predict")
def predict(payload: PredictionInput):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model file not found. Please train model first.")
    
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([{
        "Start_Lat": payload.latitude,
        "Start_Lng": payload.longitude,
        "Temperature(F)": payload.temperature_f,
        "Humidity(%)": payload.humidity,
        "Visibility(mi)": payload.visibility_mi,
        "hour": payload.hour,
        "month": payload.month,
        "Weather_Condition": payload.weather_condition
    }])
    
    pred = int(model.predict(row)[0])

    base_score = {1: 22, 2: 45, 3: 72, 4: 90}.get(pred, 50)
    modifier = 0
    detected_factors = []

    if payload.road_type == "Expressway":
        modifier += 10
        detected_factors.append("High-speed expressway corridor with heightened kinetic impact probability")
    elif payload.road_type == "National Highway":
        modifier += 6
        detected_factors.append("National highway mixed-traffic corridor with heavy commercial transit")
    elif payload.road_type == "Flyover":
        modifier += 4
        detected_factors.append("Elevated flyover with elevated curve / merging vulnerability")

    weather_lower = payload.weather_condition.lower()
    if "fog" in weather_lower or payload.visibility_mi < 1.5:
        modifier += 15
        detected_factors.append("Severe winter fog / sub-mile visibility (critical multi-vehicle pileup hazard)")
    elif "rain" in weather_lower or "thunder" in weather_lower or payload.humidity > 85:
        modifier += 12
        detected_factors.append("Monsoon wet surface hydroplaning & braking distance elongation")
    elif "haze" in weather_lower:
        modifier += 5
        detected_factors.append("Reduced urban contrast and ambient particulate haze")

    if payload.hour in [0, 1, 2, 3, 4, 5]:
        modifier += 14
        detected_factors.append("Graveyard shift hours (00:00 - 05:00) with peak driver fatigue & heavy multi-axle freight")
    elif payload.hour in [18, 19, 20, 21, 22]:
        modifier += 8
        detected_factors.append("Peak evening rush hour congestion with aggressive overtaking")

    if payload.vehicle_type == "Two-Wheeler":
        modifier += 12
        detected_factors.append("Two-wheeler rider vulnerability: heightened physical impact exposure")
    elif payload.vehicle_type == "Heavy Commercial / Truck":
        modifier += 8
        detected_factors.append("Heavy commercial vehicle momentum and large blind spots")

    composite_risk = int(np.clip(base_score + modifier, 10, 99))

    if composite_risk < 35:
        risk_level = "Low Risk"
        risk_color = "#10b981"
        badge_bg = "rgba(16, 185, 129, 0.15)"
    elif composite_risk < 60:
        risk_level = "Moderate Risk"
        risk_color = "#f59e0b"
        badge_bg = "rgba(245, 158, 11, 0.15)"
    elif composite_risk < 80:
        risk_level = "High Hazard Risk"
        risk_color = "#f97316"
        badge_bg = "rgba(249, 115, 22, 0.15)"
    else:
        risk_level = "Critical Threat Zone"
        risk_color = "#ef4444"
        badge_bg = "rgba(239, 68, 68, 0.15)"

    severity_descriptions = {
        1: {
            "title": "Severity 1: Minor Collision / Superficial Damage",
            "desc": "Expected low-speed impact, minor vehicular scrape or fender-bender with negligible structural compromise and standard traffic flow resumption."
        },
        2: {
            "title": "Severity 2: Moderate Incident / Traffic Disruption",
            "desc": "Moderate collision causing single or double lane blockage, vehicle immobilisation, and moderate emergency assistance response required."
        },
        3: {
            "title": "Severity 3: Serious Hazard / Major Trauma Risk",
            "desc": "High kinetic collision with substantial cabin deformation, multi-vehicle entrapment, and mandatory paramedic deployment."
        },
        4: {
            "title": "Severity 4: Critical Emergency / Severe High-Speed Hazard",
            "desc": "Severe catastrophic impact, potential multi-vehicle expressway pileup, structural rollover, or full corridor closure."
        }
    }

    advice = []
    if "fog" in weather_lower or payload.visibility_mi < 2.0:
        advice.append("Turn ON low-beam headlights and dedicated yellow fog lights; strictly avoid high beams as they reflect off water droplets.")
        advice.append("Do NOT park on highway shoulders in fog. Follow white road margin edge markings and highway cat-eyes.")
    if "rain" in weather_lower or payload.humidity > 80:
        advice.append("Double your following distance to at least 4 vehicle lengths to counter sudden water hydroplaning on tarmac.")
        advice.append("Avoid standing water along median lanes; brake progressively to prevent tyre lockup on ungrooved pavement.")
    if payload.hour in [0, 1, 2, 3, 4, 5]:
        advice.append("Be alert for unlit stationary freight trucks, unbarricaded road diversions, and agricultural trailers.")
        advice.append("If feeling fatigued, pull into an authorized NHAI 'Wayside Amenity' or 24/7 fuel station immediately.")
    if payload.road_type == "Expressway":
        advice.append("Maintain strict lane discipline: leftmost lane for heavy freight, center for cruising, rightmost strictly for overtaking.")
        advice.append("Never reverse or execute unauthorized U-turns near expressway exits or toll plaza ramps.")
    if payload.vehicle_type == "Two-Wheeler":
        advice.append("Wear an ISI/ECE-certified full-face helmet with strapped chin buckle and high-reflectivity neon jacket.")
        advice.append("Stay out of large blind zones directly behind or beside multi-axle freight trucks.")
    if not advice:
        advice.append("Adhere strictly to designated speed limits and watch for merging traffic at unprotected intersections.")
        advice.append("Use turn indicators at least 50 meters prior to lane maneuvers; always buckle up seatbelts front and rear.")

    helplines = [
        {"name": "National Emergency Service (Pan-India)", "number": "112", "icon": "shield"},
        {"name": "NHAI Highway Emergency Helpline", "number": "1033", "icon": "truck"},
        {"name": "Emergency Medical / Ambulance", "number": "108", "icon": "heart"},
        {"name": "Traffic Police Control Room", "number": "1095 / 103", "icon": "phone"}
    ]

    return {
        "predicted_severity": pred,
        "severity_info": severity_descriptions.get(pred, severity_descriptions[2]),
        "risk_score": composite_risk,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "badge_bg": badge_bg,
        "detected_factors": detected_factors if detected_factors else ["Standard road and environmental operating envelope"],
        "prevention_tips": advice,
        "helplines": helplines
    }
