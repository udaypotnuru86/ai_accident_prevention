# SurakshaAI India — AI Accident Prediction & Prevention System

Comprehensive full-stack road safety intelligence platform: **FastAPI backend + scikit-learn Random Forest model + Leaflet Pan-India Hotspot Map + Multi-Dashboard Corridor Analytics + Actionable Prevention Protocols**.

## Features & Dashboards
1. **AI Risk Predictor & Prevention Hub**:
   - Live severity classification (Severity 1 to 4) using machine learning.
   - Dynamic 0–100% Composite Hazard Risk Gauge with real-time hazard factor tags.
   - Tailored preventive guidelines for Indian driving conditions (Monsoon aquaplaning, winter dense fog, 2-wheeler defensive riding).
   - Pan-India emergency assistance helplines (112, 1033 NHAI Highway SOS, 108, 1095).
2. **Pan-India Hotspot Map Explorer**:
   - Interactive Leaflet map centered on India (`21.5°N, 78.9°E`) with dark matter tiles.
   - Color-coded severity markers (Severity 1: Green, 2: Amber, 3: Orange, 4: Red).
   - Region/City quick filters (Delhi NCR, Mumbai MMR, Bengaluru, Chennai, Hyderabad, Pune, Kolkata, Ahmedabad, Jaipur, Lucknow).
   - Click-anywhere map picker to simulate predictions for any custom Indian coordinate.
3. **Corridor Analytics & Trends Dashboard**:
   - Hourly crash distribution (IST) with morning fatigue and evening rush hour peaks.
   - Severity breakdown donut chart (Levels 1 to 4).
   - Seasonal trend chart highlighting monsoon spikes (Jul-Aug) and winter radiation fog (Dec-Jan).
   - High-risk blackspot corridors table (Yamuna Expressway, Mumbai-Pune Expressway, Silk Board, etc.).
4. **Safety Guidelines & Protocols**:
   - Specialized safety guides for heavy monsoon rain, North Indian winter fog, two-wheeler defensive riding, and expressways.

## 1. Install
Install Python 3.10+ and VS Code. Open this folder in VS Code, then in its terminal:

```bash
python -m venv .venv
```
Windows:
```bash
.venv\Scripts\activate
```
macOS/Linux:
```bash
source .venv/bin/activate
```
Then:
```bash
pip install -r requirements.txt
```

## 2. Train model
```bash
python train_model.py
```
This trains a baseline Random Forest severity classifier and saves `models/severity_model.joblib`.

## 3. Run website
```bash
uvicorn app:app --reload
```
Open http://127.0.0.1:8000

The map uses Leaflet/OpenStreetMap tiles and therefore needs an internet connection.

## 4. Use a real dataset
Download the US Accidents 2016–2023 dataset from Kaggle:
https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
It is large (the full CSV is about 3 GB), so begin with the provided sampled version if available, or create a manageable sample. Put the CSV at `data/US_Accidents_March23.csv`, then run:
```bash
python train_model.py --csv data/US_Accidents_March23.csv
```
The training script reads only the columns it needs and samples up to 150,000 rows for a beginner-friendly first run. Ensure your file has the expected columns:
`Severity, Start_Time, Start_Lat, Start_Lng, Temperature(F), Humidity(%), Visibility(mi), Weather_Condition`.

## Model notes
- Target: recorded severity (1–4), not crash occurrence.
- Input features: hour, month, latitude, longitude, temperature, humidity, visibility, weather condition.
- Metrics: accuracy and weighted F1; these are baseline metrics and must be interpreted alongside class balance and per-class performance.
- Do not use predictions for emergency response or official traffic control without rigorous validation.
