import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# Specific Indian accident hubs with realistic corridors and coordinates
HUBS = [
    {
        "city": "New Delhi / NCR",
        "state": "Delhi / Haryana / UP",
        "corridors": [
            ("Delhi Outer Ring Road (Mukarba Chowk)", 28.7360, 77.1620, "Arterial Road"),
            ("NH-48 Delhi-Gurugram Expressway (Sirhaul Border)", 28.4980, 77.0890, "Expressway"),
            ("Yamuna Expressway (Greater Noida - Mathura)", 28.3240, 77.5310, "Expressway"),
            ("DND Flyway & Ring Road Junction", 28.5780, 77.2640, "Flyover"),
            ("Eastern Peripheral Expressway (KMP Interchange)", 28.4120, 76.9230, "Expressway"),
            ("NH-9 Ghazipur Border Corridor", 28.6250, 77.3320, "National Highway")
        ]
    },
    {
        "city": "Mumbai MMR",
        "state": "Maharashtra",
        "corridors": [
            ("Western Express Highway (Goregaon-Kandivali)", 19.1650, 72.8540, "Arterial Road"),
            ("Eastern Express Highway (Ghatkopar Stretch)", 19.0880, 72.9150, "Arterial Road"),
            ("Mumbai-Pune Expressway (Bhor Ghat Section)", 18.7720, 73.3420, "Expressway"),
            ("Sion-Panvel Expressway (Vashi Creek Bridge)", 19.0660, 72.9980, "Expressway"),
            ("Ghodbunder Road (Thane Bypass)", 19.2680, 72.9460, "State Highway")
        ]
    },
    {
        "city": "Bengaluru",
        "state": "Karnataka",
        "corridors": [
            ("Outer Ring Road (Silk Board Junction)", 12.9175, 77.6235, "Arterial Road"),
            ("Electronic City Elevated Expressway", 12.8450, 77.6620, "Expressway"),
            ("Outer Ring Road (Marathahalli - Bellandur)", 12.9380, 77.6960, "Arterial Road"),
            ("Hebbal Flyover & Airport Expressway (NH-44)", 13.0350, 77.5970, "Flyover"),
            ("NICE Road Toll Junction (Mysuru Road)", 12.8850, 77.4920, "Expressway"),
            ("Hosur Road (Bommasandra Industrial Zone)", 12.8120, 77.6980, "National Highway")
        ]
    },
    {
        "city": "Chennai",
        "state": "Tamil Nadu",
        "corridors": [
            ("Grand Southern Trunk (GST) Road (Tambaram)", 12.9240, 80.1280, "National Highway"),
            ("Old Mahabalipuram Road (OMR IT Corridor)", 12.8950, 80.2290, "Arterial Road"),
            ("Chennai Bypass Road (Porur - Maduravoyal)", 13.0420, 80.1530, "Expressway"),
            ("Inner Ring Road (Kathipara Junction)", 13.0070, 80.2050, "Flyover"),
            ("East Coast Road (ECR Sholinganallur junction)", 12.8710, 80.2450, "State Highway")
        ]
    },
    {
        "city": "Hyderabad",
        "state": "Telangana",
        "corridors": [
            ("Nehru Outer Ring Road (Gachibowli Exit)", 17.4350, 78.3440, "Expressway"),
            ("PVNR Elevated Expressway (Mehdipatnam)", 17.3780, 78.4310, "Expressway"),
            ("NH-65 Miyapur - Patancheru Corridor", 17.5120, 78.3050, "National Highway"),
            ("LB Nagar Ring Road Junction", 17.3510, 78.5520, "Arterial Road"),
            ("ORR Shamshabad Airport Stretch", 17.2410, 78.4310, "Expressway")
        ]
    },
    {
        "city": "Pune",
        "state": "Maharashtra",
        "corridors": [
            ("Katraj-Dehu Road Bypass (Wakad - Hinjewadi)", 18.5980, 73.7620, "National Highway"),
            ("Pune-Solapur Highway (Hadapsar Stretch)", 18.4980, 73.9420, "National Highway"),
            ("Pune-Nagar Road (Wagholi)", 18.5790, 73.9780, "State Highway"),
            ("Chandani Chowk Junction (Bavdhan)", 18.5080, 73.7820, "Arterial Road")
        ]
    },
    {
        "city": "Kolkata",
        "state": "West Bengal",
        "corridors": [
            ("Eastern Metropolitan (EM) Bypass (Ruby - Science City)", 22.5250, 88.3980, "Arterial Road"),
            ("Kona Expressway (Vidyasagar Setu Toll Plaza)", 22.5710, 88.3120, "Expressway"),
            ("Belghoria Expressway (NH-12 Junction)", 22.6540, 88.3880, "Expressway"),
            ("Maa Flyover (Park Circus - EM Bypass connector)", 22.5420, 88.3750, "Flyover")
        ]
    },
    {
        "city": "Ahmedabad",
        "state": "Gujarat",
        "corridors": [
            ("Sarkhej-Gandhinagar (SG) Highway (Iskcon Crossroad)", 23.0310, 72.5080, "Arterial Road"),
            ("Sardar Patel Ring Road (Sanand Crossroads)", 22.9980, 72.4650, "State Highway"),
            ("Ahmedabad-Vadodara Expressway (CTM Toll Plaza)", 22.9850, 72.6450, "Expressway")
        ]
    },
    {
        "city": "Jaipur",
        "state": "Rajasthan",
        "corridors": [
            ("NH-48 Jaipur-Delhi Highway (Amer bypass)", 27.0120, 75.8650, "National Highway"),
            ("Tonk Road & B2 Bypass Crossing", 26.8380, 75.7950, "Arterial Road"),
            ("Ajmer Road 200 Feet Bypass", 26.8790, 75.7250, "National Highway")
        ]
    },
    {
        "city": "Lucknow / Kanpur corridor",
        "state": "Uttar Pradesh",
        "corridors": [
            ("Agra-Lucknow Expressway (Sarosa Toll)", 26.8420, 80.7950, "Expressway"),
            ("Shaheed Path Outer Ring Bypass", 26.7820, 80.9520, "Arterial Road"),
            ("NH-27 Lucknow-Kanpur Highway (Unnao Stretch)", 26.5410, 80.4850, "National Highway")
        ]
    }
]

WEATHER_CONDITIONS = ["Clear", "Haze", "Rain", "Fog", "Thunderstorm", "Cloudy"]
CAUSES = [
    "Overspeeding", 
    "Low Visibility / Winter Fog", 
    "Monsoon Hydroplaning / Wet Road", 
    "Driver Fatigue / Late Night Driving", 
    "Wrong-Side Driving / Unauthorized U-Turn", 
    "Sudden Braking / Tailgating"
]
VEHICLE_TYPES = ["Two-Wheeler", "Four-Wheeler / Car", "Heavy Commercial / Truck", "Auto-Rickshaw", "Bus"]

def generate_records(num_records=4000):
    records = []
    base_date = datetime(2023, 1, 1)

    for i in range(num_records):
        hub = random.choice(HUBS)
        corridor, base_lat, base_lng, road_type = random.choice(hub["corridors"])

        # Add small spatial jitter (~100m to 1.5km)
        lat = base_lat + np.random.normal(0, 0.008)
        lng = base_lng + np.random.normal(0, 0.008)

        # Date and time distribution with realistic Indian traffic accident peaks:
        # Peak: Evening 18:00 - 22:00, Early Morning 02:00 - 06:00 (trucks/fatigue), Afternoon 14:00 - 16:00
        hour_weights = [
            0.04, 0.05, 0.05, 0.06, 0.05, 0.04, # 00 - 05 (fatigue/night trucks)
            0.03, 0.04, 0.05, 0.04, 0.03, 0.03, # 06 - 11 (morning rush)
            0.03, 0.03, 0.04, 0.04, 0.04, 0.05, # 12 - 17
            0.07, 0.08, 0.08, 0.07, 0.05, 0.04  # 18 - 23 (evening peak rush)
        ]
        hour_weights = np.array(hour_weights) / sum(hour_weights)
        hour = np.random.choice(range(24), p=hour_weights)
        
        # Random day in 2023
        day_offset = random.randint(0, 364)
        minute = random.randint(0, 59)
        accident_dt = base_date + timedelta(days=day_offset, hours=int(hour), minutes=minute)
        month = accident_dt.month

        # Weather correlation based on Indian seasonality
        # Monsoons: June - September (months 6, 7, 8, 9)
        # North Winter Fog: Dec - Jan (months 12, 1)
        # Summer heat: April - June (months 4, 5, 6)
        is_monsoon = month in [6, 7, 8, 9]
        is_winter = month in [12, 1]
        is_north = "Delhi" in hub["city"] or "UP" in hub["state"] or "Rajasthan" in hub["city"]

        if is_monsoon:
            weather_p = [0.20, 0.10, 0.45, 0.02, 0.15, 0.08]
        elif is_winter and is_north:
            weather_p = [0.30, 0.20, 0.05, 0.35, 0.02, 0.08]
        else:
            weather_p = [0.55, 0.25, 0.08, 0.02, 0.02, 0.08]
        
        weather = np.random.choice(WEATHER_CONDITIONS, p=weather_p)

        # Temperature in Fahrenheit (60F ~ 15C, 105F ~ 40C)
        if is_winter and is_north:
            temp_f = round(random.uniform(45.0, 68.0), 1)
        elif is_monsoon:
            temp_f = round(random.uniform(75.0, 92.0), 1)
        else:
            temp_f = round(random.uniform(72.0, 106.0), 1)

        # Humidity
        if is_monsoon or weather == "Rain" or weather == "Thunderstorm":
            humidity = round(random.uniform(75.0, 98.0), 1)
        elif weather == "Fog":
            humidity = round(random.uniform(85.0, 100.0), 1)
        else:
            humidity = round(random.uniform(30.0, 65.0), 1)

        # Visibility in miles
        if weather == "Fog":
            visibility = round(random.uniform(0.1, 1.2), 1)
        elif weather in ["Rain", "Thunderstorm"]:
            visibility = round(random.uniform(1.0, 3.5), 1)
        elif weather == "Haze":
            visibility = round(random.uniform(2.5, 5.0), 1)
        else:
            visibility = round(random.uniform(6.0, 10.0), 1)

        # Vehicle involvement
        vehicle = random.choice(VEHICLE_TYPES)

        # Severity determination logic based on realistic risk factors
        # 1: Minor fender bender / bumper damage / low speed
        # 2: Moderate damage / partial road block
        # 3: Serious injury / high speed impact / rollover
        # 4: Severe / Fatal / major expressway pileup
        severity_score = 1.6 # base baseline

        if road_type == "Expressway":
            severity_score += 0.8 # higher speeds lead to worse crashes
        elif road_type == "National Highway":
            severity_score += 0.5
        elif road_type == "Flyover":
            severity_score += 0.3

        if weather in ["Fog", "Thunderstorm"]:
            severity_score += 0.7
        elif weather == "Rain":
            severity_score += 0.4

        if hour in [0, 1, 2, 3, 4, 5]: # midnight to dawn
            severity_score += 0.6
        elif hour in [19, 20, 21, 22]: # peak traffic
            severity_score += 0.2

        if vehicle in ["Heavy Commercial / Truck", "Two-Wheeler"]:
            severity_score += 0.4

        if visibility < 2.0:
            severity_score += 0.5

        # Noise
        severity_score += np.random.normal(0, 0.5)

        # Discretize into 1, 2, 3, 4
        if severity_score < 2.0:
            severity = 1
        elif severity_score < 2.8:
            severity = 2
        elif severity_score < 3.5:
            severity = 3
        else:
            severity = 4

        # Primary cause
        if weather == "Fog" or (is_winter and is_north and visibility < 2.5):
            cause = "Low Visibility / Winter Fog"
        elif weather in ["Rain", "Thunderstorm"] or (is_monsoon and humidity > 80):
            cause = "Monsoon Hydroplaning / Wet Road"
        elif hour in [0, 1, 2, 3, 4, 5]:
            cause = "Driver Fatigue / Late Night Driving"
        elif road_type == "Expressway":
            cause = random.choice(["Overspeeding", "Sudden Braking / Tailgating"])
        else:
            cause = random.choice(CAUSES)

        rec = {
            "ID": f"IND-{i+1:05d}",
            "Severity": severity,
            "Start_Time": accident_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Start_Lat": round(lat, 6),
            "Start_Lng": round(lng, 6),
            "Temperature(F)": temp_f,
            "Humidity(%)": humidity,
            "Visibility(mi)": visibility,
            "Weather_Condition": weather,
            "City": hub["city"],
            "State": hub["state"],
            "Corridor": corridor,
            "Road_Type": road_type,
            "Primary_Cause": cause,
            "Vehicle_Type": vehicle
        }
        records.append(rec)

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    df = generate_records(4500)
    out_path = "data/india_accidents.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} realistic Indian accident records at {out_path}")
    print("Severity distribution:")
    print(df["Severity"].value_counts(normalize=True).round(3))
    print("\nTop cities:")
    print(df["City"].value_counts().head(5))
