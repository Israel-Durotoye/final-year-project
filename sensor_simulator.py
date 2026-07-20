import os
import time
import random
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

URL: str = os.environ.get("VITE_SUPABASE_URL")
KEY: str = os.environ.get("VITE_SUPABASE_ANON_KEY")

if not URL or not KEY:
    raise ValueError("Supabase credentials missing from .env file.")

supabase: Client = create_client(URL, KEY)

# --- Crop Profiles for Meaningful Classification Data ---
# Bounding the environmental variables so the LSTM has distinct patterns to learn
CROP_PROFILES = {
    "Maize": {
        "temp": (20.0, 30.0), "moisture": (50.0, 70.0), "humidity": (55.0, 75.0),
        "n": (100, 150), "p": (30, 50), "k": (80, 120)
    },
    "Cassava": {
        "temp": (25.0, 35.0), "moisture": (40.0, 60.0), "humidity": (50.0, 70.0),
        "n": (50, 90), "p": (10, 30), "k": (100, 150)
    },
    "Rice": {
        "temp": (22.0, 32.0), "moisture": (80.0, 100.0), "humidity": (70.0, 90.0),
        "n": (80, 120), "p": (20, 40), "k": (30, 60)
    }
}

# --- Node Setup ---
NODES = {
    "NODE_01": {"lat": 8.4810, "lng": 4.5410, "crop": "Maize"},
    "NODE_02": {"lat": 8.4815, "lng": 4.5415, "crop": "Maize"},
    "NODE_03": {"lat": 8.4820, "lng": 4.5420, "crop": "Cassava"},
    "NODE_04": {"lat": 8.4825, "lng": 4.5425, "crop": "Cassava"},
    "NODE_05": {"lat": 8.4830, "lng": 4.5430, "crop": "Rice"},
    "NODE_06": {"lat": 8.4835, "lng": 4.5435, "crop": "Rice"},
}

print("🌱 Starting upgraded capstone hardware simulation... Press Ctrl+C to stop.")

try:
    while True:
        for node_id, config in NODES.items():
            profile = CROP_PROFILES[config["crop"]]
            
            # Structuring the payload to match the exact headers in your capstone_dataset
            data = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Node_ID": node_id,
                "Moisture_%": round(random.uniform(*profile["moisture"]), 1),
                "Temperature_C": round(random.uniform(*profile["temp"]), 1),
                "Humidity_%": round(random.uniform(*profile["humidity"]), 1),
                "Nitrogen_mg_k": round(random.uniform(*profile["n"]), 2),
                "Phosphorus_m": round(random.uniform(*profile["p"]), 2),
                "Potassium_mg_": round(random.uniform(*profile["k"]), 2),
                "Latitude": config["lat"],
                "Longitude": config["lng"],
                "Altitude_m": round(random.uniform(295.0, 305.0), 1),
                "Satellites": random.choice([4, 5, 6, 7, 8, "ERR"]),
                "Season": "Dry",
                "Target_Crop": config["crop"]
            }
            

            supabase.table("capstone_dataset").insert(data).execute()
            print(f"📡 Sent {config['crop']} telemetry for {node_id}")
        
        print("⏳ Waiting 60 seconds for next transmission...")
        time.sleep(60)

except KeyboardInterrupt:
    print("\nSimulation stopped.")