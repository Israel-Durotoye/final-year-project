import os
import time
import random
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

URL: str = os.environ.get("SUPABASE_URL")
KEY: str = os.environ.get("SUPABASE_KEY")

if not URL or not KEY:
    raise ValueError("Supabase credentials missing from .env file.")

supabase: Client = create_client(URL, KEY)

# Assign static GPS coordinates to your nodes (based on your screenshot)
NODES = {
    "NODE_01": {"lat": 36.7783, "lng": -119.4179},
    "NODE_02": {"lat": 36.7790, "lng": -119.4150},
    "NODE_03": {"lat": 36.7820, "lng": -119.4100},
    "NODE_04": {"lat": 36.7750, "lng": -119.4200},
    "NODE_05": {"lat": 36.7800, "lng": -119.4250},
    "NODE_06": {"lat": 36.7810, "lng": -119.4220},
}

print("🌱 Starting upgraded hardware simulation... Press Ctrl+C to stop.")

try:
    while True:
        for node_id, coords in NODES.items():
            data = {
                "node_id": node_id,
                "soil_moisture": round(random.uniform(15.0, 35.0), 1), # Already tracking this!
                "humidity": round(random.uniform(45.0, 80.0), 1),      # New ambient humidity
                "latitude": coords["lat"],                             # New static GPS
                "longitude": coords["lng"],                            # New static GPS
                "soil_temperature_c": round(random.uniform(22.0, 30.0), 1),
                "ph": round(random.uniform(5.5, 7.5), 1),
                "nitrogen_ppm": round(random.uniform(15.0, 40.0), 1),
                "phosphorus_ppm": round(random.uniform(30.0, 60.0), 1),
                "potassium_ppm": round(random.uniform(150.0, 250.0), 1),
                "battery_voltage": round(random.uniform(3.2, 4.2), 2),
                "communication_ok": True
            }
            
            supabase.table("sensor_telemetry").insert(data).execute()
            print(f"📡 Sent telemetry for {node_id} (Inc. Hum/Lat/Lng)")
        
        print("⏳ Waiting 60 seconds for next transmission...")
        time.sleep(60)

except KeyboardInterrupt:
    print("\nSimulation stopped.")