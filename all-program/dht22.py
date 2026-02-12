#!/usr/bin/env python3
# dht22_cam.py
import time
from datetime import datetime

import board
import adafruit_dht

# cloud libs (embedded)
from google.cloud import firestore
from google.oauth2 import service_account

# ----------------- KONFIGURASI CLOUD (ganti sebelum jalan) -----------------
SERVICE_ACCOUNT_PATH = "/home/kelompok20/menamatkan-tekkom/gcp-service-account.json"   # <-- ganti path ke file JSON kamu
FIRESTORE_COLLECTION = "sensors_env"
# -------------------------------------------------------------------------

INTERVAL = 2  # detik antara pembacaan

# ---- Inisialisasi Cloud Firestore ----
_creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_PATH)
fs_client = firestore.Client(credentials=_creds, project=_creds.project_id)

def add_sensor_auto_id(collection, payload):
    col_ref = fs_client.collection(collection)
    return col_ref.add(payload)

# ---- Inisialisasi DHT ----
dhtDevice = adafruit_dht.DHT22(board.D24)
print("📡 Membaca data DHT22 di GPIO 24... (Ctrl+C untuk berhenti)\n")

try:
    while True:
        try:
            temperature = dhtDevice.temperature
            humidity = dhtDevice.humidity
            ts = datetime.utcnow().isoformat() + "Z"

            if temperature is not None and humidity is not None:
                payload = {
                    "timestamp": ts,
                    "sensor_type": "dht22",
                    "temperature": float(temperature),
                    "humidity": float(humidity)
                }
                try:
                    add_sensor_auto_id(FIRESTORE_COLLECTION, payload)
                except Exception as e:
                    print("Gagal kirim ke Firestore:", e)
                print(f"[DHT22] {ts} T={temperature:.1f}°C H={humidity:.1f}%")
            else:
                print("⚠️ Gagal membaca data dari sensor (None).")

        except RuntimeError as error:
            # DHT22 kadang gagal baca, ini normal — retry singkat
            print("Error pembacaan (retry):", error)
            time.sleep(2)
            continue
        except Exception as e:
            print("Error tak terduga pada pembacaan DHT:", e)

        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\n🚪 Program dihentikan oleh pengguna.")
