import os
import time
import csv
from datetime import datetime

CSV_FILE = "suhu_cpu_raspberrypi.csv"

def get_cpu_temperature():
    temp = os.popen("vcgencmd measure_temp").readline()
    return temp.replace("temp=", "").replace("'C", "")

# Cek & buat header CSV kalau file belum ada
if not os.path.isfile(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Tanggal", "Waktu", "Suhu_CPU_Celcius"])

while True:
    suhu = get_cpu_temperature()
    now = datetime.now()
    
    tanggal = now.strftime("%Y-%m-%d")
    waktu = now.strftime("%H:%M:%S")

    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([tanggal, waktu, suhu])

    print(f"{tanggal} {waktu} | Suhu CPU: {suhu}°C")
    time.sleep(2)
