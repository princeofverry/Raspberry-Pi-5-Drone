#!/usr/bin/env python3
"""
run-all-sensor.py
Menjalankan thermal_cam.py, obstacle.py, dht22.py secara paralel
dan menampilkan log real-time (tanpa buffering).
"""

import subprocess
import os
import signal
import threading
import time
from datetime import datetime

BASE_DIR = os.path.expanduser("~/menamatkan-tekkom")
PYTHON = os.path.join(BASE_DIR, ".venv", "bin", "python")  # sesuaikan kalau beda
# jika kamu jalankan dari dalam .venv, bisa juga cukup "python"

SCRIPTS = {
#    "THERMAL": "thermal_cam.py",
    "OBSTACLE": "obstacle.py",
    "DHT22": "dht22.py"
}

COLORS = {
#    "THERMAL": "\033[94m",
    "OBSTACLE": "\033[93m",
    "DHT22": "\033[92m",
    "RESET": "\033[0m"
}

def reader_thread(name, proc):
    color = COLORS.get(name, COLORS["RESET"])
    # Baca baris demi baris; readline tidak menunggu sampai proses selesai kalau ada newline
    while True:
        line = proc.stdout.readline()
        if not line:
            # jika proses sudah selesai dan tak ada output lagi, hentikan
            if proc.poll() is not None:
                break
            # kecil sleep untuk menghindari busy loop
            time.sleep(0.01)
            continue
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{color}[{timestamp}] [{name}] {line.rstrip()}{COLORS['RESET']}")

def start_proc(name, script_path):
    # Pastikan interpreter pakai -u (unbuffered) dan sediakan env PYTHONUNBUFFERED
    cmd = [PYTHON, "-u", script_path]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )
    t = threading.Thread(target=reader_thread, args=(name, proc), daemon=True)
    t.start()
    return proc, t

def main():
    print("🚀 Menjalankan semua script di virtual environment...\n")
    processes = {}
    threads = []

    for name, script in SCRIPTS.items():
        script_path = os.path.join(BASE_DIR, script)
        if not os.path.exists(script_path):
            print(f"⚠️ File tidak ditemukan: {script_path}")
            continue
        print(f"▶ Menjalankan {script} ...")
        proc, t = start_proc(name, script_path)
        processes[name] = proc
        threads.append(t)

    print("\n✅ Semua script berjalan. Tekan CTRL+C untuk berhenti.\n")

    try:
        # tunggu sampai semua proses selesai atau CTRL+C
        while any(p.poll() is None for p in processes.values()):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n🛑 Dihentikan oleh user. Menghentikan semua proses...")
        for p in processes.values():
            try:
                p.send_signal(signal.SIGINT)
            except Exception:
                pass
        # beri waktu untuk shutdown rapi
        time.sleep(1)
    finally:
        for p in processes.values():
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
        print("✅ Semua proses dihentikan.")

if __name__ == "__main__":
    main()
