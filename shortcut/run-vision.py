# all code
# Verry Kurniawan 21120122130062
# Finodya Yahdun 21120122130065
# Imam Baihaqqy 21120122130078

#!/usr/bin/env python3
"""
run_both.py
Menjalankan detect dan streaming bersamaan (dua proses). 
Menangani Ctrl+C untuk menghentikan keduanya secara bersih.
"""
import subprocess
import os
import signal
import sys
import time

# konfigurasi default (ubah jika perlu)
DETECT_DIR = os.path.expanduser('~/Desktop/drone/hailo-rpi5-examples')
STREAM_DIR = os.path.expanduser('~/Desktop/drone/all-program')

DETECT_CMD = (
    'source setup_env.sh && '
    'python basic_pipelines/taking-pict.py '
    '--input rpicam '
    '--labels labels.json '
    '--hef-path model-hailo-22data.hef'
)
STREAM_CMD = 'python streaming.py'

def start_process(cmd, cwd):
    # jalankan setiap perintah dalam bash -lc agar `source` berfungsi jika ada
    return subprocess.Popen(['bash', '-lc', cmd], cwd=cwd, stdout=None, stderr=None, preexec_fn=os.setsid)

def main():
    print("Starting detect + streaming...")
    # start detect
    p_detect = start_process(DETECT_CMD, DETECT_DIR)
    print(f"Detect PID: {p_detect.pid}")
    # start streaming
    p_stream = start_process(STREAM_CMD, STREAM_DIR)
    print(f"Streaming PID: {p_stream.pid}")

    def terminate(proc, name):
        if proc and proc.poll() is None:
            print(f"Terminating {name} (pid {proc.pid})")
            try:
                # kill process group supaya child juga mati
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception as e:
                print(f"Error terminating {name}: {e}", file=sys.stderr)

    try:
        # tunggu kedua proses selesai; cek secara berkala
        while True:
            if p_detect.poll() is not None:
                print(f"Detect exited with {p_detect.returncode}")
                # kalau salah satu keluar, kita hentikan yang lain
                terminate(p_stream, 'streaming')
                break
            if p_stream.poll() is not None:
                print(f"Streaming exited with {p_stream.returncode}")
                terminate(p_detect, 'detect')
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received — stopping both processes...")
        terminate(p_detect, 'detect')
        terminate(p_stream, 'streaming')

if __name__ == "__main__":
    main()
