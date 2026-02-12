#!/usr/bin/env python3
import io
import logging
import socketserver
from http import server
from threading import Condition, Thread
import subprocess
import time
import cv2

# Halaman web HTML
PAGE = """\
<html>
<head>
<title>Webcam MJPEG Streaming Demo</title>
</head>
<body>
<h1>Webcam MJPEG Streaming Demo</h1>
<img src="stream.mjpg" width="1280" height="720" />
</body>
</html>
"""

# Buffer streaming
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        # buf is expected to be JPEG bytes
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

# HTTP Handler
class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    if frame:
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(frame))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning('Removed streaming client %s: %s', self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

# HTTP Server dengan threading
class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

# --- Start Cloudflared Tunnel (optional) ---
# Pastikan cloudflared sudah terinstall dan konfigurasi tunnel 'raspi5' ada.
cf_process = subprocess.Popen(
    ["cloudflared", "tunnel", "run", "raspi5"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT
)

# --- Setup Webcam capture (OpenCV) ---
# Ubah device_index jika perlu (0 = /dev/video0, 1 = /dev/video1, dst)
device_index = 0
width = 1280
height = 720
fps = 20  # target frame rate

cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
# Set resolusi (some webcams ignore unsupported resolutions)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)

if not cap.isOpened():
    print(f"[ERROR] Tidak bisa membuka device video index {device_index}. Cek `v4l2-ctl --list-devices`.")
    cf_process.terminate()
    raise SystemExit(1)

output = StreamingOutput()

def capture_loop():
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Frame read failed, retrying...")
                time.sleep(0.1)
                continue
            # Optional: resize to target if camera didn't honor settings
            frame = cv2.resize(frame, (width, height))

            # Encode to JPEG
            ret2, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret2:
                logging.warning("Failed to encode frame")
                continue
            jpeg_bytes = jpeg.tobytes()
            # Tulis ke output buffer
            output.write(jpeg_bytes)

            # Kontrol frame rate sederhana
            time.sleep(1.0 / fps)
    except Exception as e:
        logging.exception("Error di capture loop: %s", e)

# Jalankan capture di thread terpisah
capture_thread = Thread(target=capture_loop, daemon=True)
capture_thread.start()

try:
    address = ('', 7123)
    server = StreamingServer(address, StreamingHandler)
    print("Server jalan di port 7123. Akses http://<raspi-ip>:7123")
    print("Cloudflared tunnel juga sudah dijalankan...")
    server.serve_forever()
finally:
    print("Membersihkan...")
    cap.release()
    cf_process.terminate()
