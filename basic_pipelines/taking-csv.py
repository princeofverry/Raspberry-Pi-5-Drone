import os
import sys
import time
import uuid
import csv
from pathlib import Path
from datetime import datetime

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import numpy as np
import cv2
from PIL import Image

import hailo

# -----------------------------------------------------------------------------------------------
# Firebase Setup
# -----------------------------------------------------------------------------------------------
import firebase_admin
from firebase_admin import credentials, storage

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# =========================
# Konfigurasi umum
# =========================
FIRE_LABELS = {"fire", "smoke"}
MIN_CONFIDENCE = 0.30
UPLOAD_COOLDOWN_SEC = 2.5
LOCAL_SAVE_DIR = "/tmp/detected_fire"
FIREBASE_BUCKET = "ardutofirebase.appspot.com"
CSV_PATH = os.path.join(LOCAL_SAVE_DIR, "detections.csv")


# =========================
# Util Firebase
# =========================
def init_firebase():
    try:
        cred_path = "firebase_credentials.json"
        if not os.path.exists(cred_path):
            print(f"❌ firebase_credentials.json tidak ditemukan di: {os.path.abspath(cred_path)}")
            return False

        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET})

        b = storage.bucket()
        if not b:
            print("❌ Gagal mendapatkan bucket Firebase.")
            return False

        print(f"✅ Firebase inisialisasi OK. Bucket: {b.name}")
        return True
    except Exception as e:
        print(f"❌ Error init Firebase: {e}")
        return False


def upload_file_to_firebase(local_path, remote_path, max_retries=3, backoff=1.0):
    try:
        b = storage.bucket()
        blob = b.blob(remote_path)
        blob.cache_control = "no-store"
        blob.content_type = "image/jpeg"

        attempt = 0
        while attempt < max_retries:
            try:
                start = time.time()
                blob.upload_from_filename(local_path)
                latency = time.time() - start

                print(f"✅ Upload OK -> {remote_path} (latency {latency:.3f}s)")
                return True, latency
            except Exception as e:
                attempt += 1
                print(f"⚠️ Upload gagal (attempt {attempt}/{max_retries}): {e}")
                time.sleep(backoff * attempt)

        print("❌ Upload gagal setelah retry.")
        return False, -1
    except Exception as e:
        print(f"❌ Gagal menyiapkan upload: {e}")
        return False, -1


# =========================
# Util CSV
# =========================
def init_csv():
    os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "confidence", "upload_ok", "latency_sec"])


def append_to_csv(confidence, upload_ok, latency):
    with open(CSV_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"{confidence:.2f}",
            "yes" if upload_ok else "no",
            f"{latency:.3f}"
        ])


# -----------------------------------------------------------------------------------------------
# User-defined class
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.use_frame = True
        self._last_upload_ts = 0.0

    def can_upload_now(self):
        now = time.time()
        if now - self._last_upload_ts >= UPLOAD_COOLDOWN_SEC:
            self._last_upload_ts = now
            return True
        return False


# -----------------------------------------------------------------------------------------------
# Helper ambil frame
# -----------------------------------------------------------------------------------------------
def extract_frame_with_fallback(buffer, format_str, width, height):
    try:
        buf_size = buffer.get_size()
        if buf_size <= 0:
            return None

        data = buffer.extract_dup(0, buf_size)
        arr = np.frombuffer(data, dtype=np.uint8)
        fmt = (format_str or "").lower()

        if fmt in ("rgb", "bgr"):
            expected = width * height * 3
            if arr.size >= expected:
                return arr[:expected].reshape((height, width, 3))
        elif fmt in ("nv12", "i420", "yuv"):
            try:
                yuv = arr.reshape((height * 3 // 2, width))
                return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)
            except Exception as e:
                print(f"❌ Konversi YUV gagal: {e}")

        expected = width * height * 3
        if arr.size >= expected:
            return arr[:expected].reshape((height, width, 3))
    except Exception as e:
        print(f"❌ extract_frame_with_fallback gagal: {e}")
    return None


# -----------------------------------------------------------------------------------------------
# Callback function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data: user_app_callback_class):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()

    try:
        format_str, width, height = get_caps_from_pad(pad)
    except Exception as e:
        print(f"⚠️ get_caps_from_pad error: {e}")
        format_str, width, height = None, None, None

    try:
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    except Exception:
        detections = []

    frame_cached = None

    for det in detections:
        try:
            label = str(det.get_label() or "").strip()
            confidence = float(det.get_confidence() or 0.0)
            print(f"🎯 Deteksi: {label} {confidence:.2f}")

            if label.lower() in FIRE_LABELS and confidence >= MIN_CONFIDENCE:
                if frame_cached is None:
                    frame_cached = extract_frame_with_fallback(buffer, format_str, width, height)

                if frame_cached is not None and user_data.can_upload_now():
                    try:
                        draw_frame = frame_cached.copy()

                        # Gambar bounding box
                        try:
                            bbox_obj = det.get_bbox()
                            x_min = int(bbox_obj.xmin() * width)
                            y_min = int(bbox_obj.ymin() * height)
                            x_max = int(bbox_obj.xmax() * width)
                            y_max = int(bbox_obj.ymax() * height)

                            cv2.rectangle(draw_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                            cv2.putText(draw_frame, f"{label} {confidence:.2f}",
                                        (x_min, max(0, y_min - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        except Exception as e:
                            print(f"⚠️ Gagal gambar bbox: {e}")

                        capture_ts = time.time()
                        img_pil = Image.fromarray(draw_frame)
                        os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)
                        filename = f"fire_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
                        local_path = os.path.join(LOCAL_SAVE_DIR, filename)

                        img_pil.save(local_path, format="JPEG", quality=95)
                        print(f"📸 Disimpan: {local_path}")

                        ok, upload_latency = upload_file_to_firebase(local_path, f"detected_fire/{filename}")
                        latency_total = time.time() - capture_ts
                        print(f"⏱️ Total latency (capture→upload): {latency_total:.3f}s")

                        append_to_csv(confidence, ok, latency_total)
                    except Exception as e:
                        print(f"❌ Error simpan/upload: {e}")
        except Exception as e:
            print(f"⚠️ Error deteksi: {e}")

    return Gst.PadProbeReturn.OK


# -----------------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        project_root = Path(__file__).resolve().parent.parent
    except Exception:
        project_root = Path.cwd()

    os.environ["HAILO_ENV_FILE"] = str(project_root / ".env")
    print(f"🔧 HAILO_ENV_FILE = {os.environ['HAILO_ENV_FILE']}")

    fb_ok = init_firebase()
    if not fb_ok:
        print("⚠️ Firebase tidak siap")

    init_csv()

    try:
        Gst.init(None)
    except Exception as e:
        print(f"❌ GStreamer init error: {e}")
        sys.exit(1)

    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    print("🚀 Starting GStreamerDetectionApp ...")
    app.run()
