import os
import sys
import time
from pathlib import Path
import csv
from datetime import datetime

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import numpy as np
import cv2
import hailo

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp


# =========================
# Konfigurasi umum
# =========================
FIRE_LABELS = {"fire", "smoke"}
MIN_CONFIDENCE = 0.50
MIN_BBOX_AREA_RATIO = 0.001

LABEL_MAP = {
    0: "fire",
    1: "smoke",
}

DEBUG_PRINT_ALL = False
DEFAULT_CSV_PATH = "/tmp/hailo_detection_log_pengujian.csv"


# =====================================================================================
# User callback class
# =====================================================================================
class user_app_callback_class(app_callback_class):
    def __init__(self, csv_path=DEFAULT_CSV_PATH):
        super().__init__()

        self.frame_count = 0
        self.start_time = None
        self.last_frame_time = None

        self.csv_path = csv_path
        file_exists = os.path.exists(self.csv_path)

        self.csv_file = open(self.csv_path, "a", newline="", buffering=1)
        self.csv_writer = csv.writer(self.csv_file)

        if not file_exists:
            self.csv_writer.writerow([
                "timestamp_iso",
                "frame_index",
                "instant_fps",
                "avg_fps",
                "num_detections",
                "inference_time_ms",
                "postprocess_time_ms",
                "total_time_ms"
            ])

    def close(self):
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass


# =====================================================================================
# Helper ambil frame
# =====================================================================================
def extract_frame_with_fallback(buffer, format_str, width, height):
    try:
        data = buffer.extract_dup(0, buffer.get_size())
        arr = np.frombuffer(data, dtype=np.uint8)
        fmt = (format_str or "").lower()

        if fmt in ("rgb", "bgr"):
            return arr[:width * height * 3].reshape((height, width, 3))

        if fmt in ("nv12", "i420"):
            yuv = arr.reshape((height * 3 // 2, width))
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)

    except Exception as e:
        print("Frame extract error:", e)

    return None


# =====================================================================================
# CALLBACK UTAMA (WAKTU TERCATAT LENGKAP)
# =====================================================================================
def app_callback(pad, info, user_data: user_app_callback_class):

    t_callback_start = time.time()  # ⏱ TOTAL FRAME TIME

    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    if user_data.start_time is None:
        user_data.start_time = t_callback_start
        user_data.last_frame_time = t_callback_start

    user_data.frame_count += 1
    frame_index = user_data.frame_count

    try:
        format_str, width, height = get_caps_from_pad(pad)
    except Exception:
        format_str, width, height = None, None, None

    # =========================
    # ⏱ INFERENCE TIME (HAILO)
    # =========================
    t_infer_start = time.time()
    try:
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    except Exception:
        detections = []
    t_infer_end = time.time()

    inference_time_ms = (t_infer_end - t_infer_start) * 1000.0

    # =========================
    # ⏱ POST PROCESS
    # =========================
    t_post_start = time.time()

    frame_area = (width * height) if width and height else None

    for det in detections:
        try:
            raw_label = det.get_label()
            label = LABEL_MAP.get(int(raw_label), str(raw_label)) if isinstance(raw_label, int) else str(raw_label)
            confidence = float(det.get_confidence() or 0.0)

            if not label or confidence < MIN_CONFIDENCE:
                continue

            bbox = det.get_bbox()
            area_rel = (bbox.xmax() - bbox.xmin()) * (bbox.ymax() - bbox.ymin())

            if frame_area and area_rel < MIN_BBOX_AREA_RATIO:
                continue

            print(f"Deteksi: {label} {confidence:.2f}")

        except Exception as e:
            print("Deteksi error:", e)

    t_post_end = time.time()
    postprocess_time_ms = (t_post_end - t_post_start) * 1000.0

    # =========================
    # ⏱ TOTAL TIME
    # =========================
    total_time_ms = (time.time() - t_callback_start) * 1000.0

    # =========================
    # FPS
    # =========================
    interval = t_callback_start - user_data.last_frame_time
    instant_fps = (1.0 / interval) if interval > 1e-6 else None

    elapsed_total = t_callback_start - user_data.start_time
    avg_fps = (frame_index / elapsed_total) if elapsed_total > 0 else None

    user_data.last_frame_time = t_callback_start

    # =========================
    # CSV LOGGING
    # =========================
    user_data.csv_writer.writerow([
        datetime.utcnow().isoformat() + "Z",
        frame_index,
        f"{instant_fps:.2f}" if instant_fps else "",
        f"{avg_fps:.2f}" if avg_fps else "",
        len(detections),
        f"{inference_time_ms:.3f}",
        f"{postprocess_time_ms:.3f}",
        f"{total_time_ms:.3f}"
    ])

    return Gst.PadProbeReturn.OK


# =====================================================================================
# MAIN
# =====================================================================================
if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent
    os.environ["HAILO_ENV_FILE"] = str(project_root / ".env")
    print("HAILO_ENV_FILE =", os.environ["HAILO_ENV_FILE"])

    Gst.init(None)

    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)

    print("Running Hailo Detection + Timing Logger")
    try:
        app.run()
    finally:
        user_data.close()
