import os
import sys
import time
from pathlib import Path

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
MIN_CONFIDENCE = 0.50  # naikkan untuk kurangi noise; ganti kalau mau lebih sensitif
MIN_BBOX_AREA_RATIO = 0.001  # abaikan bbox yang sangat kecil (sebagai rasio dari frame area)

# Mapping ID->label bila model mengembalikan integer label
LABEL_MAP = {
    0: "fire",
    1: "smoke",
    # tambahkan entri lain sesuai file label/model Anda jika perlu
}

# Toggle debug untuk melihat raw det object
DEBUG_PRINT_ALL = False


# -----------------------------------------------------------------------------------------------
# User-defined class
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()


# -----------------------------------------------------------------------------------------------
# Helper ambil frame (untuk bounding box)
# -----------------------------------------------------------------------------------------------
def extract_frame_with_fallback(buffer, format_str, width, height):
    try:
        buf_size = buffer.get_size()
        if buf_size <= 0:
            return None

        data = buffer.extract_dup(0, buf_size)
        arr = np.frombuffer(data, dtype=np.uint8)
        fmt = (format_str or "").lower()

        # Kasus RGB/BGR langsung
        if fmt in ("rgb", "bgr"):
            expected = width * height * 3
            if arr.size >= expected:
                return arr[:expected].reshape((height, width, 3))

        # Kasus NV12/I420/YUV
        if fmt in ("nv12", "i420", "yuv"):
            try:
                yuv = arr.reshape((height * 3 // 2, width))
                return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)
            except Exception as e:
                print(f"❌ Konversi YUV gagal: {e}")

        # Fallback: coba interpretasi sebagai RGB terus potong sesuai kebutuhan
        expected = width * height * 3
        if arr.size >= expected:
            return arr[:expected].reshape((height, width, 3))
    except Exception as e:
        print(f"❌ extract_frame_with_fallback gagal: {e}")
    return None


# -----------------------------------------------------------------------------------------------
# Callback function (deteksi saja)
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data: user_app_callback_class):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()

    try:
        format_str, width, height = get_caps_from_pad(pad)
    except Exception:
        format_str, width, height = None, None, None

    try:
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    except Exception:
        detections = []

    frame_cached = None
    frame_area = (width * height) if (width and height) else None

    for det in detections:
        try:
            raw_label = det.get_label()

            # Normalisasi label: dukung integer label mapping dan string
            if isinstance(raw_label, (int, np.integer)):
                label = LABEL_MAP.get(int(raw_label), str(raw_label))
            else:
                label = str(raw_label or "").strip()

            confidence = float(det.get_confidence() or 0.0)

            # Debug: print semua isi det jika aktif
            if DEBUG_PRINT_ALL:
                bbox_obj_for_debug = None
                try:
                    bbox_obj_for_debug = det.get_bbox()
                except Exception:
                    bbox_obj_for_debug = None
                print("DEBUG det raw:", repr(raw_label), "->", label, "conf:", confidence,
                      "bbox:", bbox_obj_for_debug)

            # Skip jika label kosong
            if not label:
                continue

            # Skip jika confidence di bawah threshold
            if confidence < MIN_CONFIDENCE:
                continue

            # Jika ada bbox, hitung area relatif dan skip jika sangat kecil (noise)
            try:
                bbox_obj = det.get_bbox()
                # bbox coordinates biasanya relatif [0..1]
                w_rel = max(0.0, float(bbox_obj.xmax() - bbox_obj.xmin()))
                h_rel = max(0.0, float(bbox_obj.ymax() - bbox_obj.ymin()))
                area_rel = w_rel * h_rel
            except Exception:
                area_rel = None

            if frame_area and area_rel is not None:
                if area_rel < MIN_BBOX_AREA_RATIO:
                    # terlalu kecil -> kemungkinan false positive
                    if DEBUG_PRINT_ALL:
                        print(f"DEBUG skip bbox kecil area_rel={area_rel:.6f}")
                    continue

            # Hanya print / proses deteksi valid (confidence >= threshold, label ada)
            print(f"🎯 Deteksi: {label} {confidence:.2f}")

            # Jika label termasuk fire/smoke -> gambar bbox
            if label.lower() in FIRE_LABELS:
                if frame_cached is None:
                    frame_cached = extract_frame_with_fallback(buffer, format_str, width, height)

                if frame_cached is not None:
                    draw_frame = frame_cached.copy()
                    try:
                        if 'bbox_obj' not in locals():
                            bbox_obj = det.get_bbox()
                        x_min = int(max(0, bbox_obj.xmin() * width))
                        y_min = int(max(0, bbox_obj.ymin() * height))
                        x_max = int(min(width - 1, bbox_obj.xmax() * width))
                        y_max = int(min(height - 1, bbox_obj.ymax() * height))

                        cv2.rectangle(draw_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                        cv2.putText(draw_frame, f"{label} {confidence:.2f}",
                                    (x_min, max(0, y_min - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        # Anda bisa menyimpan/stream frame di sini jika ingin:
                        # cv2.imwrite(f"/tmp/fire_{time.time():.3f}.jpg", draw_frame)

                        print("🔥 Fire-like detected dengan bounding box")
                    except Exception as e:
                        print(f"⚠️ Gagal gambar bbox: {e}")
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

    try:
        Gst.init(None)
    except Exception as e:
        print(f"❌ GStreamer init error: {e}")
        sys.exit(1)

    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    print("🚀 Starting GStreamerDetectionApp (Deteksi Saja)...")
    app.run()
