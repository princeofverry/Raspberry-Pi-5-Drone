#!/usr/bin/env python3
"""
detect_and_upload.py

Refactor skrip deteksi + upload:
- Logging terstruktur
- Konfigurasi via argparse / env
- Upload non-blocking (ThreadPoolExecutor)
- Thread-safe cooldown
- Resize gambar sebelum upload
- Penanganan sinyal untuk shutdown bersih
- parse_known_args() agar argumen tak dikenal diabaikan
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import gi
import numpy as np
from PIL import Image

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402  (import setelah gi.require_version)

import hailo  # noqa: E402

# Firebase
import firebase_admin  # noqa: E402
from firebase_admin import credentials, storage  # noqa: E402

# Hailo apps / GStreamer utils
from hailo_apps.hailo_app_python.core.common.buffer_utils import (  # noqa: E402
    get_caps_from_pad,
)
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import (  # noqa: E402
    app_callback_class,
)
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import (  # noqa: E402
    GStreamerDetectionApp,
)

# -----------------------------------------------------------------------------
# Konstanta & konfigurasi default
# -----------------------------------------------------------------------------
DEFAULT_FIRE_LABELS = {"fire", "smoke"}
DEFAULT_MIN_CONFIDENCE = 0.30
DEFAULT_UPLOAD_COOLDOWN = 2.5
DEFAULT_LOCAL_SAVE_DIR = "/tmp/detected_fire"
DEFAULT_FIREBASE_BUCKET = "drone-monitoring-system-fef66.firebasestorage.app"
DEFAULT_MAX_WORKERS = 2
DEFAULT_MAX_IMAGE_SIZE = 800  # px, sisi terbesar saat resize agar upload cepat

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logger = logging.getLogger("detect_upload")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# -----------------------------------------------------------------------------
# Firebase helpers
# -----------------------------------------------------------------------------
def init_firebase(cred_path: str, bucket_name: str) -> bool:
    """
    Inisialisasi Firebase Admin SDK. cred_path wajib ada.
    """
    try:
        p = Path(cred_path)
        if not p.exists():
            logger.error("Firebase credential file not found: %s", p)
            return False

        if not firebase_admin._apps:
            cred = credentials.Certificate(str(p))
            firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})

        b = storage.bucket()
        logger.info("Firebase inisialisasi OK. Bucket: %s", b.name)
        return True
    except Exception as e:
        logger.exception("init_firebase failed: %s", e)
        return False


def upload_file_to_firebase(
    local_path: str,
    remote_path: str,
    max_retries: int = 3,
    backoff: float = 1.0,
) -> bool:
    """
    Upload file ke Firebase Storage. Return True jika berhasil.
    Cocok dipanggil dari thread (non-blocking di caller).
    """
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

                # opsional: jadikan public (pahami implikasi keamanan)
                try:
                    blob.make_public()
                    public_url = blob.public_url
                except Exception:
                    public_url = None

                logger.info(
                    "Upload OK -> %s (%.3fs) public=%s",
                    remote_path,
                    latency,
                    bool(public_url),
                )
                if public_url:
                    logger.debug("Public URL: %s", public_url)
                return True
            except Exception as e:
                attempt += 1
                logger.warning("Upload attempt %d failed: %s", attempt, e)
                time.sleep(backoff * attempt)

        logger.error("Upload failed after %d attempts", max_retries)
        return False

    except Exception as e:
        logger.exception("upload_file_to_firebase preparation failed: %s", e)
        return False


# -----------------------------------------------------------------------------
# Helper: ekstrak frame
# -----------------------------------------------------------------------------
def extract_frame_with_fallback(
    buffer,
    format_str: Optional[str],
    width: Optional[int],
    height: Optional[int],
) -> Optional[np.ndarray]:
    """
    Ekstrak frame dari buffer.
    Mengembalikan ndarray RGB (H, W, 3) atau None.
    """
    try:
        if width is None or height is None:
            logger.debug("No width/height provided")
            return None

        buf_size = buffer.get_size()
        if buf_size <= 0:
            return None

        data = buffer.extract_dup(0, buf_size)
        arr = np.frombuffer(data, dtype=np.uint8)
        fmt = (format_str or "").lower()

        # Direct RGB raw
        if fmt in ("rgb", "rgb24"):
            expected = width * height * 3
            if arr.size >= expected:
                return arr[:expected].reshape((height, width, 3))

        # BGR raw -> konversi ke RGB
        if fmt in ("bgr", "bgr24"):
            expected = width * height * 3
            if arr.size >= expected:
                frame = arr[:expected].reshape((height, width, 3))
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # NV12 / YUV420 semi-planar
        if fmt == "nv12":
            try:
                yuv = arr.reshape((height * 3 // 2, width))
                return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)
            except Exception as e:
                logger.debug("NV12 conversion failed: %s", e)

        # Fallback: coba interpretasi sebagai RGB packed
        expected = width * height * 3
        if arr.size >= expected:
            return arr[:expected].reshape((height, width, 3))

    except Exception as e:
        logger.exception("extract_frame_with_fallback failed: %s", e)

    return None


# -----------------------------------------------------------------------------
# User callback class (thread-safe cooldown)
# -----------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self, upload_cooldown: float = DEFAULT_UPLOAD_COOLDOWN) -> None:
        super().__init__()
        self.use_frame = True
        self._last_upload_ts = 0.0
        self._lock = threading.Lock()
        self.upload_cooldown = float(upload_cooldown)

    def can_upload_now(self) -> bool:
        now = time.time()
        with self._lock:
            if now - self._last_upload_ts >= self.upload_cooldown:
                self._last_upload_ts = now
                return True
        return False


# -----------------------------------------------------------------------------
# Utility: resize & save image
# -----------------------------------------------------------------------------
def save_image_jpg(
    np_rgb: np.ndarray,
    path: str,
    max_side: int = DEFAULT_MAX_IMAGE_SIZE,
    quality: int = 85,
) -> None:
    """
    Resize dengan mempertahankan rasio, sisi max == max_side, simpan JPEG.
    Input np_rgb diharapkan RGB.
    """
    img = Image.fromarray(np_rgb)
    w, h = img.size
    max_current = max(w, h)

    if max_current > max_side:
        scale = max_side / float(max_current)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)

    img.save(path, format="JPEG", quality=quality)


# -----------------------------------------------------------------------------
# Callback utama (dipanggil oleh GStreamer)
# -----------------------------------------------------------------------------
def app_callback(
    pad,
    info,
    user_data: user_app_callback_class,
    *,
    fire_labels: set[str] = DEFAULT_FIRE_LABELS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    local_save_dir: str = DEFAULT_LOCAL_SAVE_DIR,
    executor: Optional[ThreadPoolExecutor] = None,
):
    """
    Dipanggil setiap kali ada buffer. Harus ringan — upload dipindahkan ke threadpool.
    """
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()

    # Ambil caps (format, width, height)
    try:
        format_str, width, height = get_caps_from_pad(pad)
    except Exception as e:
        logger.debug("get_caps_from_pad error: %s", e)
        format_str, width, height = None, None, None

    # Ambil detections dari buffer
    try:
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    except Exception:
        detections = []

    fire_triggered = False
    frame_cached: Optional[np.ndarray] = None
    detection_count = len(detections)

    for det in detections:
        try:
            label = str(det.get_label() or "").strip()
            confidence = float(det.get_confidence() or 0.0)
            logger.debug("Detected: %s %.3f", label, confidence)

            if label.lower() in fire_labels and confidence >= min_confidence:
                fire_triggered = True

                if frame_cached is None:
                    frame_cached = extract_frame_with_fallback(
                        buffer, format_str, width, height
                    )

                if frame_cached is not None:
                    draw_frame = frame_cached.copy()

                    # Gambar bbox + label
                    try:
                        bbox = det.get_bbox()
                        x_min = int(bbox.xmin() * width)
                        y_min = int(bbox.ymin() * height)
                        x_max = int(bbox.xmax() * width)
                        y_max = int(bbox.ymax() * height)

                        bgr = cv2.cvtColor(draw_frame, cv2.COLOR_RGB2BGR)
                        cv2.rectangle(bgr, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                        cv2.putText(
                            bgr,
                            f"{label} {confidence:.2f}",
                            (x_min, max(0, y_min - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )
                        draw_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        logger.debug("Failed to draw bbox: %s", e)

                    # Cooldown upload
                    if user_data.can_upload_now():
                        try:
                            os.makedirs(local_save_dir, exist_ok=True)
                            filename = (
                                f"fire_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
                                f"_{uuid.uuid4().hex[:6]}.jpg"
                            )
                            local_path = os.path.join(local_save_dir, filename)

                            # Resize & simpan
                            save_image_jpg(draw_frame, local_path)
                            logger.info("Saved frame for upload: %s", local_path)

                            remote_path = f"detected_fire/{filename}"

                            if executor:
                                fut: Future = executor.submit(
                                    upload_file_to_firebase, local_path, remote_path
                                )

                                def _on_done(f: Future) -> None:
                                    try:
                                        ok = f.result()
                                        logger.info(
                                            "Upload finished %s -> %s",
                                            remote_path,
                                            "OK" if ok else "FAILED",
                                        )
                                    except Exception as exc:
                                        logger.exception("Upload raised: %s", exc)

                                fut.add_done_callback(_on_done)
                            else:
                                ok = upload_file_to_firebase(local_path, remote_path)
                                logger.info("Blocking upload result: %s", ok)
                        except Exception as e:
                            logger.exception("Saving/uploading failed: %s", e)

        except Exception as e:
            logger.debug("Error processing detection: %s", e)

    # Overlay preview (non-blocking)
    if user_data.use_frame:
        preview = frame_cached or extract_frame_with_fallback(
            buffer, format_str, width, height
        )
        if preview is not None:
            try:
                bgr_preview = cv2.cvtColor(preview.copy(), cv2.COLOR_RGB2BGR)
                cv2.putText(
                    bgr_preview,
                    f"Detections: {detection_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                # GStreamer app biasanya mengharapkan BGR
                user_data.set_frame(bgr_preview)
            except Exception as e:
                logger.debug("Overlay preview failed: %s", e)

    if not fire_triggered:
        logger.debug("No valid fire detection in this buffer")

    return Gst.PadProbeReturn.OK


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Detection + upload to Firebase")
    ap.add_argument(
        "--firebase-cred",
        default=os.environ.get("FIREBASE_CRED", "firebase_credentials.json"),
        help="Path ke firebase credential JSON (env FIREBASE_CRED)",
    )
    ap.add_argument(
        "--bucket",
        default=os.environ.get("FIREBASE_BUCKET", DEFAULT_FIREBASE_BUCKET),
        help="Nama bucket Firebase Storage",
    )
    ap.add_argument(
        "--local-dir",
        default=os.environ.get("LOCAL_SAVE_DIR", DEFAULT_LOCAL_SAVE_DIR),
        help="Direktori sementara untuk menyimpan gambar sebelum upload",
    )
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=float(os.environ.get("MIN_CONF", DEFAULT_MIN_CONFIDENCE)),
        help="Ambang kepercayaan deteksi",
    )
    ap.add_argument(
        "--cooldown",
        type=float,
        default=float(os.environ.get("UPLOAD_COOLDOWN", DEFAULT_UPLOAD_COOLDOWN)),
        help="Cooldown upload (detik)",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=int(os.environ.get("MAX_WORKERS", DEFAULT_MAX_WORKERS)),
        help="Jumlah worker untuk threadpool upload",
    )

    # Gunakan parse_known_args supaya skrip tidak crash karena argumen tak dikenal
    args, unknown = ap.parse_known_args()
    if unknown:
        logger.debug("Ignoring unknown CLI args: %s", unknown)

    logger.info("Starting detect_and_upload (pid=%d)", os.getpid())

    # Firebase
    fb_ok = init_firebase(args.firebase_cred, args.bucket)
    if not fb_ok:
        logger.warning("Firebase not ready. Uploads may fail (script tetap berjalan).")

    # GStreamer init
    try:
        Gst.init(None)
    except Exception as e:
        logger.exception("GStreamer init failed: %s", e)
        sys.exit(1)

    user_data = user_app_callback_class(upload_cooldown=args.cooldown)
    executor = ThreadPoolExecutor(max_workers=args.max_workers)

    # Graceful shutdown on SIGINT/SIGTERM
    stop_event = threading.Event()

    def _signal(signum, _frame):
        logger.info("Signal %s received, shutting down...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _signal)
    signal.signal(signal.SIGTERM, _signal)

    # Bungkus callback untuk menyuntikkan executor & config
    def gst_cb(pad, info, ud=user_data):
        return app_callback(
            pad,
            info,
            ud,
            fire_labels=DEFAULT_FIRE_LABELS,
            min_confidence=args.min_confidence,
            local_save_dir=args.local_dir,
            executor=executor,
        )

    app = GStreamerDetectionApp(gst_cb, user_data)
    logger.info("Starting GStreamerDetectionApp ...")
    try:
        app.run()  # blocking (GStreamer main loop)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, exiting")
    finally:
        logger.info("Shutting down executor...")
        executor.shutdown(wait=True)
        logger.info("Exit")


if __name__ == "__main__":
    main()
