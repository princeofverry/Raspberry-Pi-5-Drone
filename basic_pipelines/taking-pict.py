#!/usr/bin/env python3
"""
detect_and_upload.py

Refactor skrip deteksi + upload + overlay GPS + metadata JSON:
- Logging terstruktur
- Konfigurasi via argparse / env
- Upload non-blocking (ThreadPoolExecutor)
- Thread-safe cooldown
- Resize gambar sebelum upload
- Penanganan sinyal untuk shutdown bersih
- parse_known_args() agar argumen tak dikenal diabaikan
- Pembacaan GPS (MAVLink) di thread terpisah, overlay info saat save gambar
- Simpan & upload metadata JSON (deteksi + GPS)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cv2
import gi
import numpy as np
from PIL import Image

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

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

DEFAULT_MAVLINK_PORT = "/dev/ttyACM0"
DEFAULT_MAVLINK_BAUD = 57600

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logger = logging.getLogger("detect_upload")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# -----------------------------------------------------------------------------
# Opsi MAVLink (opsional)
# -----------------------------------------------------------------------------
try:
    from pymavlink import mavutil  # type: ignore
    PYMAVLINK_AVAILABLE = True
except Exception:
    PYMAVLINK_AVAILABLE = False
    logger.warning("pymavlink tidak tersedia. Overlay GPS & metadata GPS tetap ada tapi akan N/A.")

@dataclass
class GpsSnapshot:
    lat: Optional[float] = None   # deg
    lon: Optional[float] = None   # deg
    alt_m: Optional[float] = None # meters
    sats: Optional[int] = None
    fix_type: Optional[int] = None
    eph: Optional[float] = None
    epv: Optional[float] = None
    pos_horiz_var: Optional[float] = None
    pos_vert_var: Optional[float] = None
    timestamp_iso: str = field(default_factory=lambda: datetime.utcnow().isoformat())

class GpsState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snap = GpsSnapshot()

    def update_from_gps_raw(
        self, lat_e7, lon_e7, alt_mm, sats, fix, eph, epv
    ) -> None:
        with self._lock:
            self._snap.lat = (lat_e7 / 1e7) if lat_e7 not in (None,) else None
            self._snap.lon = (lon_e7 / 1e7) if lon_e7 not in (None,) else None
            self._snap.alt_m = (alt_mm / 1000.0) if alt_mm not in (None, ) else None
            self._snap.sats = sats
            self._snap.fix_type = fix
            self._snap.eph = eph
            self._snap.epv = epv
            self._snap.timestamp_iso = datetime.utcnow().isoformat()

    def update_from_global_position(self, lat_e7, lon_e7, alt_mm) -> None:
        with self._lock:
            self._snap.lat = (lat_e7 / 1e7) if lat_e7 not in (None,) else None
            self._snap.lon = (lon_e7 / 1e7) if lon_e7 not in (None,) else None
            self._snap.alt_m = (alt_mm / 1000.0) if alt_mm not in (None, ) else None
            self._snap.timestamp_iso = datetime.utcnow().isoformat()

    def update_ekf(self, pos_h_var: Optional[float], pos_v_var: Optional[float]) -> None:
        with self._lock:
            self._snap.pos_horiz_var = pos_h_var
            self._snap.pos_vert_var = pos_v_var
            self._snap.timestamp_iso = datetime.utcnow().isoformat()

    def snapshot(self) -> GpsSnapshot:
        with self._lock:
            return GpsSnapshot(
                lat=self._snap.lat,
                lon=self._snap.lon,
                alt_m=self._snap.alt_m,
                sats=self._snap.sats,
                fix_type=self._snap.fix_type,
                eph=self._snap.eph,
                epv=self._snap.epv,
                pos_horiz_var=self._snap.pos_horiz_var,
                pos_vert_var=self._snap.pos_vert_var,
                timestamp_iso=self._snap.timestamp_iso,
            )

def gps_thread_fn(port: str, baud: int, state: GpsState, stop_evt: threading.Event) -> None:
    if not PYMAVLINK_AVAILABLE:
        logger.info("GPS thread tidak berjalan (pymavlink tidak tersedia).")
        return
    try:
        logger.info("Connect MAVLink: %s @ %s", port, baud)
        master = mavutil.mavlink_connection(port, baud=baud, autoreconnect=True)
        logger.info("Waiting heartbeat...")
        master.wait_heartbeat(timeout=10)
        logger.info("Heartbeat from sys=%s comp=%s", master.target_system, master.target_component)

        # minta stream (opsional)
        try:
            master.mav.request_data_stream_send(
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                2, 1
            )
        except Exception:
            pass

        while not stop_evt.is_set():
            msg = master.recv_match(blocking=True, timeout=1.0)
            if msg is None:
                continue
            t = msg.get_type()

            if t == "EKF_STATUS_REPORT":
                state.update_ekf(
                    getattr(msg, "pos_horiz_variance", None),
                    getattr(msg, "pos_vert_variance", None),
                )

            elif t in ("GPS_RAW_INT", "GPS2_RAW"):
                state.update_from_gps_raw(
                    getattr(msg, "lat", None),
                    getattr(msg, "lon", None),
                    getattr(msg, "alt", None),
                    getattr(msg, "satellites_visible", None),
                    getattr(msg, "fix_type", None),
                    getattr(msg, "eph", None),
                    getattr(msg, "epv", None),
                )

            elif t == "GLOBAL_POSITION_INT":
                state.update_from_global_position(
                    getattr(msg, "lat", None),
                    getattr(msg, "lon", None),
                    getattr(msg, "alt", None),
                )
    except Exception as e:
        logger.exception("GPS thread error: %s", e)
    finally:
        try:
            master.mav.request_data_stream_send(
                master.target_system, master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL, 0, 0
            )
        except Exception:
            pass
        try:
            master.close()
        except Exception:
            pass
        logger.info("GPS thread stopped.")

# -----------------------------------------------------------------------------
# Firebase helpers
# -----------------------------------------------------------------------------
def init_firebase(cred_path: str, bucket_name: str) -> bool:
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
    try:
        b = storage.bucket()
        blob = b.blob(remote_path)
        blob.cache_control = "no-store"
        # konten diset berdasarkan ekstensi sederhana
        if remote_path.lower().endswith(".jpg") or remote_path.lower().endswith(".jpeg"):
            blob.content_type = "image/jpeg"
        elif remote_path.lower().endswith(".json"):
            blob.content_type = "application/json"
        else:
            blob.content_type = "application/octet-stream"

        attempt = 0
        while attempt < max_retries:
            try:
                start = time.time()
                blob.upload_from_filename(local_path)
                latency = time.time() - start

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

        if fmt in ("rgb", "rgb24"):
            expected = width * height * 3
            if arr.size >= expected:
                return arr[:expected].reshape((height, width, 3))

        if fmt in ("bgr", "bgr24"):
            expected = width * height * 3
            if arr.size >= expected:
                frame = arr[:expected].reshape((height, width, 3))
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if fmt == "nv12":
            try:
                yuv = arr.reshape((height * 3 // 2, width))
                return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)
            except Exception as e:
                logger.debug("NV12 conversion failed: %s", e)

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
# Utility: resize & save image + overlay GPS
# -----------------------------------------------------------------------------
def overlay_gps_text(bgr_image: np.ndarray, gps: GpsSnapshot) -> np.ndarray:
    """
    Tulis info GPS di pojok kiri bawah gambar (BGR).
    """
    lines = []

    def fmt(v, fmtstr="{:.6f}"):
        return (fmtstr.format(v) if v is not None else "N/A")

    lines.append(f"UTC: {gps.timestamp_iso[:19]}Z")
    lines.append(f"LAT: {fmt(gps.lat)}  LON: {fmt(gps.lon)}")
    lines.append(f"ALT: {fmt(gps.alt_m, '{:.1f}')} m  SATS: {fmt(gps.sats, '{:d}') if gps.sats is not None else 'N/A'}  FIX: {fmt(gps.fix_type, '{:d}') if gps.fix_type is not None else 'N/A'}")
    lines.append(f"EPH: {fmt(gps.eph, '{:.2f}')}  EPV: {fmt(gps.epv, '{:.2f}')}  EKF H: {fmt(gps.pos_horiz_var, '{:.2f}')} V: {fmt(gps.pos_vert_var, '{:.2f}')}")

    img = bgr_image.copy()
    h, w = img.shape[:2]
    x = 10
    y_start = h - 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2

    # latar semi transparan
    text_heights = []
    max_width = 0
    for i, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
        text_heights.append(th)
        max_width = max(max_width, tw)

    box_height = sum(text_heights) + 10 + (len(lines)-1)*6
    box_width = max_width + 20
    y0 = y_start - box_height
    overlay = img.copy()
    cv2.rectangle(overlay, (x-5, y0-5), (x-5+box_width, y0-5+box_height), (0,0,0), -1)
    alpha = 0.45
    img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

    # tulis teks
    y = y0 + 20
    for line in lines:
        cv2.putText(img, line, (x, y), font, scale, (255,255,255), thickness, cv2.LINE_AA)
        y += text_heights[0] + 6

    return img

def save_image_jpg(
    np_rgb: np.ndarray,
    path: str,
    max_side: int = DEFAULT_MAX_IMAGE_SIZE,
    quality: int = 85,
) -> None:
    img = Image.fromarray(np_rgb)
    w, h = img.size
    max_current = max(w, h)

    if max_current > max_side:
        scale = max_side / float(max_current)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)

    img.save(path, format="JPEG", quality=quality)

# -----------------------------------------------------------------------------
# Utility: ringkas deteksi untuk JSON
# -----------------------------------------------------------------------------
def build_detection_summary(
    detections, width: int, height: int, fire_labels: set[str], min_conf: float
) -> Dict[str, object]:
    det_all: List[Dict[str, object]] = []
    det_fire: List[Dict[str, object]] = []
    for det in detections:
        try:
            label = str(det.get_label() or "").strip()
            confidence = float(det.get_confidence() or 0.0)

            bbox = det.get_bbox()
            # Normalized
            nxmin = float(bbox.xmin())
            nymin = float(bbox.ymin())
            nxmax = float(bbox.xmax())
            nymax = float(bbox.ymax())
            # Absolute
            xmin = int(nxmin * width)
            ymin = int(nymin * height)
            xmax = int(nxmax * width)
            ymax = int(nymax * height)

            entry = {
                "label": label,
                "confidence": confidence,
                "bbox_norm": {"xmin": nxmin, "ymin": nymin, "xmax": nxmax, "ymax": nymax},
                "bbox_abs": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
            }
            det_all.append(entry)
            if label.lower() in fire_labels and confidence >= min_conf:
                det_fire.append(entry)
        except Exception:
            continue

    # pilih deteksi fire terkuat (jika ada)
    top_fire = None
    if det_fire:
        top_fire = max(det_fire, key=lambda d: d.get("confidence", 0.0))

    return {
        "count_all": len(det_all),
        "count_fire": len(det_fire),
        "top_fire": top_fire,
        "detections": det_all,
    }

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
    gps_state: Optional[GpsState] = None,
    gps_enabled: bool = False,
):
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

    # siapkan summary deteksi (untuk JSON); butuh width/height valid
    det_summary = None
    if width and height:
        try:
            det_summary = build_detection_summary(
                detections, width, height, fire_labels, min_confidence
            )
        except Exception as e:
            logger.debug("build_detection_summary failed: %s", e)

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

                        # Overlay GPS sebelum save
                        snap = gps_state.snapshot() if (gps_enabled and gps_state) else GpsSnapshot()
                        bgr = overlay_gps_text(bgr, snap)

                        draw_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        logger.debug("Failed to draw bbox/GPS overlay: %s", e)

                    # Cooldown upload
                    if user_data.can_upload_now():
                        try:
                            os.makedirs(local_save_dir, exist_ok=True)
                            base = f"fire_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
                            jpgname = f"{base}.jpg"
                            jsonname = f"{base}.json"

                            local_img_path = os.path.join(local_save_dir, jpgname)
                            local_json_path = os.path.join(local_save_dir, jsonname)

                            # Resize & simpan IMG
                            save_image_jpg(draw_frame, local_img_path)
                            logger.info("Saved frame for upload: %s", local_img_path)

                            # Siapkan payload JSON
                            snap = gps_state.snapshot() if (gps_enabled and gps_state) else GpsSnapshot()
                            payload = {
                                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                                "image": {
                                    "filename": jpgname,
                                    "local_path": local_img_path,
                                    "remote_path": f"detected_fire/{jpgname}",
                                    "width": int(draw_frame.shape[1]),
                                    "height": int(draw_frame.shape[0]),
                                },
                                "trigger": {
                                    "labels": list(sorted(fire_labels)),
                                    "min_confidence": float(min_confidence),
                                },
                                "gps": {
                                    "lat": snap.lat,
                                    "lon": snap.lon,
                                    "alt_m": snap.alt_m,
                                    "sats": snap.sats,
                                    "fix_type": snap.fix_type,
                                    "eph": snap.eph,
                                    "epv": snap.epv,
                                    "ekf_pos_horiz_var": snap.pos_horiz_var,
                                    "ekf_pos_vert_var": snap.pos_vert_var,
                                    "gps_timestamp_iso": snap.timestamp_iso,
                                },
                                "detections": det_summary if det_summary else {
                                    "count_all": detection_count
                                },
                                "meta": {
                                    "cooldown_s": float(user_data.upload_cooldown),
                                    "source_format": format_str,
                                    "frame_w": int(width) if width else None,
                                    "frame_h": int(height) if height else None,
                                },
                            }

                            # Simpan JSON lokal
                            with open(local_json_path, "w", encoding="utf-8") as f:
                                json.dump(payload, f, ensure_ascii=False, indent=2)
                            logger.info("Saved JSON metadata: %s", local_json_path)

                            # Rute remote
                            remote_img = f"detected_fire/{jpgname}"
                            remote_json = f"detected_fire/{jsonname}"

                            if executor:
                                # upload paralel
                                fut_img: Future = executor.submit(
                                    upload_file_to_firebase, local_img_path, remote_img
                                )
                                fut_json: Future = executor.submit(
                                    upload_file_to_firebase, local_json_path, remote_json
                                )

                                def _done(tag: str):
                                    def cb(f: Future) -> None:
                                        try:
                                            ok = f.result()
                                            logger.info("Upload finished %s -> %s", tag, "OK" if ok else "FAILED")
                                        except Exception as exc:
                                            logger.exception("Upload raised (%s): %s", tag, exc)
                                    return cb

                                fut_img.add_done_callback(_done("image"))
                                fut_json.add_done_callback(_done("json"))
                            else:
                                ok1 = upload_file_to_firebase(local_img_path, remote_img)
                                ok2 = upload_file_to_firebase(local_json_path, remote_json)
                                logger.info("Blocking upload results -> image:%s json:%s", ok1, ok2)

                        except Exception as e:
                            logger.exception("Saving/uploading failed: %s", e)

                # kita cukup upload sekali per frame yang memicu
                break

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
                user_data.set_frame(bgr_preview)  # GStreamer app expects BGR
            except Exception as e:
                logger.debug("Overlay preview failed: %s", e)

    if not fire_triggered:
        logger.debug("No valid fire detection in this buffer")

    return Gst.PadProbeReturn.OK


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Detection + upload to Firebase + GPS overlay + JSON metadata")
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
    # GPS / MAVLink options
    ap.add_argument(
        "--mavlink-port",
        default=os.environ.get("MAV_PORT", DEFAULT_MAVLINK_PORT),
        help="Port MAVLink (contoh: /dev/ttyACM0)",
    )
    ap.add_argument(
        "--mavlink-baud",
        type=int,
        default=int(os.environ.get("MAV_BAUD", DEFAULT_MAVLINK_BAUD)),
        help="Baudrate MAVLink (contoh: 57600)",
    )
    ap.add_argument(
        "--no-gps",
        action="store_true",
        help="Nonaktifkan pembacaan GPS/MAVLink",
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

    # GPS thread (opsional)
    stop_event = threading.Event()
    gps_state = GpsState()
    gps_enabled = (not args.no_gps) and PYMAVLINK_AVAILABLE
    gps_thread = None
    if gps_enabled:
        gps_thread = threading.Thread(
            target=gps_thread_fn,
            args=(args.mavlink_port, args.mavlink_baud, gps_state, stop_event),
            daemon=True,
        )
        gps_thread.start()
        logger.info("GPS thread started: %s @ %s", args.mavlink_port, args.mavlink_baud)
    else:
        if args.no_gps:
            logger.info("GPS disabled via --no-gps")
        else:
            logger.info("GPS disabled (pymavlink not available)")

    # Graceful shutdown on SIGINT/SIGTERM
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
            gps_state=gps_state if gps_enabled else None,
            gps_enabled=gps_enabled,
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
        stop_event.set()
        if gps_thread and gps_thread.is_alive():
            gps_thread.join(timeout=2.0)
        logger.info("Exit")


if __name__ == "__main__":
    main()
