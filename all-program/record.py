#!/usr/bin/env python3
import cv2
from datetime import datetime
import os

cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)

# Paksa MJPG agar warna benar
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Gagal membuka /dev/video0")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

recording = False
out = None

print("=== Kontrol Rekaman ===")
print("Tekan 'r' untuk mulai merekam")
print("Tekan 's' untuk stop rekam")
print("Tekan 'q' untuk keluar")
print("========================")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera!")
        break

    # OPTIONAL: Kalau tetap salah warna, aktifkan ini:
    # frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)

    if recording and out is not None:
        out.write(frame)

    cv2.imshow("Live Camera (/dev/video0)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r') and not recording:
        os.makedirs("recordings", exist_ok=True)
        filename = datetime.now().strftime("recordings/rec_%Y%m%d_%H%M%S.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
        print(f"[RECORDING STARTED] -> {filename}")
        recording = True

    if key == ord('s') and recording:
        recording = False
        if out: out.release()
        out = None
        print("[RECORDING STOPPED]")

    if key == ord('q'):
        print("Keluar...")
        break

if recording and out:
    out.release()

cap.release()
cv2.destroyAllWindows()
