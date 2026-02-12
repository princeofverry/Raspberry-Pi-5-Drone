#!/usr/bin/env python3
"""
gps_log.py
Baca GPS (GPS_RAW_INT / GLOBAL_POSITION_INT), EKF status, cetak ke console dan simpan ke gps_log.csv
Ganti PORT/BAUD sesuai setup mu.
"""
import csv
import time
from datetime import datetime
from pymavlink import mavutil

PORT = "/dev/ttyACM0"
BAUD = 57600
CSV_FILE = "gps_log.csv"

def fmt_latlon(v): return (v/1e7) if v is not None else None
def fmt_alt_mm(v): return (v/1000.0) if v is not None else None

print(f"Connect: {PORT} @ {BAUD}")
master = mavutil.mavlink_connection(PORT, baud=BAUD, autoreconnect=True)
print("Waiting heartbeat...")
master.wait_heartbeat(timeout=10)
print(f"Heartbeat from sys={master.target_system} comp={master.target_component}")

# request streams (optional)
try:
    master.mav.request_data_stream_send(master.target_system, master.target_component,
                                       mavutil.mavlink.MAV_DATA_STREAM_ALL,
                                       2, 1)
except Exception:
    pass

# prepare csv (append mode, add header if new)
header = ["timestamp_iso","lat","lon","alt_m","sats","fix_type","eph","epv","pos_horiz_variance","pos_vert_variance"]
first = False
try:
    with open(CSV_FILE, "r"):
        pass
except FileNotFoundError:
    first = True

csvf = open(CSV_FILE, "a", newline="")
writer = csv.writer(csvf)
if first:
    writer.writerow(header)
    csvf.flush()

# store last EKF variances when EKF_STATUS_REPORT arrives
last_pos_horiz_var = None
last_pos_vert_var = None

print("="*72)
print("Logging GPS ->", CSV_FILE)
print(f"{'time':19} | {'lat':10} | {'lon':11} | {'alt(m)':7} | sats | fix | eph | epv | pos_h_var")
print("-"*72)

try:
    while True:
        msg = master.recv_match(blocking=True, timeout=2)
        if msg is None:
            continue

        t = msg.get_type()

        if t == "EKF_STATUS_REPORT":
            # update last known variances (values are floats)
            last_pos_horiz_var = getattr(msg, "pos_horiz_variance", None)
            last_pos_vert_var = getattr(msg, "pos_vert_variance", None)

        if t in ("GPS_RAW_INT", "GPS2_RAW"):
            now = datetime.utcnow().isoformat()
            lat = fmt_latlon(getattr(msg, "lat", None))
            lon = fmt_latlon(getattr(msg, "lon", None))
            alt_m = fmt_alt_mm(getattr(msg, "alt", None))
            sats = getattr(msg, "satellites_visible", None)
            fix = getattr(msg, "fix_type", None)
            eph = getattr(msg, "eph", None)
            epv = getattr(msg, "epv", None)

            print(f"{now[:19]} | {lat if lat else 'N/A':10} | {lon if lon else 'N/A':11} | {alt_m if alt_m else 'N/A':7} | {sats:4} | {fix:4} | {eph} | {epv} | {last_pos_horiz_var}")
            writer.writerow([now, lat, lon, alt_m, sats, fix, eph, epv, last_pos_horiz_var, last_pos_vert_var])
            csvf.flush()

        elif t == "GLOBAL_POSITION_INT":
            # Sometimes autopilot publishes processed global position too
            now = datetime.utcnow().isoformat()
            lat = fmt_latlon(getattr(msg, "lat", None))
            lon = fmt_latlon(getattr(msg, "lon", None))
            alt_m = fmt_alt_mm(getattr(msg, "alt", None))
            # we might not have sats here
            print(f"{now[:19]} | [GLOBAL] {lat if lat else 'N/A':10} | {lon if lon else 'N/A':11} | {alt_m if alt_m else 'N/A':7} | -    | -   | - | - | {last_pos_horiz_var}")
            writer.writerow([now, lat, lon, alt_m, None, None, None, None, last_pos_horiz_var, last_pos_vert_var])
            csvf.flush()

        # else: ignore other messages (or you can print a sample for debug)

except KeyboardInterrupt:
    print("\nStopped by user")

finally:
    csvf.close()
    try:
        master.mav.request_data_stream_send(master.target_system, master.target_component,
                                           mavutil.mavlink.MAV_DATA_STREAM_ALL, 0, 0)
    except Exception:
        pass
    master.close()
    print("Connection closed.")
