# all code
# Verry Kurniawan 21120122130062
# Finodya Yahdun 21120122130065
# Imam Baihaqqy 21120122130078

#!/usr/bin/env python3
"""
run_streaming.py
Menjalankan: python streaming.py di folder ~/Desktop/drone/all-program

Jika streaming.py butuh environment yang disetup oleh setup_env.sh juga, kamu bisa jalankan setup dulu.
"""
import argparse
import subprocess
import os
import sys

def main(stream_dir, use_setup=False, setup_dir=None):
    stream_dir = os.path.expanduser(stream_dir)
    if not os.path.isdir(stream_dir):
        print(f"Folder streaming tidak ditemukan: {stream_dir}", file=sys.stderr)
        sys.exit(1)

    if use_setup:
        # jika setup_env.sh ada di direktori lain (misalnya hailo-rpi5-examples)
        if setup_dir is None:
            print("use_setup True tapi setup_dir tidak diberikan", file=sys.stderr)
            sys.exit(1)
        setup_dir = os.path.expanduser(setup_dir)
        if not os.path.isdir(setup_dir):
            print(f"Folder setup tidak ditemukan: {setup_dir}", file=sys.stderr)
            sys.exit(1)
        cmd = f"cd {stream_dir} && source {os.path.join(setup_dir,'setup_env.sh')} && python streaming.py"
        runner = ['bash', '-lc', cmd]
    else:
        runner = ['python', 'streaming.py']

    print(f"Menjalankan streaming di: {stream_dir} (use_setup={use_setup})")
    try:
        subprocess.run(runner, cwd=stream_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Streaming process exited with code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Launcher untuk streaming.py")
    ap.add_argument('--stream-dir', default='~/Desktop/drone/all-program',
                    help='Folder yang berisi streaming.py (default: ~/Desktop/drone/all-program)')
    ap.add_argument('--use-setup', action='store_true',
                    help='Jika ingin source setup_env.sh sebelum menjalankan (setup berada di --setup-dir)')
    ap.add_argument('--setup-dir', default='~/Desktop/drone/hailo-rpi5-examples',
                    help='Lokasi setup_env.sh jika --use-setup diaktifkan (default: ~/Desktop/drone/hailo-rpi5-examples)')
    args = ap.parse_args()
    main(args.stream_dir, use_setup=args.use_setup, setup_dir=args.setup_dir)
