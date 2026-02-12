# all code
# Verry Kurniawan 21120122130062
# Finodya Yahdun 21120122130065
# Imam Baihaqqy 21120122130078

#!/usr/bin/env python3
"""
run_detect.py
Menjalankan: source setup_env.sh && python basic_pipelines/just-detect.py --input rpicam --labels labels.json --hef-path model-hailo-22data.hef

Jalankan dari folder project (~/Desktop/drone/hailo-rpi5-examples) atau berikan path ke folder project.
"""
import argparse
import subprocess
import os
import sys

def main(project_dir):
    # Pastikan direktori ada
    project_dir = os.path.expanduser(project_dir)
    if not os.path.isdir(project_dir):
        print(f"Folder project tidak ditemukan: {project_dir}", file=sys.stderr)
        sys.exit(1)

    # Perintah yang akan dijalankan di shell (satu shell session sehingga source bekerja)
    cmd = (
        'source setup_env.sh && '
        'python basic_pipelines/just-detect-time*.py '
        '--input kebakaran_hutan.mp4 '
        '--labels labels.json '
        '--hef-path model80map.hef'
    )

    print(f"Menjalankan deteksi di: {project_dir}")
    try:
        # jalankan dalam bash login shell supaya source berjalan
        subprocess.run(['bash', '-lc', cmd], cwd=project_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Process exited with code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Launcher untuk just-detect.py (Hailo)")
    ap.add_argument('--project-dir',
                    default='~/Desktop/drone/hailo-rpi5-examples',
                    help='Path ke folder hailo-rpi5-examples (default: ~/Desktop/drone/hailo-rpi5-examples)')
    args = ap.parse_args()
    main(args.project_dir)
