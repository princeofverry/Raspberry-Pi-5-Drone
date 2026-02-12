# Hailo RPi5 Drone Vision Project

This project implements AI-based object detection and vision processing
on a Raspberry Pi 5 using the Hailo AI Accelerator (Hailo-8 / Hailo-8L).
The system is designed for edge inference deployment and supports
real-time detection, benchmarking, image capture, and structured data
logging.

The core implementation is located inside the `basic_pipelines/`
directory.

---

## Repository Setup

### Clone Repository

```bash
git clone https://github.com/hailo-ai/hailo-rpi5-examples.git
cd hailo-rpi5-examples
```

---

## Installation

Run the installation script:

```bash
./install.sh
```

Activate the environment:

```bash
source setup_env.sh
```

```
pip install -r requirements.txt
```

The virtual environment must be activated every time a new terminal
session is opened.

---

## Model Setup

Place the compiled Hailo model file (`.hef`) in the project root
directory:

```bash
mv model80map.hef hailo-rpi5-examples/
```

Project structure example:

    hailo-rpi5-examples/
    │
    ├── basic_pipelines/
    ├── model80map.hef
    ├── README.md
    └── ...

---

## Main Program Directory

All primary detection and vision scripts are located in:

```bash
cd basic_pipelines
```

To verify available programs:

```bash
ls
```

Example contents:

- detection.py
- detection_simple.py
- just-detect.py
- just-detect-new.py
- just-detect-pengujian.py
- just-detect-time.py
- taking-csv.py
- taking-pict.py
- depth.py
- instance_segmentation.py
- pose_estimation.py

---

## Running Programs

### Standard Object Detection

```bash
python just-detect.py
```

### Detection with Time Benchmarking

```bash
python just-detect-time.py
```

To use a different input video, modify the --input parameter in just-detect-time.py or any other shortcut script.

```
--input kebakaran_hutan.mp4 change --input <video_name>
```

### Detection for Testing / Evaluation

```bash
python just-detect-pengujian.py
```

### Export Detection Results to CSV

```bash
python taking-csv.py
```

### Capture Images

```bash
python taking-pict.py
```

---

## System Requirements

- Raspberry Pi 5
- Hailo AI Accelerator (Hailo-8 or Hailo-8L compatible with RPi5)
- Hailo RPi5 Software Environment (`install.sh`)
- Activated Python virtual environment
- Python 3.9 or higher
- Compiled `.hef` model file

---

## Output Files

Depending on the executed program, the system may generate:

- CSV detection logs
- Benchmark logs
- Runtime logs (`hailort.log`)
- Captured images

---

## Description

This project demonstrates real-time edge AI deployment using Raspberry
Pi 5 and Hailo hardware acceleration. It includes structured
benchmarking and logging capabilities suitable for research, capstone
projects, and embedded AI system development.
