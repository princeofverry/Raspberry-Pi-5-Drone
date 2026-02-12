# 🚁 Hailo RPi5 Drone Vision Project

This project runs AI-based object detection and sensor monitoring on a
**Raspberry Pi 5** using the **Hailo AI Accelerator**.\
It includes vision processing, object detection, sensor logging (DHT22,
GPS), CPU temperature monitoring, and streaming.

------------------------------------------------------------------------

## 📦 Clone Repository

``` bash
git clone https://github.com/hailo-ai/hailo-rpi5-examples.git
cd hailo-rpi5-examples
```

------------------------------------------------------------------------

## ⚙️ Installation

Run the installation script:

``` bash
./install.sh
```

Activate the environment:

``` bash
source setup_env.sh
```

⚠️ You must activate the environment every time you open a new terminal.

------------------------------------------------------------------------

## 🤖 Model Setup

Move your `.hef` model file into the root project directory:

``` bash
mv model80map.hef hailo-rpi5-examples/
```

Project structure should look like:

    hailo-rpi5-examples/
    │
    ├── all-program/
    ├── model80map.hef
    ├── README.md
    └── ...

------------------------------------------------------------------------

## 📂 Available Programs

Navigate to the program directory:

``` bash
cd all-program
```

List available programs:

``` bash
ls
```

Example contents:

-   dht22.py\
-   gps.py\
-   record.py\
-   run-all-sensor.py\
-   run-detect.py\
-   run-streaming.py\
-   run-vision.py\
-   streaming.py\
-   temp.py

------------------------------------------------------------------------

# 🚀 Running Programs

## 🔎 Object Detection

``` bash
python run-detect.py
```

## 🌡 DHT22 Sensor

``` bash
python dht22.py
```

## 📡 GPS Monitoring

``` bash
python gps.py
```

## 🧠 Vision Processing

``` bash
python run-vision.py
```

## 📊 Run All Sensors

``` bash
python run-all-sensor.py
```

## 🖥 Streaming

``` bash
python run-streaming.py
```

------------------------------------------------------------------------

# 🛠 Requirements

-   Raspberry Pi 5\
-   Hailo AI Accelerator (Hailo-8 / Hailo-8L compatible with RPi5)\
-   Hailo RPi5 Software Environment (`install.sh`)\
-   Activated virtual environment (`source setup_env.sh`)\
-   Python 3.9+\
-   `.hef` model file (e.g., `model80map.hef`)

------------------------------------------------------------------------

# 📊 Output Files

Some programs generate CSV logs:

-   gps_log.csv\
-   suhu_cpu_raspberrypi.csv\
-   suhu_cpu_raspberrypi_idle.csv

These files can be used for further data analysis.

------------------------------------------------------------------------

# 📌 Notes

-   Always activate the environment before running any program:

    ``` bash
    source setup_env.sh
    ```

-   Ensure the `.hef` model file is placed in the project root
    directory.

-   Use `ls` to verify files before execution.

------------------------------------------------------------------------

👨‍💻 Developed for Raspberry Pi 5 + Hailo AI Edge Inference System
