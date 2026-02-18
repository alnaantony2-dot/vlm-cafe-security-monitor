# vlm-cafe-security-monitor
## Real-Time VLM Cafe Security Monitor (CPU-Only)

A real-time video monitoring system using a Vision-Language Model (VLM) to analyze
crowd behavior and detect security risks such as fire, weapons, fights, and panic
events — running entirely on CPU using local inference via Ollama.

This project is designed for low-resource environments such as cafes, shops,
and small public spaces.

---

## Key Features

- Real-time webcam or video file monitoring
- Vision-Language Model inference using `qwen3-vl:2b`
- Local CPU-only execution (no GPU required)
- Threaded pipeline for smooth video capture and inference
- Crowd analysis:
  - People count
  - Crowd density (low / medium / high)
  - Fire detection
  - Weapon visibility
  - Fight detection
  - Panic or running behavior
- Risk score aggregation per frame
- JSON report generation

---

## System Architecture
Video Source
│
▼
OpenCV Capture Thread
│
▼
Frame Queue (bounded)
│
▼
VLM Inference Thread (Ollama)
│
▼
Risk Analysis + JSON Output
