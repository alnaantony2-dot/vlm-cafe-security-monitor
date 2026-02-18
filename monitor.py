import cv2
import base64
import requests
import json
import re
import time
from collections import Counter

import threading
import queue


# ============== CONFIG =================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3-vl:2b"

FRAME_INTERVAL = 1
RESIZE_WIDTH = 640
OUTPUT_FILE = "cafe_security_output.json"
FRAME_QUEUE_SIZE = 2  # never more than this
frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
results = []
stop_event = threading.Event()

# ======================================

def capture_thread(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source")

    last_sent = time.time()
    t = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * RESIZE_WIDTH / w)))

        # Send frame for analysis every N seconds
        if time.time() - last_sent >= FRAME_INTERVAL:
            if not frame_queue.full():
                frame_queue.put((frame.copy(), t))
                t += FRAME_INTERVAL
                last_sent = time.time()

        cv2.imshow("Cafe Security Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()

def inference_thread():
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            frame, t = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            data = analyze_frame(frame, t)
            results.append(data)

            print(
                f"[{t}s] People:{data['people_count']} "
                f"Fire:{data['fire_detected']} "
                f"Weapon:{data['weapons_visible']} "
                f"Fight:{data['fight_detected']} "
                f"Risk:{data['risk_score']}"
            )

        except Exception as e:
            print("Inference error:", e)





def frame_to_base64(frame):
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf).decode()

def clean_json(text):
    if not text or text.strip() == "":
        return None
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def risk_score(f):
    score = 0.0
    if f["fire_detected"]:
        score += 0.9
    if f["weapons_visible"]:
        score += 0.8
    if f["fight_detected"]:
        score += 0.6
    if f["panic_or_running"]:
        score += 0.4
    return round(min(score, 1.0), 2)


def analyze_frame(frame, t):
    prompt = f"""
Return ONLY valid JSON.

{{
  "timestamp_sec": {t},
  "people_count": number,
  "crowd_density": "low" | "medium" | "high",
  "fire_detected": true | false,
  "weapons_visible": true | false,
  "fight_detected": true | false,
  "panic_or_running": true | false,
  "confidence": number between 0 and 1,
  "notes": "short observation"
}}
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [frame_to_base64(frame)],
        "stream": False
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()

    result = clean_json(r.json()["response"])
    result["risk_score"] = risk_score(result)
    return result


def monitoriii(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source")

    results = []
    last = time.time()
    t = 0

    print("Monitoring started | Press Q to stop")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * RESIZE_WIDTH / w)))

        if time.time() - last >= FRAME_INTERVAL:
            data = analyze_frame(frame, t)
            results.append(data)

            print(
                f"[{t}s] People:{data['people_count']} "
                f"Fire:{data['fire_detected']} "
                f"Weapon:{data['weapons_visible']} "
                f"Fight:{data['fight_detected']} "
                f"Risk:{data['risk_score']}"
            )

            t += FRAME_INTERVAL
            last = time.time()

        cv2.imshow("Cafe Security Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    summary = {
        "average_people": round(
            sum(r["people_count"] for r in results) / len(results), 2
        ),
        "max_people": max(r["people_count"] for r in results),
        "fire_events": sum(r["fire_detected"] for r in results),
        "weapon_events": sum(r["weapons_visible"] for r in results),
        "fight_events": sum(r["fight_detected"] for r in results),
        "dominant_density": Counter(
            r["crowd_density"] for r in results
        ).most_common(1)[0][0],
        "frames": results
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print("Output saved:", OUTPUT_FILE)

def monitor(source):
    t1 = threading.Thread(target=capture_thread, args=(source,))
    t2 = threading.Thread(target=inference_thread)

    t1.start()
    t2.start()

    t1.join()
    stop_event.set()
    t2.join()

    if not results:
        print("No frames analyzed.")
        return

    summary = {
        "average_people": round(
            sum(r["people_count"] for r in results) / len(results), 2
        ),
        "max_people": max(r["people_count"] for r in results),
        "fire_events": sum(r["fire_detected"] for r in results),
        "weapon_events": sum(r["weapons_visible"] for r in results),
        "fight_events": sum(r["fight_detected"] for r in results),
        "dominant_density": Counter(
            r["crowd_density"] for r in results
        ).most_common(1)[0][0],
        "frames": results
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print("Output saved:", OUTPUT_FILE)



if __name__ == "__main__":
    print("1 → Webcam")
    print("2 → Video file")
    choice = input("Select input: ")

    if choice == "1":
        monitor(0)
    else:
        path = input("Enter video path: ")
        monitor(path)
