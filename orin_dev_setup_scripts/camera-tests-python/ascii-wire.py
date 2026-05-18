import cv2
import os
import numpy as np
import torch
from ultralytics import YOLO

# --- 1. SETTINGS ---
HOST_IP = "192.168.56.100" 
PORT = 5000
CHARS = " .:-=+*#%@" # ASCII palette from dark to bright

# --- 2. INITIALIZE ---
# Load YOLO to GPU
model = YOLO('yolov8n-pose.pt').to('cuda')

# Input: 720p @ 60fps
input_pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=60/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=640, height=480, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink drop=True max-buffers=1"
)

# Output: Full AI Stream to Host
output_pipeline = (
    f"appsrc ! videoconvert ! video/x-raw, format=I420 ! "
    f"x264enc tune=zerolatency bitrate=3000 speed-preset=ultrafast ! "
    f"rtph264pay ! udpsink host={HOST_IP} port={PORT}"
)

cap = cv2.VideoCapture(input_pipeline, cv2.CAP_GSTREAMER)
writer = cv2.VideoWriter(output_pipeline, cv2.CAP_GSTREAMER, 0, 30, (640, 480))

if not cap.isOpened():
    print("Camera Error. Try: sudo systemctl restart nvargus-daemon")
    exit()

print("--- AI STREAM + ASCII VISION ACTIVE ---")

try:
    while True:
        success, frame = cap.read()
        if not success: continue

        # --- PART A: AI STREAM (To Host) ---
        results = model.track(frame, persist=True, classes=[0], verbose=False, device=0)
        annotated_frame = results[0].plot()
        writer.write(annotated_frame)

        # --- PART B: 16x16 ASCII (In Terminal) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tiny = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        
        # Build the ASCII string
        ascii_output = "\033[H" # ANSI escape code to move cursor to top
        for row in tiny:
            line = ""
            for pixel in row:
                char_idx = int(pixel * (len(CHARS) - 1) / 255)
                line += CHARS[char_idx] + " " # Double-width for better aspect ratio
            ascii_output += line + "\n"
        
        # One-shot print to prevent flickering
        print(ascii_output, end="")

except KeyboardInterrupt:
    print("\nPowering down...")
finally:
    cap.release()
    writer.release()
