import cv2
import numpy as np
import torch
from ultralytics import YOLO

# 1. Configuration
HOST_IP = "192.168.56.100" 
PORT = 5000
GRID_SIZE = 16

# 2. Load Model to GPU
model = YOLO('yolov8n-pose.pt').to('cuda')

# 3. CSI Input Pipeline (IMX219)
# We pull the full 720p for the AI, then we will downsample in Python
input_pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=60/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=640, height=480, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink drop=True"
)

# 4. Local Network Output Pipeline (Standard View)
output_pipeline = (
    f"appsrc ! videoconvert ! video/x-raw, format=I420 ! "
    f"x264enc tune=zerolatency bitrate=4000 speed-preset=ultrafast ! "
    f"rtph264pay ! udpsink host={HOST_IP} port={PORT}"
)

cap = cv2.VideoCapture(input_pipeline, cv2.CAP_GSTREAMER)
writer = cv2.VideoWriter(output_pipeline, cv2.CAP_GSTREAMER, 0, 30, (640, 480))

if not cap.isOpened():
    print("CRITICAL: Camera failed. Run: sudo systemctl restart nvargus-daemon")
    exit()

counted_ids = set()

print(f"--- 16x16 Mapping & AI Stream Started ---")

try:
    while True:
        success, frame = cap.read()
        if not success:
            break

        # --- PART A: AI Tracking (GPU) ---
        results = model.track(frame, persist=True, classes=[0], verbose=False, device=0)
        
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            for obj_id in ids:
                counted_ids.add(obj_id)

        # Draw the visual "Wire" overlay for the stream
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, f"Count: {len(counted_ids)}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        writer.write(annotated_frame)

        # --- PART B: 16x16 Grayscale Mapping ---
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Downsample to 16x16 "Dots"
        dots_16x16 = cv2.resize(gray, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA)
        
        # 3. Flatten for reference (Mapping dots 0-255)
        # Access any dot via: dots_16x16[y, x]
        flattened_dots = dots_16x16.flatten()

        # Optional: Print the 16x16 grid to the terminal every few frames
        # This provides a real-time "text" view of what the Orin sees
        print("\033[H", end="") # Move cursor to top of terminal
        print(f"--- 16x16 Grid Map (People Seen: {len(counted_ids)}) ---")
        for row in dots_16x16:
            print(" ".join(f"{p:3}" for p in row))

except KeyboardInterrupt:
    print("\nShutting down...")

finally:
    cap.release()
    writer.release()
