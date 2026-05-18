import cv2
import numpy as np
import torch
from ultralytics import YOLO

# 1. Configuration
HOST_IP = "192.168.56.100" 
PORT = 5000

# 2. Load Model to GPU
model = YOLO('yolov8n-pose.pt').to('cuda')

# 3. HIGH-STABILITY INPUT PIPELINE
input_pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=60/1 ! "
    "nvvidconv ! video/x-raw, width=640, height=480, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "appsink drop=True max-buffers=1"
)

# 4. OUTPUT PIPELINE (This sends the 16x16 VISUAL to your host)
output_pipeline = (
    f"appsrc ! videoconvert ! video/x-raw, format=I420 ! "
    f"x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast ! "
    f"rtph264pay ! udpsink host={HOST_IP} port={PORT}"
)

cap = cv2.VideoCapture(input_pipeline, cv2.CAP_GSTREAMER)
writer = cv2.VideoWriter(output_pipeline, cv2.CAP_GSTREAMER, 0, 30, (480, 480))

if not cap.isOpened():
    print("Camera failed. Run: sudo systemctl restart nvargus-daemon")
    exit()

print(f"--- Streaming 16x16 Visual to {HOST_IP} ---")

try:
    while True:
        success, frame = cap.read()
        if not success: continue

        # --- THE 16x16 VISUAL LOGIC ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Shrink to 16x16
        tiny_dots = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA)
        
        # Upscale to 480x480 for a clear "blocky" view
        visual_grid = cv2.resize(tiny_dots, (480, 480), interpolation=cv2.INTER_NEAREST)
        
        # Convert back to BGR so the video writer can handle it
        visual_bgr = cv2.cvtColor(visual_grid, cv2.COLOR_GRAY2BGR)

        # (Optional) Still run AI in background if you want to see counts in console
        # results = model.track(frame, persist=True, classes=[0], verbose=False, device=0)

        # Stream the blocky view to your host
        writer.write(visual_bgr)

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    cap.release()
    writer.release()
