import cv2
import torch
from ultralytics import YOLO

# 1. Configuration
# Your host machine's IP address
HOST_IP = "192.168.56.100" 
PORT = 5000

# 2. Load Model to GPU
# .to('cuda') ensures the model weights are on the Orin's GPU
model = YOLO('yolov8n-pose.pt').to('cuda')

# 3. CSI Input Pipeline (IMX219 / RPi Cam v2)
# Optimized for 720p @ 60fps to match your hardware's Mode 4
input_pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=60/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=640, height=480, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink drop=True"
)

# 4. Local Network Output Pipeline
# Bitrate is set to 4000kbps for high quality over local network
output_pipeline = (
    f"appsrc ! videoconvert ! video/x-raw, format=I420 ! "
    f"x264enc tune=zerolatency bitrate=4000 speed-preset=ultrafast ! "
    f"rtph264pay ! udpsink host={HOST_IP} port={PORT}"
)

# Initialize Capture and Writer
cap = cv2.VideoCapture(input_pipeline, cv2.CAP_GSTREAMER)
writer = cv2.VideoWriter(output_pipeline, cv2.CAP_GSTREAMER, 0, 30, (640, 480))

if not cap.isOpened():
    print("CRITICAL: Camera failed to open. Try:")
    print("sudo systemctl restart nvargus-daemon")
    exit()

# Set for tracking unique individuals
counted_ids = set()

print(f"--- Local AI Stream Started ---")
print(f"Streaming to: {HOST_IP}:{PORT}")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

try:
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Inference: device=0 forces the use of the Orin Nano GPU
        results = model.track(frame, persist=True, classes=[0], verbose=False, device=0)

        # Update Counter logic
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            for obj_id in ids:
                counted_ids.add(obj_id)

        # Plot "Wire" Skeletons
        annotated_frame = results[0].plot()

        # Add Visual Counter
        cv2.putText(annotated_frame, f"People Counted: {len(counted_ids)}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Push to Network
        writer.write(annotated_frame)

except KeyboardInterrupt:
    print("\nShutting down...")

finally:
    cap.release()
    writer.release()
