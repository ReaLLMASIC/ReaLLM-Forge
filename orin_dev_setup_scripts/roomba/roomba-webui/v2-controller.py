import cv2
import numpy as np
import threading
import time
import serial
import os
import random
import atexit
import sys
from datetime import datetime
from flask import Flask, Response, render_template_string, request, jsonify

# --- App Initialization & Globals ---
app = Flask(__name__)
current_frame = None  
lock = threading.Lock()

# --- 🛠️ AUTOMATED OPENCV VARIANT ENVIRONMENT GUARD ---
def check_opencv_gstreamer_support():
    build_info = cv2.getBuildInformation()
    if "GStreamer:" in build_info:
        lines = build_info.split('\n')
        for line in lines:
            if "GStreamer:" in line and "YES" in line:
                print("✅ ENVIRONMENT CHECK: OpenCV variant with native GStreamer support verified.")
                return True
                
    print("\n" + "!"*60)
    print("🚨 CRITICAL ENVIRONMENT ERROR: Invalid OpenCV Variant Detected!")
    print("Your current OpenCV installation lacks GStreamer support.")
    print("The CSI camera acceleration pipeline will fail to open.")
    print("-"*60)
    print("FIX ACTIONS FOR YOUR JETSON ORIN WORKSTATION:")
    print("1. If inside a virtual env, allow system packages:")
    print("   Use: python3 -m venv --system-site-packages <env_name>")
    print("2. Remove the conflicting pip binary distribution package:")
    print("   Run: pip uninstall opencv-python opencv-python-headless")
    print("3. Symlink or verify the NVIDIA pre-compiled system module:")
    print("   The native JetPack OpenCV module includes full GStreamer compilation.")
    print("!"*60 + "\n")
    return False

# Run the compilation flag check immediately on script execution
if not check_opencv_gstreamer_support():
    sys.exit("Fatal: Invalid OpenCV binary variant. Please fix dependencies.")

# --- DATASET GRID CONFIGURATION ---
SAMPLE_SIZE = 32

# Navigation, Logging, & Stream Globals
current_speed = 150  # Stable line tracking target velocity (mm/s)
vacuum_on = False
last_keys_set = set()
auto_mode = False
show_grayscale = False  # Controls if the HUD stream renders raw video or downsampled AI pixels
current_action_label = ""  
active_key_string = "idling"  

# 🎞️ GLOBAL FRAME COUNTER INITIALIZATION
frame_count = 0

# RECORD & PLAYBACK STATE METRICS
operational_mode = "manual"
movement_history = []  
current_move_start = None
last_saved_keys = set()
playback_index = 0  

# Accumulated Odometry & Tracking Angle Registers
total_distance = 0.0  
current_heading = 0.0  
last_fetched_angle = 0  

# Raw Discrete Wheel Telemetry Registers
live_left_speed_mms = 0.0
live_right_speed_mms = 0.0

# Live Performance Metrics
live_speed_mms = 0.0
battery_pct = 100.0

# Global Paths for Unique Local Dataset Logging Files
output_dir = ""
csv_path = ""
csv_file = None

# --- Roomba Serial Setup ---
try:
    roomba = serial.Serial('/dev/ttyUSB0', baudrate=115200, timeout=0.1)
    roomba.write(bytes([128, 131])) 
    print("Roomba serial connected successfully.")
except Exception as e:
    roomba = None
    print(f"Roomba not connected: {e}. Running in simulation mode.")

# --- Hardware Control Functions ---
def drive_roomba(velocity, radius):
    """Sends raw drive commands to the Roomba via Open Interface."""
    if roomba:
        v_high, v_low = (velocity >> 8) & 0xFF, velocity & 0xFF
        r_high, r_low = (radius >> 8) & 0xFF, radius & 0xFF
        roomba.write(bytes([137, v_high, v_low, r_high, r_low]))

def set_vacuum(state):
    if roomba:
        val = 7 if state else 0 
        roomba.write(bytes([138, val]))

def dock_roomba():
    if roomba:
        set_vacuum(False)
        roomba.write(bytes([143]))
        print("🔌 Roomba sent to Seek Dock (Command 143).")

def cancel_docking_sequence():
    if roomba:
        roomba.write(bytes([128, 131])) 
        time.sleep(0.02)
        drive_roomba(0, 0)
        print("📥 Control Restored: Passive Docking aborted.")

def update_sensors_efficiently():
    global total_distance, current_heading, live_speed_mms, battery_pct, last_fetched_angle
    global live_left_speed_mms, live_right_speed_mms
    bump_left, bump_right, wheel_drop = 0, 0, 0
    if not roomba: return 0, 0, 0
    try:
        roomba.flushInput()
        roomba.write(bytes([142, 0])) 
        time.sleep(0.03) 
        if roomba.inWaiting() >= 26:
            data = roomba.read(26)
            byte0 = data[0]
            bump_right = 1 if (byte0 & 0x01) else 0
            bump_left = 1 if (byte0 & 0x02) else 0
            wheel_drop = 1 if (byte0 & 0x04 or byte0 & 0x08) else 0
            
            delta_dist = int.from_bytes(data[12:14], byteorder='big', signed=True)
            total_distance += delta_dist
            
            last_fetched_angle = int.from_bytes(data[14:16], byteorder='big', signed=True)
            current_heading = (current_heading + last_fetched_angle) % 360
            
            charge = int.from_bytes(data[22:24], byteorder='big', signed=False)
            capacity = int.from_bytes(data[24:26], byteorder='big', signed=False)
            if capacity > 0:
                battery_pct = max(0.0, min(100.0, (charge / capacity) * 100.0))
            live_speed_mms = delta_dist / 0.055

        roomba.write(bytes([149, 2, 43, 44]))
        time.sleep(0.015)
        if roomba.inWaiting() >= 4:
            wheel_data = roomba.read(4)
            live_left_speed_mms = int.from_bytes(wheel_data[0:2], byteorder='big', signed=True)
            live_right_speed_mms = int.from_bytes(wheel_data[2:4], byteorder='big', signed=True)

    except Exception as e:
        print(f"Serial read telemetry error: {e}")
    return bump_left, bump_right, wheel_drop

def draw_guidelines(img):
    h, w = img.shape[:2]
    left_start, left_end = (int(w * 0.10), h), (int(w * 0.38), int(h * 0.52))
    right_start, right_end = (int(w * 0.90), h), (int(w * 0.62), int(h * 0.52))
    cv2.line(img, left_start, left_end, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(img, right_start, right_end, (0, 255, 0), 2, cv2.LINE_AA)
    return img

# --- Playback Thread Engine ---
def execution_playback_worker():
    global operational_mode, current_action_label, playback_index, active_key_string
    while playback_index < len(movement_history):
        if operational_mode != "playback":
            return
            
        saved_keys, duration, saved_speed = movement_history[playback_index]
        active_key_string = saved_keys[0] if saved_keys else "idling"
        
        if 'idling' in saved_keys or not saved_keys:
            drive_roomba(0, 0); current_action_label = ""
        elif 'space' in saved_keys:
            drive_roomba(0, 0); current_action_label = "stop"
        elif 'w' in saved_keys:
            drive_roomba(saved_speed, 32767); current_action_label = "forward"
        elif 's' in saved_keys:
            drive_roomba(-saved_speed, 32767); current_action_label = "backward"
        elif 'a' in saved_keys:
            drive_roomba(saved_speed, 1); current_action_label = "left"
        elif 'd' in saved_keys:
            drive_roomba(saved_speed, -1); current_action_label = "right"
        else:
            drive_roomba(0, 0); current_action_label = "stop"
            
        start_wait = time.time()
        while time.time() - start_wait < duration:
            if operational_mode != "playback":
                drive_roomba(0, 0)
                return
            time.sleep(0.02)
            
        playback_index += 1
        
    drive_roomba(0, 0)
    current_action_label = ""
    active_key_string = "idling"
    operational_mode = "manual"
    playback_index = 0

# --- Closed-Loop Hardware Encoder Turning Mechanism ---
def execute_closed_loop_turn(target_degrees, direction_sign):
    global current_action_label, last_fetched_angle, active_key_string
    print(f"📐 ENCODER CONTROL: Target turn initialized for {target_degrees}°")
    
    turn_speed = 180
    accumulated_turn = 0.0
    active_key_string = "a" if direction_sign == 1 else "d"
    drive_roomba(turn_speed, direction_sign)
    current_action_label = f"encoder_turning_{target_degrees}"
    
    while abs(accumulated_turn) < (target_degrees - 4): 
        update_sensors_efficiently()
        accumulated_turn += last_fetched_angle
        time.sleep(0.015) 
        
    drive_roomba(0, 0)
    current_action_label = ""
    active_key_string = "idling"

# --- Background Video Stream, Line Follower, & Tracking Loop ---
def camera_and_logic_loop():
    global current_frame, auto_mode, show_grayscale, current_action_label, csv_file, SAMPLE_SIZE, current_speed, active_key_string, frame_count
    global output_dir, csv_path
    
    pipeline = (
        f"nvarguscamerasrc sensor-id=0 "
        f"exposuretimerange='34000000 34000000' "
        f"gainrange='1 8' "
        f"wbmode=1 ! "
        f"video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, width=640, height=360, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )
    
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened(): return

    session_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    documents_dir = os.path.expanduser("~/Documents")
    output_dir = os.path.join(documents_dir, f"roomba_dataset_{session_time}")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"dataset_{SAMPLE_SIZE}x{SAMPLE_SIZE}.csv")
    csv_file = open(csv_path, 'w')
    
    total_pixels = SAMPLE_SIZE * SAMPLE_SIZE
    pixel_headers = ",".join([f"p{i}" for i in range(total_pixels)])
    csv_file.write(f"timestamp,frame_id,action,pressed_keys,bump_left,bump_right,wheel_dropped,total_distance_mm,heading_deg,speed_mm_s,left_wheel_v,right_wheel_v,battery_percent,{pixel_headers}\n")

    last_steering_error = 0.0
    last_time = time.time()

    blue_lower = np.array([95, 130, 20])
    blue_upper = np.array([125, 255, 255])

    wheel_drop_started = None
    hard_safety_triggered = False

    while True:
        ret, frame = cap.read()
        if not ret: break

        b_left, b_right, w_drop = update_sensors_efficiently()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_frame = cv2.resize(gray, (SAMPLE_SIZE, SAMPLE_SIZE), interpolation=cv2.INTER_AREA)
        pixels = small_frame.flatten()
        pixel_str = ",".join(map(str, pixels))

        synced_time = time.time()
        
        # 🛠️ FIXED FRAME INITIALIZER COLUMN ARRAYS
        csv_file.write(f"{synced_time},{frame_count},{current_action_label},{active_key_string},{b_left},{b_right},{w_drop},{total_distance:.1f},{current_heading:.1f},{live_speed_mms:.1f},{live_left_speed_mms},{live_right_speed_mms},{battery_pct:.1f},{pixel_str}\n")
        frame_count += 1

        if w_drop == 1:
            if wheel_drop_started is None: wheel_drop_started = synced_time
            if (synced_time - wheel_drop_started) >= 5.0: hard_safety_triggered = True
        else:
            wheel_drop_started = None
            hard_safety_triggered = False

        if hard_safety_triggered:
            drive_roomba(0, 0)
            current_action_label = ""
            active_key_string = "idling"
            auto_mode = False

        h, w = frame.shape[:2]
        current_time = time.time()

        roi_y_start = 160
        roi_y_end = 360
        roi_x_start = 220
        roi_x_end = 420
        
        roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
        img_yuv = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        equalized_roi = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        hsv_roi = cv2.cvtColor(equalized_roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, blue_lower, blue_upper)
        
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tape_found = False
        cx = (roi_x_end - roi_x_start) // 2 

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 250: 
                tape_found = True
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cv2.drawContours(frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end], [largest_contour], -1, (0, 255, 0), 3)
                    cv2.circle(frame, (roi_x_start + cx, roi_y_start + int((roi_y_end - roi_y_start)/2)), 6, (255, 0, 0), -1)

            cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (255, 120, 0), 2)

        if auto_mode:
            if b_left == 1 or b_right == 1:
                active_key_string = "s"
                drive_roomba(-120, 32767)
                time.sleep(0.4)
                execute_closed_loop_turn(180, random.choice([1, -1]))
            elif tape_found:
                dt = current_time - last_time
                if dt <= 0: dt = 0.033
                
                global_cx = roi_x_start + cx
                screen_center_x = w / 2
                
                steering_error = global_cx - screen_center_x
                error_derivative = (steering_error - last_steering_error) / dt
                
                Kp, Kd = 2.4, 0.16  
                pd_correction = (steering_error * Kp) + (error_derivative * Kd)
                
                if abs(pd_correction) < 5:
                    oi_radius = 32767  
                    active_key_string = "w"
                else:
                    calculated_radius = int(25000 / abs(pd_correction))
                    calculated_radius = max(150, min(2000, calculated_radius))
                    
                    if pd_correction > 0:
                        oi_radius = -calculated_radius  
                        active_key_string = "d"  
                    else:
                        oi_radius = calculated_radius   
                        active_key_string = "a"  
                
                if abs(steering_error) > 40: 
                    drive_roomba(110, oi_radius)  
                else: 
                    drive_roomba(current_speed, oi_radius)  
                    
                current_action_label = "line_following"
                last_steering_error = steering_error
                last_time = current_time
            else:
                current_action_label = "searching_tape"
                active_key_string = "s"
                drive_roomba(-100, 32767)
                time.sleep(0.4)
                
                execute_closed_loop_turn(180, 1)
                
                scan_directions = [-1, 1, -1]
                scan_durations = [1.2, 2.4, 1.2]
                
                tape_locked = False
                for sweep_dir, sweep_time in zip(scan_directions, scan_durations):
                    if tape_found:
                        tape_locked = True
                        break
                    active_key_string = "d" if sweep_dir == -1 else "a"
                    drive_roomba(110, sweep_dir)
                    
                    start_sweep = time.time()
                    while time.time() - start_sweep < sweep_time:
                        if tape_found:
                            tape_locked = True
                            break
                        time.sleep(0.02)
                        
                if tape_locked:
                    drive_roomba(0, 0)
                    active_key_string = "idling"
                    last_steering_error = 0.0
                    last_time = time.time()
                else:
                    drive_roomba(0, 0)
                    active_key_string = "idling"
                    auto_mode = False
                    current_action_label = ""
        else:
            if current_action_label == "sticky_spin_left":
                drive_roomba(150, 1)
                active_key_string = "a"
            elif current_action_label == "sticky_spin_right":
                drive_roomba(150, -1)
                active_key_string = "d"
            elif current_action_label == "sticky_spin_random":
                if active_key_string not in ["a", "d"]:
                    active_key_string = random.choice(["a", "d"])
                chosen_radius = 1 if active_key_string == "a" else -1
                drive_roomba(150, chosen_radius)
                
            last_steering_error = 0.0
            last_time = current_time

        with lock:
            if show_grayscale:
                upscaled_ai_preview = cv2.resize(small_frame, (640, 360), interpolation=cv2.INTER_NEAREST)
                current_frame = cv2.cvtColor(upscaled_ai_preview, cv2.COLOR_GRAY2BGR)
            else:
                hud_frame = frame.copy()
                current_frame = draw_guidelines(hud_frame)

# --- Flask Web Server Routes ---
@app.route('/')
def index():
    html = """
    <html>
        <head>
            <title>Roomba AI Command Dashboard</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
            <style>
                :root {
                    --bg-primary: #0f0f12;
                    --bg-surface: #191921;
                    --bg-glass: rgba(25, 25, 33, 0.75);
                    --accent-cyan: #00ffcc;
                    --accent-blue: #3a86ff;
                    --accent-orange: #ff9f1c;
                    --accent-red: #ff5555;
                    --accent-green: #55ff55;
                    --text-main: #f0f0f5;
                    --text-muted: #8a8a9e;
                }
                
                body { background-color: var(--bg-primary); color: var(--text-main); font-family: -apple-system, BlinkMacSystemFont, monospace; margin: 0; padding: 20px; user-select: none; }
                h1 { font-size: 20px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 20px; color: var(--text-main); text-align: center;}
                
                .dashboard-container { display: flex; flex-direction: column; align-items: center; max-width: 980px; margin: 0 auto; gap: 20px; }
                
                .viewport-row { display: flex; flex-direction: row; gap: 20px; width: 880px; max-width: 100%; align-items: stretch; justify-content: center; }
                @media(max-width: 900px) { .viewport-row { flex-direction: column; width: 640px; } }
                
                .video-hud-container { position: relative; width: 640px; aspect-ratio: 16/9; background: #000; border: 2px solid #2d2d3d; border-radius: 12px; overflow: hidden; box-shadow: 0 8px 32px rgba(0,0,0,0.5); }
                .video-hud-container img { width: 100%; height: 100%; display: block; object-fit: cover; }
                
                .logger-panel-box { width: 220px; background: var(--bg-surface); border: 2px solid #2d2d3d; border-radius: 12px; padding: 15px; display: flex; flex-direction: column; gap: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.5); box-sizing: border-box; }
                .log-data-card { background: #111116; border: 1px solid #272736; border-radius: 8px; padding: 10px; display: flex; flex-direction: column; gap: 4px; text-align: left;}
                .log-label { font-size: 10px; text-transform: uppercase; color: var(--text-muted); font-weight: bold; letter-spacing: 0.5px; }
                .log-value-keys { color: #ffff00; font-size: 16px; font-weight: bold; letter-spacing: 1px; font-family: -apple-system, monospace; text-align: center; padding: 4px 0;}
                .log-value-action { color: #00ffcc; font-size: 13px; font-weight: bold; font-family: monospace; overflow-wrap: break-word; word-break: break-all; min-height: 18px;}
                .log-value-frames { color: #e2afff; font-size: 15px; font-weight: bold; text-align: center; font-family: monospace; padding: 2px 0; }

                .hud-telemetry-box { position: absolute; top: 15px; left: 15px; background: var(--bg-glass); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 12px; text-align: left; font-size: 12px; backdrop-filter: blur(8px); pointer-events: none; z-index: 10; display: grid; gap: 4px; min-width: 160px; }
                .hud-telemetry-box div { display: flex; justify-content: space-between; gap: 10px; }
                .hud-status-text { color: var(--accent-cyan); font-weight: bold; }
                .hud-battery-text { color: var(--accent-orange); font-weight: bold; }
                .hud-speed-text { color: var(--accent-blue); }
                .hud-vac-off { color: var(--accent-red); font-weight: bold; }
                .hud-vac-on { color: var(--accent-green); font-weight: bold; }

                .control-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; width: 880px; max-width: 100%; }
                @media(max-width: 900px) { .control-grid { grid-template-columns: 1fr; width: 640px; } }
                
                .panel-card { background: var(--bg-surface); border: 1px solid #2d2d3d; border-radius: 12px; padding: 15px; display: flex; flex-direction: column; gap: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
                .panel-title { font-size: 11px; text-transform: uppercase; color: var(--text-muted); letter-spacing: 1px; border-bottom: 1px solid #2d2d3d; padding-bottom: 6px; margin-bottom: 4px; font-weight: bold; text-align: left;}
                
                .btn-group { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
                .btn-group-full { display: flex; flex-direction: column; gap: 8px; }
                
                .dashboard-btn { background: #22222e; color: var(--text-main); border: 1px solid #3d3d52; border-radius: 6px; padding: 10px 14px; font-size: 11px; font-weight: bold; font-family: monospace; cursor: pointer; transition: all 0.2s ease; display: flex; align-items: center; justify-content: center; gap: 6px; min-height: 38px; box-sizing: border-box; text-decoration: none; }
                .dashboard-btn:hover { background: #2c2240; border-color: var(--accent-blue); }
                .dashboard-btn:active { transform: scale(0.98); }
                
                .btn-cyan { border-color: var(--accent-cyan); color: var(--accent-cyan); }
                .btn-cyan:hover { background: rgba(0, 255, 204, 0.1); }
                .btn-orange { border-color: var(--accent-orange); color: var(--accent-orange); }
                .btn-orange:hover { background: rgba(255, 159, 28, 0.1); }
                .btn-red { border-color: var(--accent-red); color: var(--accent-red); }
                .btn-red:hover { background: rgba(255, 85, 85, 0.1); }
                .btn-green { border-color: var(--accent-green); color: var(--accent-green); }
                .btn-green:hover { background: rgba(85, 255, 85, 0.1); }
                
                .active-tracking { background: rgba(0, 255, 255, 0.15) !important; border-color: var(--accent-cyan) !important; color: var(--accent-cyan) !important; box-shadow: 0 0 12px rgba(0,255,245,0.2); }
                .active-recording { background: rgba(255, 85, 85, 0.15) !important; border-color: var(--accent-red) !important; color: var(--accent-red) !important; animation: pulse-border 1.5s infinite; }
                @keyframes pulse-border { 0% { box-shadow: 0 0 0 0 rgba(255,85,85,0.4); } 70% { box-shadow: 0 0 0 6px rgba(255,85,85,0); } 100% { box-shadow: 0 0 0 0 rgba(255,85,85,0); } }

                .dpad-overlay { position: absolute; bottom: 15px; left: 50%; transform: translateX(-50%); display: grid; grid-template-columns: repeat(3, 50px); grid-template-rows: repeat(3, 45px); gap: 8px; pointer-events: none; z-index: 5; transition: opacity 0.25s ease; opacity: 0; }
                .dpad-visible { opacity: 1 !important; pointer-events: auto !important; }
                .touch-btn { background: rgba(15, 15, 18, 0.85); color: var(--accent-cyan); border: 1px solid var(--accent-cyan); border-radius: 6px; font-size: 16px; display: flex; align-items: center; justify-content: center; cursor: pointer; pointer-events: auto; backdrop-filter: blur(4px); transition: all 0.1s; }
                .touch-btn:active { background: var(--accent-cyan); color: #000; }
                .center-space { grid-column: 2; grid-row: 2; font-size: 9px; font-weight: bold; border-color: var(--text-muted); color: var(--text-muted); }

                .instructions { color: var(--text-muted); font-size: 11px; line-height: 1.6; text-align: center; max-width: 880px; background: var(--bg-surface); padding: 12px; border-radius: 8px; border: 1px solid #2d2d3d; width: 100%; box-sizing: border-box; }
                .key { display: inline-block; background: #2a2a38; padding: 2px 6px; border-radius: 4px; margin: 0 2px; color: #fff; font-weight: bold; border: 1px solid #444;}
            </style>
            <script>
                let keys = new Set();
                const validKeys = ['w','s','a','d','q','e','v','space'];
                
                document.addEventListener('keydown', (e) => {
                    let key = e.key.toLowerCase();
                    if (key === ' ') key = 'space';
                    if (validKeys.includes(key) && !keys.has(key)) { keys.add(key); sendState(); }
                });
                
                document.addEventListener('keyup', (e) => {
                    let key = e.key.toLowerCase();
                    if (key === ' ') key = 'space';
                    if (keys.has(key)) { keys.delete(key); sendState(); }
                });
                
                function sendState() { fetch('/keyboard_input?keys=' + Array.from(keys).join('-')); }
                
                function toggleAutoMode() {
                    fetch('/toggle_auto').then(res => res.text()).then(txt => { 
                        let autoBtn = document.getElementById("hud-auto-btn");
                        if (txt.includes("ACTIVE")) { autoBtn.classList.add("active-tracking"); } 
                        else { autoBtn.classList.remove("active-tracking"); }
                    });
                }

                function toggleGrayscaleView() {
                    fetch('/toggle_grayscale').then(res => res.json()).then(data => {
                        let btn = document.getElementById("hud-gray-btn");
                        if (data.grayscale) { btn.classList.add("active-tracking"); btn.innerText = "👁️ Standard HUD View"; } 
                        else { btn.classList.remove("active-tracking"); btn.innerText = "🌓 AI Matrix View (Grayscale)"; }
                    });
                }

                function advanceMacroState() {
                    fetch('/advance_macro_mode').then(res => res.json()).then(data => {
                        let btn = document.getElementById("hud-macro-btn");
                        let pBtn = document.getElementById("hud-pause-btn");
                        document.getElementById("hud-mode-status").innerText = data.label.toUpperCase() + " MODE";
                        btn.className = "dashboard-btn " + data.css_class;
                        btn.innerText = data.button_text;
                        pBtn.style.display = data.show_pause ? "inline-block" : "none";
                        if(data.pause_text) pBtn.innerText = data.pause_text;
                    });
                }

                document.addEventListener('fullscreenchange', () => {
                    let fsBtn = document.getElementById("hud-fs-btn");
                    if (document.fullscreenElement) { fsBtn.innerText = "📺 Minimize HUD"; } 
                    else { fsBtn.innerText = "📺 Full Screen"; }
                });

                function togglePlayPause() {
                    fetch('/toggle_play_pause').then(res => res.json()).then(data => {
                        document.getElementById("hud-mode-status").innerText = data.label.toUpperCase() + " MODE";
                        document.getElementById("hud-pause-btn").innerText = data.pause_text;
                        document.getElementById("hud-macro-btn").className = "dashboard-btn " + data.css_class;
                    });
                }
                
                function toggleDpadVisibility() {
                    let dpad = document.getElementById("touch-dpad");
                    let btn = document.getElementById("hud-dpad-toggle");
                    if (dpad.classList.contains("dpad-visible")) {
                        dpad.classList.remove("dpad-visible");
                        btn.classList.remove("btn-cyan");
                    } else {
                        dpad.classList.add("dpad-visible");
                        btn.classList.add("btn-cyan");
                    }
                }

                function toggleFullScreen() {
                    let container = document.getElementById("hud-root");
                    if (!document.fullscreenElement) { container.requestFullscreen(); } 
                    else { document.exitFullscreen(); }
                }

                document.addEventListener('fullscreenchange', () => {
                    let fsBtn = document.getElementById("hud-fs-btn");
                    if (document.fullscreenElement) { fsBtn.innerText = "📺 Minimize HUD"; } 
                    else { fsBtn.innerText = "📺 Full Screen"; }
                });

                function triggerFullScreenMode(actionType) { fetch('/trigger_sticky_spin?mode=' + actionType); }

                function triggerDocking() {
                    fetch('/dock').then(res => res.text()).then(txt => {
                        document.getElementById("hud-mode-status").innerText = txt;
                        document.getElementById("hud-auto-btn").classList.remove("active-tracking");
                        document.getElementById("hud-dock-btn").style.display = "none";
                        document.getElementById("hud-cancel-dock-btn").style.display = "inline-block";
                    });
                }

                function triggerCancelDocking() {
                    fetch('/cancel_dock').then(res => res.text()).then(txt => {
                        document.getElementById("hud-mode-status").innerText = txt;
                        document.getElementById("hud-dock-btn").style.display = "inline-block";
                        document.getElementById("hud-cancel-dock-btn").style.display = "none";
                    });
                }
                
                function triggerFixedDegreeTurn(angle, dir) { fetch('/trigger_degree_turn?angle=' + angle + '&dir=' + dir); }
                function sendSingleAction(actionKey) { fetch('/keyboard_input?keys=' + actionKey); }
                function touchStartAction(actionKey) { keys.add(actionKey); sendState(); }
                function touchEndAction(actionKey) { keys.delete(actionKey); sendState(); }

                setInterval(() => {
                    fetch('/robot_stats')
                        .then(res => res.json())
                        .then(data => {
                            document.getElementById("hud-battery").innerText = data.battery.toFixed(1) + "%";
                            document.getElementById("hud-live-speed").innerText = data.live_speed.toFixed(1) + " mm/s";
                            document.getElementById("hud-target-speed").innerText = data.target_speed + " mm/s";
                            
                            document.getElementById("live-logged-keys").innerText = data.logged_keys.toUpperCase();
                            document.getElementById("live-frame-count").innerText = data.frame_count;

                            if (data.logged_action === "") {
                                document.getElementById("live-logged-action").innerHTML = "<i style='color:var(--text-muted); font-weight:normal;'>[empty_idle]</i>";
                            } else {
                                document.getElementById("live-logged-action").innerText = data.logged_action;
                            }
                            
                            let vacLabel = document.getElementById("hud-vacuum");
                            let hudVacBtn = document.getElementById("hud-vac-btn");
                            if (data.vacuum) {
                                vacLabel.innerText = "ACTIVE"; vacLabel.className = "hud-vac-on";
                                hudVacBtn.classList.add("btn-green"); hudVacBtn.innerText = "🌪️ Vacuum: ON";
                            } else {
                                vacLabel.innerText = "OFF"; vacLabel.className = "hud-vac-off";
                                hudVacBtn.classList.remove("btn-green"); hudVacBtn.innerText = "🌪️ Vacuum: OFF";
                            }
                        });
                }, 200);
            </script>
        </head>
        <body>
            <div class="dashboard-container">
                <h1>Roomba AI Command Center</h1>
                
                <div class="viewport-row">
                    <div id="hud-root" class="video-hud-container">
                        <img src="/video_feed" />
                        
                        <div class="hud-telemetry-box">
                            <div><span>🖥️ System:</span> <span id="hud-mode-status" class="hud-status-text">MANUAL</span></div>
                            <div><span>⚡ Battery:</span> <span id="hud-battery" class="hud-battery-text">100.0%</span></div>
                            <div><span>🏎️ Telemetry:</span> <span id="hud-live-speed" class="hud-speed-text">0.0 mm/s</span></div>
                            <div><span>🎯 Baseline:</span> <span id="hud-target-speed" style="color:#e2afff;">150 mm/s</span></div>
                            <div><span>🌪️ Vacuum:</span> <span id="hud-vacuum" class="hud-vac-off">OFF</span></div>
                        </div>
                        
                        <div id="touch-dpad" class="dpad-overlay">
                            <div class="touch-btn" style="grid-column: 2; grid-row: 1;" ontouchstart="touchStartAction('w')" ontouchend="touchEndAction('w')" onmousedown="touchStartAction('w')" onmouseup="touchEndAction('w')">▲</div>
                            <div class="touch-btn" style="grid-column: 1; grid-row: 2;" ontouchstart="touchStartAction('a')" ontouchend="touchEndAction('a')" onmousedown="touchStartAction('a')" onmouseup="touchEndAction('a')">◀</div>
                            <div class="touch-btn center-space" style="grid-column: 2; grid-row: 2;" ontouchstart="touchStartAction('space')" ontouchend="touchEndAction('space')" onmousedown="touchStartAction('space')" onmouseup="touchEndAction('space')">STOP</div>
                            <div class="touch-btn" style="grid-column: 3; grid-row: 2;" ontouchstart="touchStartAction('d')" ontouchend="touchEndAction('d')" onmousedown="touchStartAction('d')" onmouseup="touchEndAction('d')">▶</div>
                            <div class="touch-btn" style="grid-column: 2; grid-row: 3;" ontouchstart="touchStartAction('s')" ontouchend="touchEndAction('s')" onmousedown="touchStartAction('s')" onmouseup="touchEndAction('s')">▼</div>
                        </div>
                    </div>

                    <div class="logger-panel-box">
                        <div class="panel-title" style="border:none; padding:0; margin:0;">📟 Dataset Stream</div>
                        
                        <div class="log-data-card" style="margin-top:5px; border-color: rgba(255,255,0,0.25);">
                            <div class="log-label">Logged Key Value</div>
                            <div id="live-logged-keys" class="log-value-keys">IDLING</div>
                        </div>

                        <div class="log-data-card" style="border-color: rgba(0,255,204,0.2);">
                            <div class="log-label">Current Action Matrix</div>
                            <div id="live-logged-action" class="log-value-action"></div>
                        </div>

                        <div class="log-data-card" style="flex-grow:1; border-color: rgba(226, 175, 255, 0.25);">
                            <div class="log-label">Processed Frame ID</div>
                            <div id="live-frame-count" class="log-value-frames">0</div>
                        </div>
                    </div>
                </div>

                <div class="control-grid">
                    <div class="panel-card">
                        <div class="panel-title">Navigation & Vision Modes</div>
                        <button id="hud-auto-btn" class="dashboard-btn var(--accent-cyan)" onclick="toggleAutoMode()">🤖 Blue Line Follower</button>
                        <button id="hud-gray-btn" class="dashboard-btn" onclick="toggleGrayscaleView()">🌓 AI Matrix View (Grayscale)</button>
                        <button class="dashboard-btn" onclick="toggleFullScreen()">📺 Full Screen HUD</button>
                    </div>

                    <div class="panel-card">
                        <div class="panel-title">Macro Dataset Recorder</div>
                        <button id="hud-macro-btn" class="dashboard-btn btn-manual" onclick="advanceMacroState()">🎙️ Start Recording</button>
                        <button id="hud-pause-btn" class="dashboard-btn btn-orange" style="display:none;" onclick="togglePlayPause()">⏸️ Pause Playback</button>
                        <div class="btn-group">
                            <button id="hud-dock-btn" class="dashboard-btn btn-orange" onclick="triggerDocking()">🔌 Seek Base</button>
                            <button id="hud-cancel-dock-btn" class="dashboard-btn btn-red" onclick="triggerCancelDocking()">Cancel Dock</button>
                            <button id="hud-dpad-toggle" class="dashboard-btn" onclick="toggleDpadVisibility()">🎮 Toggle Touch HUD</button>
                        </div>
                    </div>

                    <div class="panel-card">
                        <div class="panel-title">Continuous Spin Utilities</div>
                        <div class="btn-group-full">
                            <button class="dashboard-btn btn-orange" onclick="triggerFullScreenMode('sticky_spin_random')">🔄 Sticky Spin Random</button>
                            <div class="btn-group">
                                <button class="dashboard-btn btn-red" onclick="triggerFullScreenMode('sticky_spin_left')">🔄 Spin Left Indefinitely</button>
                                <button class="dashboard-btn btn-green" onclick="triggerFullScreenMode('sticky_spin_right')">🔄 Spin Right Indefinitely</button>
                            </div>
                        </div>
                    </div>

                    <div class="panel-card">
                        <div class="panel-title">Hardware Actuators & Precision Steps</div>
                        <div class="btn-group">
                            <button class="dashboard-btn" onclick="triggerFixedDegreeTurn(90, 'left')">📐 Left 90°</button>
                            <button class="dashboard-btn" onclick="triggerFixedDegreeTurn(90, 'right')">📐 Right 90°</button>
                        </div>
                        <button class="dashboard-btn" style="border-color:#b5179e; color:#b5179e;" onclick="triggerFixedDegreeTurn(180, 'left')">📐 Spin Around 180°</button>
                        <div class="btn-group">
                            <button id="hud-vac-btn" class="dashboard-btn" onclick="sendSingleAction('v')">🌪️ Vacuum Toggle</button>
                            <div class="btn-group">
                                <button class="dashboard-btn" onclick="sendSingleAction('e')">🚀 Volts +</button>
                                <button class="dashboard-btn" onclick="sendSingleAction('q')">🐌 Volts -</button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="instructions">
                    <p><span class="key">W</span> / <span class="key">S</span> = Forward/Backward | <span class="key">A</span> / <span class="key">D</span> = Spin Axis | <span class="key">SPACE</span> = Emergency Brake</p>
                    <p><span class="key">Q</span> / <span class="key">E</span> = Velocity Scaling (50 - 600 mm/s) | <span class="key">V</span> = Vacuum Toggle</p>
                </div>
            </div>
        </body>
    </html>
    """
    return render_template_string(html)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with lock:
                if current_frame is None: continue
                ret, jpeg = cv2.imencode('.jpg', current_frame)
            if ret: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.03) 
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_auto')
def toggle_auto():
    global auto_mode, current_action_label, operational_mode, active_key_string
    auto_mode = not auto_mode
    operational_mode = "manual"
    drive_roomba(0, 0)
    current_action_label = ""
    active_key_string = "idling"
    return "AUTO MODE ACTIVE" if auto_mode else "MANUAL MODE"

@app.route('/toggle_grayscale')
def toggle_grayscale():
    global show_grayscale
    show_grayscale = not show_grayscale
    return jsonify({"grayscale": show_grayscale})

@app.route('/advance_macro_mode')
def advance_macro_mode():
    global operational_mode, movement_history, current_move_start, last_saved_keys, auto_mode, playback_index, active_key_string
    auto_mode = False
    
    if operational_mode in ["manual", "paused"]:
        operational_mode = "record"
        movement_history = []
        playback_index = 0
        last_saved_keys = set(['idling'])
        current_move_start = time.time()
        active_key_string = "idling"
        return jsonify({"label": "recording", "css_class": "btn-record active-recording", "button_text": "⏹️ Stop & Playback", "show_pause": False})
        
    elif operational_mode == "record":
        if current_move_start is not None:
            duration = time.time() - current_move_start
            if duration > 0.05:
                movement_history.append(([active_key_string], duration, current_speed))
                
        drive_roomba(0, 0)
        active_key_string = "idling"
        if len(movement_history) > 0:
            operational_mode = "playback"
            playback_index = 0
            threading.Thread(target=execution_playback_worker, daemon=True).start()
            return jsonify({"label": "playback", "css_class": "btn-playback", "button_text": "🎙️ Clear & Re-Record", "show_pause": True, "pause_text": "⏸️ Pause Playback"})
        else:
            operational_mode = "manual"
            return jsonify({"label": "manual", "css_class": "btn-manual", "button_text": "🎙️ Start Recording", "show_pause": False})
            
    else:
        operational_mode = "manual"
        playback_index = 0
        drive_roomba(0, 0)
        active_key_string = "idling"
        return jsonify({"label": "manual", "css_class": "btn-manual", "button_text": "🎙️ Start Recording", "show_pause": False})

@app.route('/toggle_play_pause')
def toggle_play_pause():
    global operational_mode, active_key_string
    if operational_mode == "playback":
        operational_mode = "paused"
        drive_roomba(0, 0)
        active_key_string = "idling"
        return jsonify({"label": "paused", "css_class": "btn-paused", "pause_text": "▶️ Resume Playback"})
    elif operational_mode == "paused":
        operational_mode = "playback"
        threading.Thread(target=execution_playback_worker, daemon=True).start()
        return jsonify({"label": "playback", "css_class": "btn-playback", "pause_text": "⏸️ Pause Playback"})
    return jsonify({"label": operational_mode, "css_class": "btn-manual", "pause_text": "⏸️ Pause Playback"})

@app.route('/trigger_degree_turn')
def trigger_degree_turn():
    angle_param = request.args.get('angle', 90, type=int)
    direction_param = request.args.get('dir', 'left')
    direction_sign = 1 if direction_param == 'left' else -1
    threading.Thread(target=execute_closed_loop_turn, args=(angle_param, direction_sign,), daemon=True).start()
    return "OK", 200

@app.route('/trigger_sticky_spin')
def trigger_sticky_spin():
    global current_action_label, active_key_string, auto_mode
    auto_mode = False
    mode_param = request.args.get('mode', 'sticky_spin_random')
    current_action_label = mode_param
    active_key_string = "idling"  
    return "OK", 200

@app.route('/dock')
def dock():
    global auto_mode, operational_mode, current_action_label, vacuum_on, active_key_string
    auto_mode = False
    operational_mode = "manual"
    vacuum_on = False 
    current_action_label = "docking"
    active_key_string = "idling"
    dock_roomba()
    return "SEEKING HOME DOCK..."

@app.route('/cancel_dock')
def cancel_dock():
    global current_action_label, active_key_string
    current_action_label = ""
    active_key_string = "idling"
    cancel_docking_sequence()
    return "MANUAL CONTROL RESTORED"

@app.route('/robot_stats')
def robot_stats():
    return jsonify({
        "live_speed": abs(live_speed_mms),
        "target_speed": current_speed,
        "battery": battery_pct,
        "vacuum": vacuum_on,
        "logged_keys": active_key_string,
        "logged_action": current_action_label,
        "frame_count": frame_count  
    })

@app.route('/keyboard_input')
def keyboard_input():
    global current_speed, vacuum_on, last_keys_set, auto_mode, current_action_label, operational_mode, current_move_start, last_saved_keys, active_key_string
    raw_keys = request.args.get('keys', '')
    current_keys = set(raw_keys.split('-')) if raw_keys else set()
    
    newly_pressed = current_keys - last_keys_set
    if 'q' in newly_pressed: current_speed = max(50, current_speed - 50)
    if 'e' in newly_pressed: current_speed = min(600, current_speed + 50)
    if 'v' in newly_pressed: 
        vacuum_on = not vacuum_on
        set_vacuum(vacuum_on)
        
    last_keys_set = current_keys

    if (operational_mode in ["playback", "paused"] or current_action_label == "docking" or "sticky" in current_action_label) and len(current_keys) > 0 and 'space' not in current_keys:
        operational_mode = "manual"
        cancel_docking_sequence()
        current_action_label = ""
        active_key_string = "idling"

    if not auto_mode and operational_mode != "playback" and operational_mode != "paused" and current_action_label != "precision_turning" and "encoder_turning" not in current_action_label and current_action_label != "docking" and "sticky" not in current_action_label:
        filtered_move_keys = current_keys.intersection({'w', 'a', 's', 'd', 'space'})
        
        if not filtered_move_keys:
            active_key_string = "idling"
        elif 'space' in filtered_move_keys:
            active_key_string = "space"
        elif 'w' in filtered_move_keys:
            active_key_string = "w"
        elif 's' in filtered_move_keys:
            active_key_string = "s"
        elif 'a' in filtered_move_keys:
            active_key_string = "a"
        elif 'd' in filtered_move_keys:
            active_key_string = "d"

        if operational_mode == "record":
            if filtered_move_keys != last_saved_keys:
                now = time.time()
                if current_move_start is not None:
                    duration = now - current_move_start
                    if duration > 0.05:
                        movement_history.append(([active_key_string], duration, current_speed))
                last_saved_keys = filtered_move_keys
                current_move_start = now

        if not filtered_move_keys:
            drive_roomba(0, 0)
            current_action_label = ""
        elif 'space' in filtered_move_keys: 
            drive_roomba(0, 0)
            current_action_label = "stop"
        elif 'w' in filtered_move_keys: 
            drive_roomba(current_speed, 32767)
            current_action_label = "forward"
        elif 's' in filtered_move_keys: 
            drive_roomba(-current_speed, 32767)
            current_action_label = "backward"
        elif 'a' in filtered_move_keys: 
            drive_roomba(current_speed, 1)
            current_action_label = "left"
        elif 'd' in filtered_move_keys: 
            drive_roomba(current_speed, -1)
            current_action_label = "right"
        
    return "OK", 200

# --- Standard Local Cleanup Exit Hook Handler ---
def safely_close_local_session_on_exit():
    global csv_file
    drive_roomba(0, 0)  
    if csv_file:
        try:
            csv_file.flush()
            csv_file.close()
            print("\nDataset loop closed. Local telemetry CSV saved securely to your Documents folder.")
        except Exception as e:
            print(f"Error packing file resource: {e}")

atexit.register(safely_close_local_session_on_exit) 

if __name__ == '__main__':
    threading.Thread(target=camera_and_logic_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
