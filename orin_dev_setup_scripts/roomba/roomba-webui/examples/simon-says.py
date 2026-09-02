import cv2
import numpy as np
import threading
import time
import serial
import os
import random
from datetime import datetime
from flask import Flask, Response, render_template_string, request, jsonify

# --- App Initialization & Globals ---
app = Flask(__name__)
current_frame = None  
lock = threading.Lock()

# --- DATASET GRID CONFIGURATION ---
SAMPLE_SIZE = 25

# Navigation, Logging, & Stream Globals
current_speed = 150  # Stable line tracking target velocity (mm/s)
vacuum_on = False
last_keys_set = set()
auto_mode = False
show_grayscale = False  
current_action_label = "stop"

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

# Live Performance Metrics
live_speed_mms = 0.0
battery_pct = 100.0

# CSV Logging File Descriptor
csv_file = None

# --- Roomba Serial Setup ---
try:
    roomba = serial.Serial('/dev/ttyUSB0', baudrate=115200, timeout=0.1)
    roomba.write(bytes([128, 131])) # Start Safe Mode
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

def cancel_docking_sequence():
    if roomba:
        roomba.write(bytes([128, 131])) 
        time.sleep(0.02)
        drive_roomba(0, 0)

def update_sensors_efficiently():
    global total_distance, current_heading, live_speed_mms, battery_pct, last_fetched_angle
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
    global operational_mode, current_action_label, playback_index
    while playback_index < len(movement_history):
        if operational_mode != "playback":
            return
            
        saved_keys, duration, saved_speed = movement_history[playback_index]
        
        if 'space' in saved_keys or not saved_keys:
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
    current_action_label = "stop"
    operational_mode = "manual"
    playback_index = 0

# --- Closed-Loop Hardware Encoder Turning Mechanism ---
def execute_closed_loop_turn(target_degrees, direction_sign):
    global current_action_label, last_fetched_angle
    print(f"📐 ENCODER CONTROL: Target turn initialized for {target_degrees}°")
    
    turn_speed = 180
    accumulated_turn = 0.0
    drive_roomba(turn_speed, direction_sign)
    current_action_label = f"encoder_turning_{target_degrees}"
    
    while abs(accumulated_turn) < (target_degrees - 4): 
        update_sensors_efficiently()
        accumulated_turn += last_fetched_angle
        time.sleep(0.015) 
        
    drive_roomba(0, 0)
    current_action_label = "stop"
    print(f"🏁 ENCODER LOCKED: Turn complete at true angle value: {abs(accumulated_turn)}°")

# --- Background Video Stream, Line Follower, & Tracking Loop ---
def camera_and_logic_loop():
    global current_frame, auto_mode, show_grayscale, current_action_label, csv_file, SAMPLE_SIZE, current_speed
    
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
    csv_file.write(f"timestamp,action,bump_left,bump_right,wheel_dropped,total_distance_mm,heading_deg,speed_mm_s,battery_percent,{pixel_headers}\n")

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
        csv_file.write(f"{synced_time},{current_action_label},{b_left},{b_right},{w_drop},{total_distance:.1f},{current_heading:.1f},{live_speed_mms:.1f},{battery_pct:.1f},{pixel_str}\n")

        if w_drop == 1:
            if wheel_drop_started is None: wheel_drop_started = synced_time
            if (synced_time - wheel_drop_started) >= 5.0: hard_safety_triggered = True
        else:
            wheel_drop_started = None
            hard_safety_triggered = False

        if hard_safety_triggered:
            drive_roomba(0, 0)
            current_action_label = "stop"
            auto_mode = False

        # --- GLOBAL BLUE LINE SCANNER AND HUD RENDERING ENGINE ---
        h, w = frame.shape[:2]
        current_time = time.time()

        # 🛠️ OVERHAULED EXPANDED SQUARE INSPECTION WINDOW (200x200)
        # Symmetrically sets bounds to 200px wide and 200px high directly on the bumper base plane
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
            # Area constraint adjusted slightly to fit the wider square footprint field
            if cv2.contourArea(largest_contour) > 250: 
                tape_found = True
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cv2.drawContours(frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end], [largest_contour], -1, (0, 255, 0), 3)
                    cv2.circle(frame, (roi_x_start + cx, roi_y_start + int((roi_y_end - roi_y_start)/2)), 6, (255, 0, 0), -1)

            cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (255, 120, 0), 2)

        # --- AUTO MODE: LINE FOLLOWER WITH CLOSED-LOOP ENCODER RECOVERY ---
        if auto_mode:
            if b_left == 1 or b_right == 1:
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
                else:
                    calculated_radius = int(25000 / abs(pd_correction))
                    calculated_radius = max(150, min(2000, calculated_radius))
                    
                    if pd_correction > 0:
                        oi_radius = -calculated_radius  
                    else:
                        oi_radius = calculated_radius   
                
                if abs(steering_error) > 40: 
                    drive_roomba(110, oi_radius)  
                else: 
                    drive_roomba(current_speed, oi_radius)  
                    
                current_action_label = "line_following"
                last_steering_error = steering_error
                last_time = current_time
            else:
                print("🚧 TAPE LOST: Initiating hardware encoder path scan...")
                current_action_label = "searching_tape"
                
                drive_roomba(-100, 32767)
                time.sleep(0.4)
                
                print(" └─ Spinning precise 180 via wheel metrics...")
                execute_closed_loop_turn(180, 1)
                
                scan_directions = [-1, 1, -1]
                scan_durations = [1.2, 2.4, 1.2]
                
                tape_locked = False
                for sweep_dir, sweep_time in zip(scan_directions, scan_durations):
                    if tape_found:
                        tape_locked = True
                        break
                        
                    print(f" └─ Checking square field overlay bounds...")
                    drive_roomba(110, sweep_dir)
                    
                    start_sweep = time.time()
                    while time.time() - start_sweep < sweep_time:
                        if tape_found:
                            tape_locked = True
                            break
                        time.sleep(0.02)
                        
                if tape_locked:
                    print("🎯 TARGET RE-LOCKED: Blue path acquired. Resuming PD tracking loop.")
                    drive_roomba(0, 0)
                    last_steering_error = 0.0
                    last_time = time.time()
                else:
                    print("🛑 RECOVERY FAILED: Path completely lost. Halting for manual intervention.")
                    drive_roomba(0, 0)
                    auto_mode = False
                    current_action_label = "stop"
        else:
            last_steering_error = 0.0
            last_time = current_time

        with lock:
            if show_grayscale:
                current_frame = cv2.resize(small_frame, (640, 360), interpolation=cv2.INTER_NEAREST)
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)
            else:
                hud_frame = frame.copy()
                current_frame = draw_guidelines(hud_frame)

# --- Flask Web Server Routes ---
@app.route('/')
def index():
    html = """
    <html>
        <head>
            <title>Roomba AI Data Center</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
            <style>
                body { background-color: #1a1a1a; color: white; text-align: center; font-family: monospace; margin: 0; padding: 10px; user-select: none; -webkit-user-select: none; }
                .video-hud-container { position: relative; display: inline-block; width: 640px; max-width: 100%; margin: 0 auto; background: #1a1a1a; }
                .video-hud-container img { width: 100%; height: auto; border: 2px solid #555; border-radius: 6px; display: block; box-sizing: border-box; }
                
                .video-hud-container:fullscreen { width: 100vw !important; height: 100vh !important; display: flex; align-items: center; justify-content: center; background-color: black; }
                .video-hud-container:fullscreen img { width: auto; height: 100vh; max-width: 100vw; border: none; border-radius: 0; }

                .hud-telemetry-box { position: absolute; top: 15px; left: 15px; background: rgba(38, 38, 43, 0.75); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 6px; padding: 10px 14px; text-align: left; font-size: 13px; backdrop-filter: blur(4px); pointer-events: none; z-index: 10; }
                .hud-telemetry-box div { margin: 5px 0; }
                .hud-status-text { color: #00ffcc; font-weight: bold; }
                .hud-battery-text { color: #ffb703; }
                .hud-speed-text { color: #a2d2ff; }

                .hud-controls-box { position: absolute; top: 15px; right: 15px; display: flex; flex-direction: column; gap: 8px; pointer-events: auto; z-index: 10; }
                .hud-btn { padding: 8px 12px; font-size: 11px; background: rgba(0, 0, 0, 0.6); color: white; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 4px; cursor: pointer; font-family: monospace; font-weight: bold; backdrop-filter: blur(2px); }
                .hud-btn-active-auto { border-color: #00ffff; color: #00ffff; font-weight: bold; }
                .hud-btn-vac { border-color: #ff6b6b; color: #ff6b6b; }
                .hud-btn-vac.active { border-color: #28a745; color: #28a745; }
                .hud-btn-spd { border-color: #e2afff; color: #e2afff; }
                .hud-btn-degree { border-color: #ffd166; color: #ffd166; }
                
                .btn-manual { border-color: #a2d2ff; color: #a2d2ff; }
                .btn-record { border-color: #ff6b6b; color: #ff6b6b; font-weight: bold; animation: pulse 1.5s infinite; }
                .btn-playback { border-color: #28a745; color: #28a745; font-weight: bold; }
                .btn-paused { border-color: #ffb703; color: #ffb703; font-weight: bold; }
                @keyframes pulse { 0% { opacity: 0.6; } 50% { opacity: 1.0; } 100% { opacity: 0.6; } }

                .dpad-overlay { position: absolute; bottom: 5px; left: 50%; transform: translateX(-50%); display: grid; grid-template-columns: 75px 90px 75px; grid-template-rows: repeat(3, 65px); gap: 15px; pointer-events: none; z-index: 5; transition: opacity 0.25s ease; }
                .dpad-hidden { opacity: 0 !important; pointer-events: none !important; }
                
                .touch-btn { background: rgba(0, 255, 204, 0.15); color: #00ffcc; border: 2px solid rgba(0, 255, 204, 0.5); border-radius: 50%; font-size: 24px; display: flex; align-items: center; justify-content: center; cursor: pointer; width: 100%; height: 65px; pointer-events: auto; backdrop-filter: blur(3px); }
                .touch-btn:active { background: rgba(0, 255, 204, 0.65); color: white; border-color: #ffffff; }
                .side-btn { border-radius: 50%; width: 65px; height: 65px; justify-self: center; }
                .center-capsule { border-radius: 20px; }
                .center-space { grid-column: 2; grid-row: 2; background: rgba(255, 255, 255, 0.12); border-color: rgba(255, 255, 255, 0.35); font-size: 11px; font-weight: bold; color: #ccc; }
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
                        if (txt.includes("ACTIVE")) { autoBtn.classList.add("hud-btn-active-auto"); } 
                        else { autoBtn.classList.remove("hud-btn-active-auto"); }
                    });
                }

                function advanceMacroState() {
                    fetch('/advance_macro_mode').then(res => res.json()).then(data => {
                        let btn = document.getElementById("hud-macro-btn");
                        let pBtn = document.getElementById("hud-pause-btn");
                        document.getElementById("hud-mode-status").innerText = data.label.toUpperCase() + " MODE";
                        btn.className = "hud-btn " + data.css_class;
                        btn.innerText = data.button_text;
                        pBtn.style.display = data.show_pause ? "inline-block" : "none";
                        if(data.pause_text) pBtn.innerText = data.pause_text;
                    });
                }

                function togglePlayPause() {
                    fetch('/toggle_play_pause').then(res => res.json()).then(data => {
                        document.getElementById("hud-mode-status").innerText = data.label.toUpperCase() + " MODE";
                        document.getElementById("hud-pause-btn").innerText = data.pause_text;
                        document.getElementById("hud-macro-btn").className = "hud-btn " + data.css_class;
                    });
                }
                
                function toggleDpadVisibility() {
                    let dpad = document.getElementById("touch-dpad");
                    let btn = document.getElementById("hud-dpad-toggle");
                    if (dpad.classList.contains("dpad-hidden")) {
                        dpad.classList.remove("dpad-hidden");
                        btn.style.borderColor = "#00ffcc";
                        btn.style.color = "#00ffcc";
                    } else {
                        dpad.classList.add("dpad-hidden");
                        btn.style.borderColor = "rgba(255,255,255,0.3)";
                        btn.style.color = "#ffffff";
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
                
                function triggerFixedDegreeTurn(angle, dir) { fetch('/trigger_degree_turn?angle=' + angle + '&dir=' + dir); }
                function sendSingleAction(actionKey) { fetch('/keyboard_input?keys=' + actionKey); }
                function touchStartAction(actionKey) { keys.add(actionKey); sendState(); }
                function touchEndAction(actionKey) { keys.delete(actionKey); sendState(); }
                document.addEventListener('contextmenu', event => event.preventDefault());

                setInterval(() => {
                    fetch('/robot_stats')
                        .then(res => res.json())
                        .then(data => {
                            document.getElementById("hud-battery").innerText = data.battery.toFixed(1) + "%";
                            document.getElementById("hud-live-speed").innerText = data.live_speed.toFixed(1) + " mm/s";
                            document.getElementById("hud-target-speed").innerText = data.target_speed + " mm/s";
                        });
                }, 500);
            </script>
        </head>
        <body>
            <h2>Roomba Control & Immersive HUD Data Collector</h2>
            
            <div id="hud-root" class="video-hud-container">
                <img src="/video_feed" />
                
                <div class="hud-telemetry-box">
                    <div>🖥️ Mode: <span id="hud-mode-status" class="hud-status-text">MANUAL MODE</span></div>
                    <div>⚡ Battery: <span id="hud-battery" class="hud-battery-text">100.0%</span></div>
                    <div>🏎️ Live Speed: <span id="hud-live-speed" class="hud-speed-text">0.0 mm/s</span></div>
                    <div>🎯 Target Vel: <span id="hud-target-speed" style="color:#e2afff;">150 mm/s</span></div>
                </div>

                <div class="hud-controls-box">
                    <button id="hud-fs-btn" class="hud-btn" style="border-color:#a2d2ff; color:#a2d2ff;" onclick="toggleFullScreen()">📺 Full Screen</button>
                    <button id="hud-macro-btn" class="hud-btn btn-manual" onclick="advanceMacroState()">🎙️ Start Recording</button>
                    <button id="hud-pause-btn" class="hud-btn btn-paused" style="display:none;" onclick="togglePlayPause()">⏸️ Pause Playback</button>
                    
                    <button class="hud-btn hud-btn-degree" onclick="triggerFixedDegreeTurn(90, 'left')">📐 Turn Left 90°</button>
                    <button class="hud-btn hud-btn-degree" onclick="triggerFixedDegreeTurn(90, 'right')">📐 Turn Right 90°</button>
                    <button class="hud-btn hud-btn-degree" style="border-color:#00ffff; color:#00ffff;" onclick="triggerFixedDegreeTurn(180, 'left')">📐 Spin Around 180°</button>
                    
                    <button id="hud-auto-btn" class="hud-btn" onclick="toggleAutoMode()">Blue Line Follower</button>
                    <button id="hud-dpad-toggle" class="hud-btn" style="border-color:#00ffcc; color:#00ffcc;" onclick="toggleDpadVisibility()">🎮 Toggle HUD D-Pad</button>
                    
                    <button id="hud-vac-btn" class="hud-btn hud-btn-vac" onclick="sendSingleAction('v')">Vacuum: OFF</button>
                    <button class="hud-btn hud-btn-spd" onclick="sendSingleAction('e')">Speed +</button>
                    <button class="hud-btn hud-btn-spd" onclick="sendSingleAction('q')">Speed -</button>
                </div>
                
                <div id="touch-dpad" class="dpad-overlay">
                    <div class="touch-btn center-capsule" style="grid-column: 2; grid-row: 1;" ontouchstart="touchStartAction('w')" ontouchend="touchEndAction('w')" onmousedown="touchStartAction('w')" onmouseup="touchEndAction('w')">▲</div>
                    <div class="touch-btn side-btn" style="grid-column: 1; grid-row: 2;" ontouchstart="touchStartAction('a')" ontouchend="touchEndAction('a')" onmousedown="touchStartAction('a')" onmouseup="touchEndAction('a')">◀</div>
                    <div class="touch-btn center-space" style="grid-column: 2; grid-row: 2;" ontouchstart="touchStartAction('space')" ontouchend="touchEndAction('space')" onmousedown="touchStartAction('space')" onmouseup="touchEndAction('space')">STOP</div>
                    <div class="touch-btn side-btn" style="grid-column: 3; grid-row: 2;" ontouchstart="touchStartAction('d')" ontouchend="touchEndAction('d')" onmousedown="touchStartAction('d')" onmouseup="touchEndAction('d')">▶</div>
                    <div class="touch-btn center-capsule" style="grid-column: 2; grid-row: 3;" ontouchstart="touchStartAction('s')" ontouchend="touchEndAction('s')" onmousedown="touchStartAction('s')" onmouseup="touchEndAction('s')">▼</div>
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
    global auto_mode, current_action_label, operational_mode
    auto_mode = not auto_mode
    operational_mode = "manual"
    drive_roomba(0, 0)
    current_action_label = "stop"
    return "AUTO MODE ACTIVE" if auto_mode else "MANUAL MODE"

@app.route('/advance_macro_mode')
def advance_macro_mode():
    global operational_mode, movement_history, current_move_start, last_saved_keys, auto_mode, playback_index
    auto_mode = False
    
    if operational_mode in ["manual", "paused"]:
        operational_mode = "record"
        movement_history = []
        playback_index = 0
        last_saved_keys = set(['space'])
        current_move_start = time.time()
        return jsonify({"label": "recording", "css_class": "btn-record", "button_text": "⏹️ Stop & Playback", "show_pause": False})
        
    elif operational_mode == "record":
        if current_move_start is not None:
            duration = time.time() - current_move_start
            if duration > 0.05:
                movement_history.append((list(last_saved_keys), duration, current_speed))
                
        drive_roomba(0, 0)
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
        return jsonify({"label": "manual", "css_class": "btn-manual", "button_text": "🎙️ Start Recording", "show_pause": False})

@app.route('/toggle_play_pause')
def toggle_play_pause():
    global operational_mode
    if operational_mode == "playback":
        operational_mode = "paused"
        drive_roomba(0, 0)
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

@app.route('/robot_stats')
def robot_stats():
    return jsonify({
        "live_speed": abs(live_speed_mms),
        "target_speed": current_speed,
        "battery": battery_pct,
        "vacuum": vacuum_on
    })

@app.route('/keyboard_input')
def keyboard_input():
    global current_speed, vacuum_on, last_keys_set, auto_mode, current_action_label, operational_mode, current_move_start, last_saved_keys
    raw_keys = request.args.get('keys', '')
    current_keys = set(raw_keys.split('-')) if raw_keys else set()
    
    newly_pressed = current_keys - last_keys_set
    if 'q' in newly_pressed: current_speed = max(50, current_speed - 50)
    if 'e' in newly_pressed: current_speed = min(250, current_speed + 50)
    if 'v' in newly_pressed: 
        vacuum_on = not vacuum_on
        set_vacuum(vacuum_on)
        
    last_keys_set = current_keys

    if (operational_mode in ["playback", "paused"]) and len(current_keys) > 0 and 'space' not in current_keys:
        operational_mode = "manual"
        drive_roomba(0, 0)
        current_action_label = "stop"

    if not auto_mode and operational_mode != "playback" and operational_mode != "paused" and current_action_label != "precision_turning" and "encoder_turning" not in current_action_label:
        if operational_mode == "record":
            filtered_move_keys = current_keys.intersection({'w', 'a', 's', 'd', 'space'})
            if filtered_move_keys != last_saved_keys:
                now = time.time()
                if current_move_start is not None:
                    duration = now - current_move_start
                    if duration > 0.05:
                        movement_history.append((list(last_saved_keys), duration, current_speed))
                last_saved_keys = filtered_move_keys
                current_move_start = now

        if 'space' in current_keys: drive_roomba(0, 0); current_action_label = "stop"
        elif 'w' in current_keys: drive_roomba(current_speed, 32767); current_action_label = "forward"
        elif 's' in current_keys: drive_roomba(-current_speed, 32767); current_action_label = "backward"
        elif 'a' in current_keys: drive_roomba(current_speed, 1); current_action_label = "left"
        elif 'd' in current_keys: drive_roomba(current_speed, -1); current_action_label = "right"
        else: drive_roomba(0, 0); current_action_label = "stop"
        
    return "OK", 200

if __name__ == '__main__':
    threading.Thread(target=camera_and_logic_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
