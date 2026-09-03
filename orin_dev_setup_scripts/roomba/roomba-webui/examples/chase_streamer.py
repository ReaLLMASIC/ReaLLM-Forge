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
current_speed = 200
vacuum_on = False
last_keys_set = set()
auto_mode = False
show_grayscale = False  
current_action_label = "stop"

# --- NEW: CHASE MODE GLOBALS ---
chase_mode = False

# Accumulated Odometry Tracking
total_distance = 0.0  
current_heading = 0.0  

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
        print("📥 Passive Docking aborted. Safe Mode serial control restored.")

def update_sensors_efficiently():
    global total_distance, current_heading, live_speed_mms, battery_pct
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
            delta_angle = int.from_bytes(data[14:16], byteorder='big', signed=True)
            current_heading = (current_heading + delta_angle) % 360
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
    
    y_safe = int(h * 0.62)
    w_safe_half = int(w * 0.10)
    cv2.line(img, (w//2 - w_safe_half, y_safe), (w//2 + w_safe_half, y_safe), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "SAFE", (w//2 - 95, y_safe - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    
    y_caution = int(h * 0.78)
    w_caution_half = int(w * 0.18)
    cv2.line(img, (w//2 - w_caution_half, y_caution), (w//2 + w_caution_half, y_caution), (0, 165, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "CAUTION", (w//2 + 55, y_caution - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
    
    y_stop = int(h * 0.90)
    w_stop_half = int(w * 0.26)
    cv2.line(img, (w//2 - w_stop_half, y_stop), (w//2 + w_stop_half, y_stop), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(img, "STOP", (w//2 - 110, y_stop - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    
    cv2.drawMarker(img, (w//2, int(h * 0.52)), (0, 255, 204), cv2.MARKER_CROSS, 15, 1, cv2.LINE_AA)
    return img

# --- Autonomous, Chasing, & Video Loop ---
def camera_and_logic_loop():
    global current_frame, auto_mode, chase_mode, show_grayscale, current_action_label, csv_file, SAMPLE_SIZE
    
    pipeline = (
        f"nvarguscamerasrc sensor-id=0 ! "
        f"video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, width=640, height=360, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )
    
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened(): return

    session_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    output_dir = f"roomba_dataset_{session_time}"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"dataset_{SAMPLE_SIZE}x{SAMPLE_SIZE}.csv")
    csv_file = open(csv_path, 'w')
    
    total_pixels = SAMPLE_SIZE * SAMPLE_SIZE
    pixel_headers = ",".join([f"p{i}" for i in range(total_pixels)])
    csv_file.write(f"timestamp,action,bump_left,bump_right,wheel_dropped,total_distance_mm,heading_deg,speed_mm_s,battery_percent,{pixel_headers}\n")

    auto_state = "FORWARD"  
    state_end_time = 0.0
    chosen_spin_dir = 1
    collision_side = "none" 
    wheel_drop_started = None
    hard_safety_triggered = False

    # Pink Tracking Mask Boundaries (HSV Space Calibration)
    pink_lower = np.array([140, 70, 70])
    pink_upper = np.array([175, 255, 255])

    while True:
        ret, frame = cap.read()
        if not ret: break

        b_left, b_right, w_drop = update_sensors_efficiently()

        # Generate Downsampled Matrix Data Array
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_frame = cv2.resize(gray, (SAMPLE_SIZE, SAMPLE_SIZE), interpolation=cv2.INTER_AREA)
        pixels = small_frame.flatten()
        pixel_str = ",".join(map(str, pixels))

        synced_time = time.time()
        csv_file.write(f"{synced_time},{current_action_label},{b_left},{b_right},{w_drop},{total_distance:.1f},{current_heading:.1f},{live_speed_mms:.1f},{battery_pct:.1f},{pixel_str}\n")

        # Wheel drop safety processing
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
            chase_mode = False

        # --- MODE 1: CHASE MODE LOGIC ---
        elif chase_mode:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, pink_lower, pink_upper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                
                if radius > 10:  # Valid target scale verification filter
                    # Draw visual target lock tracking context layout on frames
                    cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 255), 2)
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    
                    # Compute Steering Correction Value (Target Center Offset)
                    screen_center_x = frame.shape[1] / 2
                    steering_error = x - screen_center_x
                    
                    # Convert pixel offset error directly to an Open Interface radius value
                    if abs(steering_error) < 20:
                        oi_radius = 32767 # Dead straight ahead ahead trajectory
                    elif steering_error < 0:
                        oi_radius = max(1, int(300 + steering_error * 2)) # Dynamic left turn
                    else:
                        oi_radius = min(-1, int(-300 + steering_error * 2)) # Dynamic right turn
                        
                    # Compute Velocity Profile based on Ball Diameter Distance
                    # Target radius of 65px means the robot stops roughly ~20cm away from the ball
                    if radius < 55:
                        drive_roomba(160, oi_radius) # Drive forward to close range gap
                        current_action_label = "chase_forward"
                    elif radius > 75:
                        drive_roomba(-130, oi_radius) # Backup if the target rolls too close
                        current_action_label = "chase_backward"
                    else:
                        drive_roomba(0, 0) # Maintain distance threshold locked
                        current_action_label = "chase_hold"
                else:
                    drive_roomba(0, 0)
                    current_action_label = "chase_search"
            else:
                # If target tracking is dropped completely, spin in place slowly to scan room context
                drive_roomba(100, 1)
                current_action_label = "chase_spin_scan"

        # --- MODE 2: WALL EVASION AUTO MODE ---
        elif auto_mode:
            current_time = time.time()
            if auto_state == "FORWARD":
                if b_left == 1 or b_right == 1:
                    auto_state = "REVERSING"
                    state_end_time = current_time + 0.8  
                    current_action_label = "backward"
                    drive_roomba(-current_speed, 32767)
                    collision_side = "center" if (b_left==1 and b_right==1) else "left" if b_left==1 else "right"
                else:
                    current_action_label = "forward"
                    drive_roomba(current_speed, 32767)
            elif auto_state == "REVERSING":
                if current_time >= state_end_time:
                    auto_state = "SPINNING"
                    state_end_time = current_time + 1.2  
                    chosen_spin_dir = -1 if collision_side == "left" else 1 if collision_side == "right" else random.choice([1, -1])
                    current_action_label = "spin"
                    drive_roomba(current_speed, chosen_spin_dir)
                else:
                    drive_roomba(-current_speed, 32767)
            elif auto_state == "SPINNING":
                if current_time >= state_end_time:
                    auto_state = "FORWARD"
                    collision_side = "none" 
                    current_action_label = "forward"
                    drive_roomba(current_speed, 32767)
                else:
                    drive_roomba(current_speed, chosen_spin_dir)
        elif current_action_label != "docking":
            auto_state = "FORWARD"

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

                .hud-telemetry-box { position: absolute; top: 15px; left: 15px; background: rgba(38, 38, 43, 0.75); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 6px; padding: 10px 14px; text-align: left; font-size: 13px; backdrop-filter: blur(4px); -webkit-backdrop-filter: blur(4px); pointer-events: none; z-index: 10; }
                .hud-telemetry-box div { margin: 5px 0; }
                .hud-status-text { color: #00ffcc; font-weight: bold; }
                .hud-battery-text { color: #ffb703; }
                .hud-speed-text { color: #a2d2ff; }
                .hud-vac-off { color: #ff6b6b; font-weight: bold; }
                .hud-vac-on { color: #28a745; font-weight: bold; }

                .hud-controls-box { position: absolute; top: 15px; right: 15px; display: flex; flex-direction: column; gap: 8px; pointer-events: auto; z-index: 10; }
                .hud-btn { padding: 8px 12px; font-size: 11px; background: rgba(0, 0, 0, 0.6); color: white; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 4px; cursor: pointer; font-family: monospace; font-weight: bold; backdrop-filter: blur(2px); -webkit-backdrop-filter: blur(2px); }
                .hud-btn:hover { background: rgba(255, 255, 255, 0.2); }
                .hud-btn-active-auto { border-color: #28a745; color: #28a745; }
                .hud-btn-active-chase { border-color: #ff00ff; color: #ff00ff; }
                .hud-btn-dock { border-color: #ffb703; color: #ffb703; }
                .hud-btn-cancel-dock { border-color: #dc3545; color: #dc3545; display: none; }
                
                .hud-btn-vac { border-color: #ff6b6b; color: #ff6b6b; }
                .hud-btn-vac.active { border-color: #28a745; color: #28a745; }
                .hud-btn-spd { border-color: #a2d2ff; color: #a2d2ff; }

                .dpad-overlay { position: absolute; bottom: 5px; left: 50%; transform: translateX(-50%); display: grid; grid-template-columns: 75px 90px 75px; grid-template-rows: repeat(3, 65px); gap: 15px; pointer-events: none; z-index: 5; transition: opacity 0.2s ease-in-out; }
                .touch-btn { background: rgba(0, 0, 0, 0.55); color: #00ffcc; border: 2px solid rgba(0, 255, 204, 0.5); border-radius: 50%; font-size: 24px; display: flex; align-items: center; justify-content: center; cursor: pointer; width: 100%; height: 65px; pointer-events: auto; backdrop-filter: blur(3px); -webkit-backdrop-filter: blur(3px); }
                .touch-btn:active { background: rgba(0, 0, 0, 0.55); color: white; border-color: #ffffff; }
                
                .side-btn { border-radius: 50%; width: 65px; height: 65px; justify-self: center; }
                .center-capsule { border-radius: 20px; }
                .center-space { grid-column: 2; grid-row: 2; background: rgba(255, 255, 255, 0.12); border-color: rgba(255, 255, 255, 0.35); font-size: 11px; font-weight: bold; color: #ccc; }
                .center-space:active { background: rgba(255, 255, 255, 0.35); color: white; }
            </style>
            <script>
                let keys = new Set();
                const validKeys = ['w','s','a','d','q','e','v','x','space'];
                
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
                        document.getElementById("hud-mode-status").innerText = txt; 
                        let autoBtn = document.getElementById("hud-auto-btn");
                        let chaseBtn = document.getElementById("hud-chase-btn");
                        chaseBtn.classList.remove("hud-btn-active-chase");
                        if (txt.includes("ACTIVE")) { autoBtn.classList.add("hud-btn-active-auto"); } 
                        else { autoBtn.classList.remove("hud-btn-active-auto"); }
                        document.getElementById("hud-dock-btn").style.display = "inline-block";
                        document.getElementById("hud-cancel-dock-btn").style.display = "none";
                    });
                }

                function toggleChaseMode() {
                    fetch('/toggle_chase').then(res => res.text()).then(txt => {
                        document.getElementById("hud-mode-status").innerText = txt;
                        let autoBtn = document.getElementById("hud-auto-btn");
                        let chaseBtn = document.getElementById("hud-chase-btn");
                        autoBtn.classList.remove("hud-btn-active-auto");
                        if (txt.includes("CHASE")) { chaseBtn.classList.add("hud-btn-active-chase"); }
                        else { chaseBtn.classList.remove("hud-btn-active-chase"); }
                        document.getElementById("hud-dock-btn").style.display = "inline-block";
                        document.getElementById("hud-cancel-dock-btn").style.display = "none";
                    });
                }

                function toggleViewMode() {
                    fetch('/toggle_view').then(res => res.text()).then(txt => {
                        document.getElementById("view-btn").innerText = txt;
                        let dpad = document.getElementById("touch-dpad");
                        if (txt.includes("Matrix")) { dpad.style.opacity = "1"; dpad.style.pointerEvents = "auto"; } 
                        else { dpad.style.opacity = "0"; dpad.style.pointerEvents = "none"; }
                    });
                }

                function toggleFullScreen() {
                    let container = document.getElementById("hud-root");
                    if (!document.fullscreenElement) { container.requestFullscreen(); } 
                    else { document.exitFullscreen(); }
                }

                function triggerDocking() {
                    fetch('/dock').then(res => res.text()).then(txt => {
                        document.getElementById("hud-mode-status").innerText = txt;
                        document.getElementById("hud-auto-btn").classList.remove("hud-btn-active-auto");
                        document.getElementById("hud-chase-btn").classList.remove("hud-btn-active-chase");
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
                            
                            let vacLabel = document.getElementById("hud-vacuum");
                            let hudVacBtn = document.getElementById("hud-vac-btn");
                            if (data.vacuum) {
                                vacLabel.innerText = "ACTIVE"; vacLabel.className = "hud-vac-on";
                                hudVacBtn.classList.add("active"); hudVacBtn.innerText = "Vacuum: ON";
                            } else {
                                vacLabel.innerText = "OFF"; vacLabel.className = "hud-vac-off";
                                hudVacBtn.classList.remove("active"); hudVacBtn.innerText = "Vacuum: OFF";
                            }
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
                    <div>🎯 Target Vel: <span id="hud-target-speed" style="color:#e2afff;">200 mm/s</span></div>
                    <div>🌪️ Vacuum: <span id="hud-vacuum" class="hud-vac-off">OFF</span></div>
                </div>

                <div class="hud-controls-box">
                    <button id="hud-fs-btn" class="hud-btn" onclick="toggleFullScreen()">📺 Full Screen</button>
                    <button id="view-btn" class="hud-btn" onclick="toggleViewMode()">AI Matrix View</button>
                    <button id="hud-auto-btn" class="hud-btn" onclick="toggleAutoMode()">Auto Mode</button>
                    <button id="hud-chase-btn" class="hud-btn" onclick="toggleChaseMode()">Chase Mode</button>
                    <button id="hud-dock-btn" class="hud-btn hud-btn-dock" onclick="triggerDocking()">Seek Base</button>
                    <button id="hud-cancel-dock-btn" class="hud-btn hud-btn-cancel-dock" onclick="triggerCancelDocking()">Cancel Dock</button>
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
    global auto_mode, chase_mode, current_action_label
    auto_mode = not auto_mode
    chase_mode = False
    if not auto_mode:
        drive_roomba(0, 0)
        current_action_label = "stop"
        return "MANUAL MODE"
    return "AUTO MODE ACTIVE"

@app.route('/toggle_chase')
def toggle_chase():
    global auto_mode, chase_mode, current_action_label
    chase_mode = not chase_mode
    auto_mode = False
    if not chase_mode:
        drive_roomba(0, 0)
        current_action_label = "stop"
        return "MANUAL MODE"
    return "CHASE MODE ACTIVE"

@app.route('/toggle_view')
def toggle_view():
    global show_grayscale
    show_grayscale = not show_grayscale
    if show_grayscale: return "Switch to Raw Video View"
    return "Switch to AI Matrix View"

@app.route('/dock')
def dock():
    global auto_mode, chase_mode, current_action_label, vacuum_on
    auto_mode = False
    chase_mode = False
    vacuum_on = False 
    current_action_label = "docking"
    dock_roomba()
    return "SEEKING HOME DOCK..."

@app.route('/cancel_dock')
def cancel_dock():
    global current_action_label
    current_action_label = "stop"
    cancel_docking_sequence()
    return "MANUAL CONTROL RESTORED"

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
    global current_speed, vacuum_on, last_keys_set, auto_mode, chase_mode, current_action_label
    raw_keys = request.args.get('keys', '')
    current_keys = set(raw_keys.split('-')) if raw_keys else set()
    
    newly_pressed = current_keys - last_keys_set
    if 'q' in newly_pressed: current_speed = max(50, current_speed - 50)
    if 'e' in newly_pressed: current_speed = min(500, current_speed + 50)
    if 'v' in newly_pressed: 
        vacuum_on = not vacuum_on
        set_vacuum(vacuum_on)
    if 'x' in newly_pressed:
        auto_mode = False
        chase_mode = False
        drive_roomba(0, 0)
        current_action_label = "stop"
        
    last_keys_set = current_keys

    if not auto_mode and not chase_mode and current_action_label != "docking":
        if 'space' in current_keys: drive_roomba(0, 0); current_action_label = "stop"
        elif 'w' in current_keys: drive_roomba(current_speed, 32767); current_action_label = "forward"
        elif 's' in current_keys: drive_roomba(-current_speed, 32767); current_action_label = "backward"
        elif 'a' in current_keys: drive_roomba(current_speed, 1); current_action_label = "left"
        elif 'd' in current_keys: drive_roomba(current_speed, -1); current_action_label = "right"
        else: drive_roomba(0, 0); current_action_label = "stop"
        
    elif (auto_mode or chase_mode or current_action_label == "docking") and len(current_keys) > 0 and 'space' not in current_keys:
        auto_mode = False
        chase_mode = False
        current_action_label = "stop"
        cancel_docking_sequence()
        
    return "OK", 200

if __name__ == '__main__':
    threading.Thread(target=camera_and_logic_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
