import cv2
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
SAMPLE_SIZE = 3

# Navigation, Logging, & Stream Globals
current_speed = 200
vacuum_on = False
last_keys_set = set()
auto_mode = False
show_grayscale = False  
current_action_label = "stop"

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

def update_sensors_efficiently():
    """Queries Sensor Group Packet 0 (26 bytes total) to extract all telemetry at once."""
    global total_distance, current_heading, live_speed_mms, battery_pct
    bump_left, bump_right, wheel_drop = 0, 0, 0
    
    if not roomba:
        return 0, 0, 0

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
                calc_pct = (charge / capacity) * 100.0
                battery_pct = max(0.0, min(100.0, calc_pct))
                
            live_speed_mms = delta_dist / 0.055

    except Exception as e:
        print(f"Serial read telemetry error: {e}")
        
    return bump_left, bump_right, wheel_drop

# --- Autonomous & Video Processing Loop ---
def camera_and_logic_loop():
    global current_frame, auto_mode, show_grayscale, current_action_label, csv_file, SAMPLE_SIZE
    
    pipeline = (
        f"nvarguscamerasrc sensor-id=0 ! "
        f"video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, width=640, height=360, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )
    
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Camera failed to open.")
        return

    # Folder Setup
    session_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    output_dir = f"roomba_dataset_{session_time}"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"dataset_{SAMPLE_SIZE}x{SAMPLE_SIZE}.csv")
    csv_file = open(csv_path, 'w')
    
    total_pixels = SAMPLE_SIZE * SAMPLE_SIZE
    pixel_headers = ",".join([f"p{i}" for i in range(total_pixels)])
    csv_file.write(f"timestamp,action,bump_left,bump_right,wheel_dropped,total_distance_mm,heading_deg,speed_mm_s,battery_percent,{pixel_headers}\n")
    print(f"Saving dynamic {SAMPLE_SIZE}x{SAMPLE_SIZE} dataset inside folder: {output_dir}")

    # Behavior State Machine
    auto_state = "FORWARD"  
    state_end_time = 0.0
    chosen_spin_dir = 1
    
    # NEW: Obstacle side memory cash
    # Stores which bumper side initially triggered the backup phase
    collision_side = "none" 

    # Wheel Drop Damping Variables
    wheel_drop_started = None
    hard_safety_triggered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        b_left, b_right, w_drop = update_sensors_efficiently()

        # Dynamic Resizing & Flattening
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_frame = cv2.resize(gray, (SAMPLE_SIZE, SAMPLE_SIZE), interpolation=cv2.INTER_AREA)
        pixels = small_frame.flatten()
        pixel_str = ",".join(map(str, pixels))

        # Log Data securely to CSV rows
        synced_time = time.time()
        csv_file.write(f"{synced_time},{current_action_label},{b_left},{b_right},{w_drop},{total_distance:.1f},{current_heading:.1f},{live_speed_mms:.1f},{battery_pct:.1f},{pixel_str}\n")

        # --- WHEEL DROP CONFIRMATION FILTER ---
        if w_drop == 1:
            if wheel_drop_started is None:
                wheel_drop_started = synced_time
                print("⚠️ Warning: Wheel drop registered. Starting 5-second validation check...")
            if (synced_time - wheel_drop_started) >= 5.0:
                hard_safety_triggered = True
        else:
            if wheel_drop_started is not None:
                print("✅ Wheel drop cleared. Resuming normal operations.")
            wheel_drop_started = None
            hard_safety_triggered = False

        # --- SAFETY DEACTIVATION CODES ---
        if hard_safety_triggered:
            drive_roomba(0, 0)
            current_action_label = "stop"
            if auto_mode:
                print("🚨 CRITICAL OVERRIDE: Wheel drop sustained over 5 seconds! Disabling Auto Mode.")
                auto_mode = False

        # --- AUTO STATE ENGINE ---
        elif auto_mode:
            current_time = time.time()

            if auto_state == "FORWARD":
                if b_left == 1 or b_right == 1:
                    auto_state = "REVERSING"
                    state_end_time = current_time + 0.8  
                    current_action_label = "backward"
                    drive_roomba(-current_speed, 32767)
                    
                    # Lock in collision memory side context immediately on impact frame
                    if b_left == 1 and b_right == 1:
                        collision_side = "center"
                    elif b_left == 1:
                        collision_side = "left"
                    else:
                        collision_side = "right"
                    print(f"💥 Obstacle detected on the {collision_side.upper()}! Reversing...")
                else:
                    current_action_label = "forward"
                    drive_roomba(current_speed, 32767)

            elif auto_state == "REVERSING":
                if current_time >= state_end_time:
                    auto_state = "SPINNING"
                    state_end_time = current_time + 1.2  
                    
                    # INTELLIGENT SPIN DIRECTION CHOOSING
                    if collision_side == "left":
                        chosen_spin_dir = -1  # Spin Right (Clockwise) away from left wall
                    elif collision_side == "right":
                        chosen_spin_dir = 1   # Spin Left (Counter-Clockwise) away from right wall
                    else:
                        # Full frontal or dead-center hit: randomly spin clear
                        chosen_spin_dir = random.choice([1, -1])
                        
                    current_action_label = "spin"
                    drive_roomba(current_speed, chosen_spin_dir)
                else:
                    current_action_label = "backward"
                    drive_roomba(-current_speed, 32767)

            elif auto_state == "SPINNING":
                if current_time >= state_end_time:
                    auto_state = "FORWARD"
                    collision_side = "none" # Clear target memory
                    current_action_label = "forward"
                    drive_roomba(current_speed, 32767)
                else:
                    current_action_label = "spin"
                    drive_roomba(current_speed, chosen_spin_dir)
        else:
            auto_state = "FORWARD"

        with lock:
            if show_grayscale:
                current_frame = cv2.resize(small_frame, (640, 360), interpolation=cv2.INTER_NEAREST)
            else:
                current_frame = frame.copy()

# --- Flask Web Server Routes ---
@app.route('/')
def index():
    html = """
    <html>
        <head>
            <title>Roomba AI Data Center</title>
            <style>
                body { background-color: #1a1a1a; color: white; text-align: center; font-family: monospace; margin-top: 20px; }
                img { border: 2px solid #555; max-width: 100%; border-radius: 6px; }
                .status-panel { margin: 15px; font-size: 18px; color: #00ffcc; }
                .telemetry-grid { display: flex; justify-content: center; gap: 40px; margin: 15px auto; max-width: 600px; background: #26262b; padding: 10px; border-radius: 6px; border: 1px solid #444; }
                .telemetry-item { font-size: 16px; color: #ffb703; }
                .btn-container { display: flex; justify-content: center; gap: 15px; margin-top: 10px; }
                .instructions { color: #aaa; margin-top: 20px; font-size: 14px; line-height: 1.6; }
                .key { display: inline-block; background: #333; padding: 4px 8px; border-radius: 4px; margin: 0 4px; color: #fff; font-weight: bold;}
                button { padding: 10px 20px; font-size: 16px; background-color: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; font-family: monospace; font-weight: bold; }
                button:hover { background-color: #218838; }
                .btn-view { background-color: #007bff; }
                .btn-view:hover { background-color: #0069d9; }
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
                        document.getElementById("mode-status").innerText = txt; 
                    });
                }

                function toggleViewMode() {
                    fetch('/toggle_view').then(res => res.text()).then(txt => {
                        document.getElementById("view-btn").innerText = txt;
                    });
                }

                setInterval(() => {
                    fetch('/robot_stats')
                        .then(res => res.json())
                        .then(data => {
                            document.getElementById("ui-speed").innerText = data.speed.toFixed(1) + " mm/s";
                            document.getElementById("ui-battery").innerText = data.battery.toFixed(1) + "%";
                        });
                }, 1000);
            </script>
        </head>
        <body>
            <h2>Roomba Control & Dynamic Dataset Collector</h2>
            <img src="/video_feed" width="640" height="360" />
            
            <div class="status-panel">Control State: <span id="mode-status">MANUAL MODE</span></div>
            
            <div class="telemetry-grid">
                <div class="telemetry-item">⚡ Battery: <span id="ui-battery">100.0%</span></div>
                <div class="telemetry-item">🏎️ Live Speed: <span id="ui-speed">0.0 mm/s</span></div>
            </div>

            <div class="btn-container">
                <button onclick="toggleAutoMode()">Toggle AUTO / MANUAL Mode</button>
                <button id="view-btn" class="btn-view" onclick="toggleViewMode()">Switch to AI Matrix View</button>
            </div>
            
            <div class="instructions">
                <p><strong>Click screen window to focus before driving!</strong></p>
                <p><span class="key">W</span> / <span class="key">S</span> = Drive Manual | <span class="key">A</span> / <span class="key">D</span> = Spin In Place | <span class="key">SPACE</span> = Stop</p>
                <p><span class="key">Q</span> / <span class="key">E</span> = Speed | <span class="key">V</span> = Vacuum | <span class="key">X</span> = Force Stop Auto Mode</p>
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
    global auto_mode, current_action_label
    auto_mode = not auto_mode
    if not auto_mode:
        drive_roomba(0, 0)
        current_action_label = "stop"
        return "MANUAL MODE"
    return "AUTO MODE ACTIVE"

@app.route('/toggle_view')
def toggle_view():
    global show_grayscale
    show_grayscale = not show_grayscale
    if show_grayscale:
        return "Switch to Raw Video View"
    return "Switch to AI Matrix View"

@app.route('/robot_stats')
def robot_stats():
    return jsonify({
        "speed": abs(live_speed_mms),
        "battery": battery_pct
    })

@app.route('/keyboard_input')
def keyboard_input():
    global current_speed, vacuum_on, last_keys_set, auto_mode, current_action_label
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
        drive_roomba(0, 0)
        current_action_label = "stop"
        
    last_keys_set = current_keys

    if not auto_mode:
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
