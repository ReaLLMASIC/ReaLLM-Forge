#!/bin/bash

# Use the repo's .venv automatically if ./setup_env.sh has been run (no need to activate)
_VENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv"
[ -x "$_VENV/bin/python3" ] && export PATH="$_VENV/bin:$PATH"

# Navigate to the script's directory to ensure relative paths stay intact
cd "$(dirname "$0")"

echo "============================================="
echo "   Starting Polar H10 Biometric Lab          "
echo "============================================="

# 1. The Archive System: Move old logs to prevent UI ghosting, but keep the data!
# Recordings go to <Documents>/Bio-dash/Polar H10/<device>/<date>/ (one session per connection)
echo "💾 Recordings: $(python3 -c 'import bt_debug; print(bt_debug.dashboard_dir("Polar H10"))' 2>/dev/null || echo '~/Documents/Bio-dash/Polar H10')"

# 2. Start the Hardware Engine in the background
echo "[BLE] Launching hardware engine (advanced_worker.py)..."
python3 advanced_worker.py &
WORKER_PID=$!

# Give the Bluetooth engine a brief moment to initialize before booting the web layer
sleep 1.5

# 3. Start the Web Server in the background
echo "[WEB] Launching dashboard interface web server (app.py)..."
python3 app.py &
WEB_PID=$!

# Trap Ctrl+C (SIGINT) and exit signals to kill both background jobs cleanly
cleanup() {
    echo -e "\n\n============================================="
    echo "   Shutting down Polar H10 Lab...            "
    echo "============================================="
    
    echo "[WEB] Stopping web server (PID: $WEB_PID)..."
    kill $WEB_PID 2>/dev/null
    
    echo "[BLE] Stopping hardware engine (PID: $WORKER_PID)..."
    kill $WORKER_PID 2>/dev/null
    
    echo "✅ All processes terminated successfully."
    exit 0
}

# Assign the cleanup function to handle termination traps
trap cleanup SIGINT SIGTERM

echo -e "\n✅ System is LIVE!"
echo -e "🔗 Open your browser to: http://localhost:5001"
echo -e "⏳ Press [Ctrl + C] to stop both services.\n"

# Keep the script alive so it continues to intercept the trap signals
# If either process stops on its own, stop the other and exit with an error: a background
# service is then restarted, and a problem such as a missing package shows up as a failure
# instead of a dashboard that looks started but isn't there.
wait -n
code=$?
echo "❌ A dashboard process stopped unexpectedly (exit $code)."
echo "   If the lines above say \"No module named ...\", run ./setup_env.sh in the release folder (each release has its own .venv)."
kill $(jobs -p) 2>/dev/null
exit 1
