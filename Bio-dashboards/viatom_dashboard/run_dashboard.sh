#!/bin/bash

# Use the repo's .venv automatically if ./setup_env.sh has been run (no need to activate)
_VENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv"
[ -x "$_VENV/bin/python3" ] && export PATH="$_VENV/bin:$PATH"

# Navigate to the script's directory to ensure relative paths stay intact
cd "$(dirname "$0")"

echo "============================================="
echo "   Starting Viatom Checkme O2 Ultra Dashboard "
echo "============================================="

# Recordings go to <Documents>/Bio-dash/Viatom O2/<device>/<date>/ (one session per connection)
echo "💾 Recordings: $(python3 -c 'import bt_debug; print(bt_debug.dashboard_dir("Viatom O2"))' 2>/dev/null || echo '~/Documents/Bio-dash/Viatom O2')"

# 1. Start the BLE Worker in the background
echo "[BLE] Launching biometric background worker..."
python3 ble_worker.py &
BLE_PID=$!

# Give the Bluetooth engine a brief moment to initialize before booting the web layer
sleep 1.5

# 2. Start the Flask Application in the foreground
echo "[WEB] Launching dashboard interface web server..."
python3 app.py &
WEB_PID=$!

# Trap Ctrl+C (SIGINT) and exit signals to kill both background jobs cleanly
cleanup() {
    echo -e "\n\n============================================="
    echo "   Shutting down Viatom Dashboard Engines...   "
    echo "============================================="
    
    echo "[WEB] Stopping web server (PID: $WEB_PID)..."
    kill $WEB_PID 2>/dev/null
    
    echo "[BLE] Stopping Bluetooth worker (PID: $BLE_PID)..."
    kill $BLE_PID 2>/dev/null
    
    echo "✅ All processes terminated successfully."
    exit 0
}

# Assign the cleanup function to handle termination traps
trap cleanup SIGINT SIGTERM

echo -e "🔗 Open your browser to: http://localhost:5003"

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
