#!/bin/bash

# Use the repo's .venv automatically if ./setup_env.sh has been run (no need to activate)
_VENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv"
[ -x "$_VENV/bin/python3" ] && export PATH="$_VENV/bin:$PATH"

cd "$(dirname "$0")"

echo "============================================="
echo "   Starting Atmos Air Quality Lab            "
echo "============================================="

# Recordings go to <Documents>/Bio-dash/Atmos/<device>/<date>/ (one session per connection)
echo "💾 Recordings: $(python3 -c 'import bt_debug; print(bt_debug.dashboard_dir("Atmos"))' 2>/dev/null || echo '~/Documents/Bio-dash/Atmos')"

echo "[BLE] Launching hardware engine (sensor_worker.py)..."
python3 sensor_worker.py &
WORKER_PID=$!

sleep 1.5

echo "[WEB] Launching dashboard interface (app.py)..."
python3 app.py &
WEB_PID=$!

cleanup() {
    echo -e "\n\n============================================="
    echo "   Shutting down Air Lab...                  "
    echo "============================================="
    kill $WEB_PID 2>/dev/null
    kill $WORKER_PID 2>/dev/null
    echo "✅ All processes terminated successfully."
    exit 0
}

trap cleanup SIGINT SIGTERM

echo -e "\n✅ System is LIVE!"
echo -e "🔗 Open your browser to: http://localhost:5002"
echo -e "⏳ Press [Ctrl + C] to stop both services.\n"

wait
