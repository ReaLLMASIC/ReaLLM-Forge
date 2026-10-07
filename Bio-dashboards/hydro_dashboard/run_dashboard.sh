#!/bin/bash

# Use the repo's .venv automatically if ./setup_env.sh has been run (no need to activate)
_VENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv"
[ -x "$_VENV/bin/python3" ] && export PATH="$_VENV/bin:$PATH"

cd "$(dirname "$0")"

echo "============================================="
echo "   Starting Hydro-dash (HidrateSpark)        "
echo "============================================="
# Recordings go to <Documents>/Bio-dash/HidrateSpark/<bottle>/<date>/ (one session per connection)
echo "💾 Recordings: $(python3 -c 'import bt_debug; print(bt_debug.dashboard_dir("HidrateSpark"))' 2>/dev/null || echo '~/Documents/Bio-dash/HidrateSpark')"
echo "ℹ  The bottle allows one connection at a time -- force-quit the HidrateSpark phone app."

echo "[BLE] Launching bottle worker (hydro_worker.py)..."
python3 hydro_worker.py &
WORKER_PID=$!
sleep 1.5

echo "[WEB] Launching dashboard (app.py)..."
python3 app.py &
WEB_PID=$!

cleanup() {
    echo -e "\n\n============================================="
    echo "   Shutting down Hydro-dash...               "
    echo "============================================="
    kill $WEB_PID 2>/dev/null
    kill $WORKER_PID 2>/dev/null
    echo "✅ All processes terminated successfully."
    exit 0
}
trap cleanup SIGINT SIGTERM

echo -e "\n✅ System is LIVE!"
echo -e "🔗 Open your browser to: http://localhost:5004"
echo -e "⏳ Press [Ctrl + C] to stop both services.\n"
wait
