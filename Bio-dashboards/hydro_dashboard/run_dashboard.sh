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
# If either process stops on its own, stop the other and exit with an error: a background
# service is then restarted, and a problem such as a missing package shows up as a failure
# instead of a dashboard that looks started but isn't there.
wait -n
code=$?
echo "❌ A dashboard process stopped unexpectedly (exit $code)."
echo "   If the lines above say \"No module named ...\", run ./setup_env.sh in the release folder (each release has its own .venv)."
kill $(jobs -p) 2>/dev/null
exit 1
