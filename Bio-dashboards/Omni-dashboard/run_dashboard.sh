#!/bin/bash

# Use the repo's .venv automatically if ./setup_env.sh has been run (no need to activate)
_VENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv"
[ -x "$_VENV/bin/python3" ] && export PATH="$_VENV/bin:$PATH"

# Run from this script's folder so relative paths (omni.py, logs_omni/) work from anywhere
cd "$(dirname "$0")"

echo "========================================"
echo "    OMNI-DASH IGNITION SEQUENCE"
echo "========================================"

echo "[1/4] Hunting for lingering Python processes..."
# Force-kill any ghost threads still running the dashboard
pkill -9 -f "python3 omni.py" 2>/dev/null || echo "  -> No ghost Python processes found."

echo "[2/4] Forcing Port 5000 open..."
# Aggressively kill anything holding the Flask web port
fuser -k 5000/tcp 2>/dev/null || echo "  -> Port 5000 is clean."

echo "[3/4] Flushing Linux Bluetooth Cache..."
# This prevents the DBus "ghost locks" so you never have to reboot the Orin again!
# (It will ask for your Orin password here)
# Clears stale BlueZ state. Needs a password, so it's skipped when running as a
# service (no terminal to ask on) -- the service manager already starts us fresh.
if [ -z "${BIODASH_SERVICE:-}" ]; then
    sudo systemctl restart bluetooth
elif sudo -n true 2>/dev/null; then
    sudo -n systemctl restart bluetooth
else
    echo "  -> Running as a service: skipping the Bluetooth restart (needs sudo)."
fi
sleep 2 # Give the BlueZ daemon a moment to wake back up

# Recordings go to <Documents>/Bio-dash/Omni/<device>/<date>/ (one session per connection)
echo "💾 Recordings: $(python3 -c 'import bt_debug; print(bt_debug.dashboard_dir("Omni"))' 2>/dev/null || echo '~/Documents/Bio-dash/Omni')"

echo "[4/4] FIRING MAIN ENGINE..."
echo "========================================"
python3 omni.py &
OMNI_PID=$!

# Stop omni.py when this script is stopped (Ctrl+C here, or ./launch.sh stopping it)
cleanup() {
    echo -e "\n🛑 Stopping Omni (PID: $OMNI_PID)..."
    kill $OMNI_PID 2>/dev/null
    wait $OMNI_PID 2>/dev/null
    echo "✅ Omni stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM

echo -e "🔗 Open your browser to: http://localhost:5000"
# If either process stops on its own, stop the other and exit with an error: a background
# service is then restarted, and a problem such as a missing package shows up as a failure
# instead of a dashboard that looks started but isn't there.
wait -n
code=$?
echo "❌ A dashboard process stopped unexpectedly (exit $code)."
echo "   If the lines above say \"No module named ...\", run ./setup_env.sh in the release folder (each release has its own .venv)."
kill $(jobs -p) 2>/dev/null
exit 1
