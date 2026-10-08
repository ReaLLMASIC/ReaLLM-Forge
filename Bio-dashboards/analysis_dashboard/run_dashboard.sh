#!/bin/bash
# Bio-dash Analysis: summaries, every signal on one clock, CSV export, overnight reports.
# Reads the recordings folder only (no Bluetooth). Port 5005.

# Use the repo's .venv automatically if ./setup_env.sh has been run (no need to activate)
_VENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv"
[ -x "$_VENV/bin/python3" ] && export PATH="$_VENV/bin:$PATH"

cd "$(dirname "$0")"

echo "============================================="
echo "   Starting Bio-dash Analysis               "
echo "============================================="
echo -e "🔗 Open your browser to: http://localhost:5005\n"

# One process: if it stops on its own, the exit code says so (a background service is restarted).
python3 app.py
code=$?
if [ $code -ne 0 ]; then
    echo "❌ The analysis dashboard stopped unexpectedly (exit $code)."
    echo "   If the lines above say \"No module named ...\", run ./setup_env.sh in the release folder."
    exit 1
fi
