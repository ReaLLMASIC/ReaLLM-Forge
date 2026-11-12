#!/usr/bin/env bash
# Run Bio-dash in this terminal (Ctrl+C stops it). For a background service: ./biodash.py
#
#   ./launch.sh                 http://localhost:5000
#   ./launch.sh --port 8080
#   ./launch.sh --demo          made-up readings, no Bluetooth: try the app out
#
# Which devices it looks for is chosen in the app (first-run setup, or Settings).
cd "$(dirname "$0")" || exit 1
exec bash service_run.sh "$@"
