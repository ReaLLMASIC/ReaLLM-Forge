#!/usr/bin/env bash
# Runs Bio-dash (server.py) with this release's .venv if it has one.
# Used by the service ./biodash.py installs, and by ./launch.sh.
# server.py keeps the computer awake while it runs (BIODASH_INHIBIT=0 turns that off,
# BIODASH_INHIBIT_LID=1 also blocks lid-close sleep).
cd "$(dirname "$0")" || exit 1
PY=python3
[ -x .venv/bin/python3 ] && PY=.venv/bin/python3
exec "$PY" server.py "$@"
