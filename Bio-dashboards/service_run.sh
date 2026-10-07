#!/usr/bin/env bash
# Runs one dashboard, keeping the computer awake while it runs.
#   service_run.sh <dashboard-folder>       e.g.  service_run.sh polar_dashboard
# Used by the systemd services that ./biodash.py installs (and by launch.sh).
#
# Keep-awake: the dashboard runs under `systemd-inhibit`, which blocks suspend and idle
# sleep for as long as it's running (lid-close is left alone -- set BIODASH_INHIBIT_LID=1
# to block that too). If the system refuses the inhibitor (no permission), the dashboard
# still starts; a warning is printed instead. BIODASH_INHIBIT=0 turns it off.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
dir="${1:?usage: service_run.sh <dashboard-folder>}"
cd "$here/$dir" || { echo "No dashboard folder '$dir'"; exit 1; }

what="sleep:idle"
[ "${BIODASH_INHIBIT_LID:-0}" = "1" ] && what="sleep:idle:handle-lid-switch"

if [ "${BIODASH_INHIBIT:-1}" != "0" ] && command -v systemd-inhibit >/dev/null 2>&1 \
   && systemd-inhibit --what="$what" --mode=block --who=Bio-dash --why=test true >/dev/null 2>&1; then
    echo "☀ Keeping this computer awake while $dir runs (systemd-inhibit: $what)."
    exec systemd-inhibit --what="$what" --mode=block --who="Bio-dash" \
         --why="Recording live sensor data ($dir)" bash run_dashboard.sh
fi
[ "${BIODASH_INHIBIT:-1}" != "0" ] && echo "⚠ Couldn't take a sleep inhibitor (systemd-inhibit missing or not permitted); the computer may sleep."
exec bash run_dashboard.sh
