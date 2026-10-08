#!/usr/bin/env bash
# Start several Bio-dash dashboards from one terminal. Ctrl+C stops them all.
#
#   ./launch.sh                    polar + viatom + atmos (the default)
#   ./launch.sh polar atmos        just these
#   ./launch.sh omni atmos         Omni (Polar + O2 together) plus Atmos
#
# Dashboards and ports:  omni 5000 · polar 5001 · atmos 5002 · viatom 5003 · hydro 5004 · analysis 5005
# Each dashboard's header links to the others, with a dot showing which are running.
set -u
cd "$(dirname "$0")"

# ☀ Keep the computer awake for as long as the dashboards run: re-run this script once
# under systemd-inhibit (blocks suspend + idle sleep; BIODASH_INHIBIT_LID=1 also blocks
# lid-close, BIODASH_INHIBIT=0 turns it off). If it isn't permitted we just carry on.
if [ -z "${BIODASH_INHIBITED:-}" ] && [ "${BIODASH_INHIBIT:-1}" != "0" ] && command -v systemd-inhibit >/dev/null 2>&1; then
    what="sleep:idle"; [ "${BIODASH_INHIBIT_LID:-0}" = "1" ] && what="sleep:idle:handle-lid-switch"
    if systemd-inhibit --what="$what" --mode=block --who=Bio-dash --why=test true >/dev/null 2>&1; then
        echo "☀ Keeping this computer awake while the dashboards run ($what)."
        BIODASH_INHIBITED=1 exec systemd-inhibit --what="$what" --mode=block --who="Bio-dash" \
            --why="Recording live sensor data" "$0" "$@"
    fi
    echo "⚠ Couldn't take a sleep inhibitor; the computer may sleep while the dashboards run."
fi

declare -A DIR=( [polar]=polar_dashboard [viatom]=viatom_dashboard [atmos]=atmos_dashboard [omni]=Omni-dashboard [hydro]=hydro_dashboard [analysis]=analysis_dashboard )
declare -A PORT=( [polar]=5001 [viatom]=5003 [atmos]=5002 [omni]=5000 [hydro]=5004 [analysis]=5005 )

wanted=("$@")
[ ${#wanted[@]} -eq 0 ] && wanted=(polar viatom atmos)

for d in "${wanted[@]}"; do
    [ -n "${DIR[$d]:-}" ] || { echo "Unknown dashboard '$d'. Choose from: polar viatom atmos omni hydro analysis"; exit 1; }
done

# Omni drives the Polar and the O2 itself -- running the standalone dashboards too would
# have two programs fighting over the same devices.
if [[ " ${wanted[*]} " == *" omni "* ]] && { [[ " ${wanted[*]} " == *" polar "* ]] || [[ " ${wanted[*]} " == *" viatom "* ]]; }; then
    echo "Omni already runs the Polar H10 and the O2; launch it without 'polar' / 'viatom'."
    exit 1
fi

# Omni's launcher restarts the Bluetooth service (sudo). Ask for the password now,
# and start Omni first so the restart can't cut the other dashboards' connections.
order=()
if [[ " ${wanted[*]} " == *" omni "* ]]; then
    sudo -v || exit 1
    order+=(omni)
fi
for d in "${wanted[@]}"; do [ "$d" != omni ] && order+=("$d"); done

pids=()
stop_all() {
    echo -e "\n🛑 Stopping: ${order[*]}"
    for p in "${pids[@]}"; do kill -TERM "$p" 2>/dev/null; done
    wait 2>/dev/null
    echo "✅ All dashboards stopped."
    exit 0
}
trap stop_all INT TERM

for d in "${order[@]}"; do
    # each dashboard's own launcher, output prefixed with its name
    bash "${DIR[$d]}/run_dashboard.sh" > >(sed -u "s/^/[$d] /") 2>&1 &
    pids+=($!)
    [ "$d" = omni ] && sleep 6 || sleep 1     # let Omni's Bluetooth restart settle
done

sleep 3
echo
echo "================ Bio-dash ================"
for d in "${order[@]}"; do printf "  %-7s http://localhost:%s\n" "$d" "${PORT[$d]}"; done
echo "=========================================="
echo "Ctrl+C stops everything."
wait
