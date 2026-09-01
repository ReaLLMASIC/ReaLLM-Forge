#!/bin/bash

# Ensure the script is run with sudo
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root: sudo $0"
  exit 1
fi

REBOOT_REQUIRED=0 # Session memory for major changes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Dependency Check ---
check_deps() {
    local missing=()
    for cmd in whiptail nmcli rfkill; do
        command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
    done
    if [ ${#missing[@]} -ne 0 ]; then
        echo "Missing required command(s): ${missing[*]}"
        echo "Install them with: sudo apt install whiptail network-manager rfkill"
        exit 1
    fi
}
check_deps

# --- Backup Helper ---
backup_once() {
    # $1 = file to back up, once, as <file>.bak (won't clobber a backup
    # already made manually via the "Backup Configs" menu option).
    local f="$1"
    [ -f "$f" ] && [ ! -f "${f}.bak" ] && cp "$f" "${f}.bak"
}

# --- System & Identity Functions ---

change_hostname() {
    CURRENT_HOST=$(hostname)
    NEW_HOST=$(whiptail --title "System Options" --inputbox "Enter new hostname:" 8 45 "$CURRENT_HOST" 3>&1 1>&2 2>&3)
    if [ $? -eq 0 ] && [ ! -z "$NEW_HOST" ]; then
        hostnamectl set-hostname "$NEW_HOST"
        # Word-boundary match so this only replaces the hostname token itself,
        # not any substring match elsewhere in /etc/hosts.
        sed -i "s/\<$CURRENT_HOST\>/$NEW_HOST/g" /etc/hosts
        whiptail --msgbox "Hostname changed to $NEW_HOST." 8 45
        REBOOT_REQUIRED=1
    fi
}

manage_wifi() {
    whiptail --infobox "Scanning for Wi-Fi networks..." 8 45
    WIFI_LIST=$(nmcli -t -f SSID dev wifi list | grep -v '^--' | head -n 10)
    mapfile -t WIFI_ARR <<< "$WIFI_LIST"
    MENU_ARR=()
    for ssid in "${WIFI_ARR[@]}"; do MENU_ARR+=("$ssid" ""); done
    CHOSEN_SSID=$(whiptail --title "Wi-Fi Setup" --menu "Select a network:" 15 60 8 "${MENU_ARR[@]}" 3>&1 1>&2 2>&3)
    if [ ! -z "$CHOSEN_SSID" ]; then
        PASS=$(whiptail --title "Wi-Fi Setup" --passwordbox "Enter password for $CHOSEN_SSID:" 8 45 3>&1 1>&2 2>&3)
        [ $? -eq 0 ] && clear && nmcli dev wifi connect "$CHOSEN_SSID" password "$PASS" && read -p "Done. Press Enter..."
    fi
}

toggle_bluetooth() {
    if rfkill list bluetooth | grep -q "yes"; then
        rfkill unblock bluetooth && whiptail --msgbox "Bluetooth ENABLED." 8 45
    else
        rfkill block bluetooth && whiptail --msgbox "Bluetooth DISABLED." 8 45
    fi
}

system_options_menu() {
    SYS_CHOICE=$(whiptail --title "System Options" --menu "Configure Identity & Connectivity" 18 65 5 \
    "S1" "Change Hostname" \
    "S2" "Configure Wi-Fi" \
    "S3" "Toggle Bluetooth" \
    "S4" "Back to Main Menu" 3>&1 1>&2 2>&3)
    case $SYS_CHOICE in
        S1) change_hostname ;;
        S2) manage_wifi ;;
        S3) toggle_bluetooth ;;
    esac
}

# --- Core Hardware Functions ---

ensure_auto_profile() {
    if ! grep -q "FAN_PROFILE auto" /etc/nvfancontrol.conf; then
        # A genuinely distinct middle-ground curve between "cool" (full speed
        # very early) and "quiet" (stays low until margin 40). Tune to taste.
        sed -i '/THERMAL_GROUP 0 {/i \
        FAN_PROFILE auto {\
                0       0       200     5000\
                30      0       200     5000\
                45      0       140     3500\
                55      0       90      2200\
                65      0       40      1200\
                105     0       0       0\
        }' /etc/nvfancontrol.conf
    fi
}

toggle_clocks_service() {
    # Uses the same unit name as orin_dev_setup_scripts/06-jetson-clocks-service.sh
    # so this toggle recognizes and controls that service instead of creating a duplicate.
    if systemctl is-active --quiet jetson-clocks; then
        systemctl disable jetson-clocks --now && whiptail --msgbox "jetson-clocks DISABLED." 8 45
    else
        [ ! -f /etc/systemd/system/jetson-clocks.service ] && cat <<EOF > /etc/systemd/system/jetson-clocks.service
[Unit]
Description=Maximize Jetson Performance
After=nvpmodel.service
[Service]
Type=oneshot
ExecStart=/usr/sbin/nvpmodel -m 2
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
EOF
        systemctl daemon-reload && systemctl enable jetson-clocks --now && whiptail --msgbox "jetson-clocks ENABLED." 8 45
    fi
}

clocks_status_label() {
    if systemctl is-active --quiet jetson-clocks 2>/dev/null; then
        echo "Toggle jetson-clocks Performance (ENABLED)"
    else
        echo "Toggle jetson-clocks Performance (DISABLED)"
    fi
}

power_mode_label() {
    local mode_line
    mode_line=$(nvpmodel -q 2>/dev/null | head -n1)
    if [ -n "$mode_line" ]; then
        echo "Set Power Mode (nvpmodel) [current: $mode_line]"
    else
        echo "Set Power Mode (nvpmodel)"
    fi
}

fan_mode_label() {
    local mode
    mode=$(grep -m1 '^[[:space:]]*FAN_DEFAULT_PROFILE' /etc/nvfancontrol.conf 2>/dev/null | awk '{print $2}')
    if [ -n "$mode" ]; then
        echo "Fan Management (Quiet/Cool/Auto) [current: $mode]"
    else
        echo "Fan Management (Quiet/Cool/Auto)"
    fi
}

# --- AI & LLM Lab ---

ai_lab_menu() {
    AI_CHOICE=$(whiptail --title "AI Setup & LLM Lab" --menu "Select Component" 18 65 5 \
    "L1" "Install PyTorch Stack (full, via 00-setup-jetson-pytorch.sh)" \
    "L2" "Install Ollama Only" \
    "L3" "Install Node.js & OpenClaw" \
    "L4" "Back to Main Menu" 3>&1 1>&2 2>&3)
    case $AI_CHOICE in
        L1)
            clear
            if [ -f "$SCRIPT_DIR/00-setup-jetson-pytorch.sh" ]; then
                bash "$SCRIPT_DIR/00-setup-jetson-pytorch.sh"
            else
                # Fallback if the full script isn't found next to this one:
                # torch only. Missing torchvision/torchaudio and the
                # cuSPARSELt/cuDSS system libs the full script installs.
                apt update && apt install -y python3-pip libopenblas-dev libjpeg-dev zlib1g-dev libavcodec-dev libavformat-dev libswscale-dev && pip3 install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/02f/de421eabbf626/torch-2.9.1-cp310-cp310-linux_aarch64.whl"
            fi
            ;;
        L2) clear && curl -fsSL https://ollama.com/install.sh | sh && whiptail --msgbox "Ollama installed!" 8 45;;
        L3) clear && curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt-get install -y nodejs && sudo ollama launch openclaw && whiptail --msgbox "Node.js and OpenClaw ready!" 8 45;;
    esac
}

# --- Advanced Options ---

toggle_desktop() {
    CURRENT_TARGET=$(systemctl get-default)
    if [ "$CURRENT_TARGET" == "graphical.target" ]; then
        if whiptail --title "Advanced Options" --yesno "Desktop GUI is ENABLED. Disable it for CLI-only boot?" 10 60; then
            systemctl set-default multi-user.target && REBOOT_REQUIRED=1
        fi
    else
        if whiptail --title "Advanced Options" --yesno "Desktop GUI is DISABLED (CLI Mode). Enable it?" 10 60; then
            systemctl set-default graphical.target && REBOOT_REQUIRED=1
        fi
    fi
}

advanced_menu() {
    ADV_CHOICE=$(whiptail --title "Advanced Options" --menu "Select Task" 18 65 6 \
    "A1" "Enable/Disable Desktop GUI" \
    "A2" "Check L4T / JetPack Version" \
    "A3" "Docker: Toggle NVIDIA Runtime" \
    "A4" "Run System OTA Updates" \
    "A5" "Back to Main Menu" 3>&1 1>&2 2>&3)
    case $ADV_CHOICE in
        A1) toggle_desktop ;;
        A2) VERSION=$(cat /etc/nv_tegra_release) && whiptail --title "L4T" --msgbox "$VERSION" 12 70;;
        A3) apt-get update && apt-get install -y nvidia-container-toolkit && REBOOT_REQUIRED=1 && whiptail --msgbox "Container Runtime installed!" 8 45;;
        A4) clear && apt-get update && apt-get upgrade -y && REBOOT_REQUIRED=1;;
    esac
}

# --- Exit Logic ---

final_exit() {
    if [ "$REBOOT_REQUIRED" -eq 1 ]; then
        if whiptail --title "Reboot Required" --yesno "Major changes were made. Reboot now?" 8 45; then
            reboot
        fi
    fi
    clear
    exit 0
}

# --- Main Program Loop ---

while true; do
    CHOICE=$(whiptail --title "Orin Toolbox: Jetson Config" --menu "Select an Option" 20 78 10 \
    "1" "System Options (Host/Wifi/BT)" \
    "2" "$(clocks_status_label)" \
    "3" "$(power_mode_label)" \
    "4" "$(fan_mode_label)" \
    "5" "AI Setup & LLM Lab" \
    "6" "Configure 40-Pin Header (jetson-io)" \
    "7" "Advanced Options (GUI/OTA/Docker)" \
    "8" "View Detailed Verbose Config" \
    "9" "Backup Configs (.bak)" \
    "10" "Exit" 3>&1 1>&2 2>&3)

    [ $? -ne 0 ] && final_exit # Handles Cancel/Esc

    case $CHOICE in
        1) system_options_menu ;;
        2) toggle_clocks_service ;;
        3)
            backup_once /etc/nvpmodel.conf
            MODE=$(whiptail --title "Power Modes" --menu "Select profile:" 15 60 5 $(grep "^< POWER_MODEL" /etc/nvpmodel.conf | grep -v "id_num" | sed -E 's/.*ID=([0-9]+) NAME=([^ >]+).*/\1 \2/') 3>&1 1>&2 2>&3)
            [ ! -z "$MODE" ] && nvpmodel -m "$MODE" && REBOOT_REQUIRED=1;;
        4)
            backup_once /etc/nvfancontrol.conf
            ensure_auto_profile
            FAN_MODE=$(whiptail --title "Fan Management" --menu "Profile:" 15 60 3 "quiet" "Quiet" "cool" "Cool" "auto" "Custom Auto" 3>&1 1>&2 2>&3)
            if [ ! -z "$FAN_MODE" ]; then
                sed -i "s/FAN_DEFAULT_PROFILE .*/FAN_DEFAULT_PROFILE $FAN_MODE/" /etc/nvfancontrol.conf
                systemctl restart nvfancontrol
                # Keep the repo's copy of nvfancontrol.conf in sync so re-running
                # 02-change-default-fan-curve.sh later doesn't silently revert
                # this choice back to whatever the repo file says.
                if [ -f "$SCRIPT_DIR/nvfancontrol.conf" ]; then
                    sed -i "s/FAN_DEFAULT_PROFILE .*/FAN_DEFAULT_PROFILE $FAN_MODE/" "$SCRIPT_DIR/nvfancontrol.conf"
                fi
            fi;;
        5) ai_lab_menu ;;
        6) /opt/nvidia/jetson-io/jetson-io.py && REBOOT_REQUIRED=1;;
        7) advanced_menu ;;
        8) nvpmodel -q --verbose > /tmp/nvp.txt && whiptail --textbox /tmp/nvp.txt 20 75;;
        9) cp /etc/nvpmodel.conf /etc/nvpmodel.conf.bak && cp /etc/nvfancontrol.conf /etc/nvfancontrol.conf.bak && whiptail --msgbox "Backups created!" 8 45;;
        10) final_exit ;;
    esac
done
