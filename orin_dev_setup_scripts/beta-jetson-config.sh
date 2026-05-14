#!/bin/bash

# Ensure the script is run with sudo [cite: 17, 63]
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root: sudo $0"
  exit 1
fi

REBOOT_REQUIRED=0 # Session memory for major changes [cite: 255]

# --- System & Identity Functions [cite: 356, 366] ---

change_hostname() {
    CURRENT_HOST=$(hostname)
    NEW_HOST=$(whiptail --title "System Options" --inputbox "Enter new hostname:" 8 45 "$CURRENT_HOST" 3>&1 1>&2 2>&3)
    if [ $? -eq 0 ] && [ ! -z "$NEW_HOST" ]; then
        hostnamectl set-hostname "$NEW_HOST" [cite: 357]
        sed -i "s/$CURRENT_HOST/$NEW_HOST/g" /etc/hosts [cite: 357]
        whiptail --msgbox "Hostname changed to $NEW_HOST." 8 45
        REBOOT_REQUIRED=1
    fi
}

manage_wifi() {
    whiptail --infobox "Scanning for Wi-Fi networks..." 8 45
    WIFI_LIST=$(nmcli -t -f SSID dev wifi list | grep -v '^--' | head -n 10) [cite: 358]
    mapfile -t WIFI_ARR <<< "$WIFI_LIST"
    MENU_ARR=()
    for ssid in "${WIFI_ARR[@]}"; do MENU_ARR+=("$ssid" ""); done
    CHOSEN_SSID=$(whiptail --title "Wi-Fi Setup" --menu "Select a network:" 15 60 8 "${MENU_ARR[@]}" 3>&1 1>&2 2>&3)
    if [ ! -z "$CHOSEN_SSID" ]; then
        PASS=$(whiptail --title "Wi-Fi Setup" --passwordbox "Enter password for $CHOSEN_SSID:" 8 45 3>&1 1>&2 2>&3)
        [ $? -eq 0 ] && clear && nmcli dev wifi connect "$CHOSEN_SSID" password "$PASS" && read -p "Done. Press Enter..." [cite: 359]
    fi
}

toggle_bluetooth() {
    if rfkill list bluetooth | grep -q "yes"; then [cite: 360]
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

# --- Core Hardware Functions [cite: 28, 47] ---

ensure_auto_profile() {
    if ! grep -q "FAN_PROFILE auto" /etc/nvfancontrol.conf; then [cite: 216]
        # Injects custom curve provided in your nvfancontrol.conf [cite: 173, 217]
        sed -i '/THERMAL_GROUP 0 {/i \
        FAN_PROFILE auto {\
                0       0       255     6000\
                40      0       255     6000\
                50      0       180     4200\
                60      0       100     2500\
                105     0       0       0\
        }' /etc/nvfancontrol.conf
    fi
}

toggle_clocks_service() {
    if systemctl is-active --quiet jetson_clocks; then
        systemctl disable jetson_clocks --now && whiptail --msgbox "jetson_clocks DISABLED." 8 45 [cite: 165]
    else
        [ ! -f /etc/systemd/system/jetson_clocks.service ] && cat <<EOF > /etc/systemd/system/jetson_clocks.service [cite: 168]
[Unit]
Description=Maximize Jetson Performance
After=nvpmodel.service
[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
EOF
        systemctl daemon-reload && systemctl enable jetson_clocks --now && whiptail --msgbox "jetson_clocks ENABLED." 8 45 [cite: 167, 171]
    fi
}

# --- AI & LLM Lab [cite: 320, 332] ---

ai_lab_menu() {
    AI_CHOICE=$(whiptail --title "AI Setup & LLM Lab" --menu "Select Component" 18 65 5 \
    "L1" "Install PyTorch Stack (Jetson AI Lab)" \
    "L2" "Install Ollama Only" \
    "L3" "Install Node.js & OpenClaw" \
    "L4" "Back to Main Menu" 3>&1 1>&2 2>&3)
    case $AI_CHOICE in
        L1) clear && apt update && apt install -y python3-pip libopenblas-dev libjpeg-dev zlib1g-dev libavcodec-dev libavformat-dev libswscale-dev && pip3 install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/02f/de421eabbf626/torch-2.9.1-cp310-cp310-linux_aarch64.whl"[cite: 322, 323];;
        L2) clear && curl -fsSL https://ollama.com/install.sh | sh && whiptail --msgbox "Ollama installed!" 8 45[cite: 324, 333];;
        L3) clear && curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt-get install -y nodejs && sudo ollama launch openclaw && whiptail --msgbox "Node.js and OpenClaw ready!" 8 45[cite: 324, 334];;
    esac
}

# --- Advanced Options [cite: 259, 393] ---

toggle_desktop() {
    CURRENT_TARGET=$(systemctl get-default) [cite: 384]
    if [ "$CURRENT_TARGET" == "graphical.target" ]; then
        if whiptail --title "Advanced Options" --yesno "Desktop GUI is ENABLED. Disable it for CLI-only boot?" 10 60; then
            systemctl set-default multi-user.target && REBOOT_REQUIRED=1 [cite: 390]
        fi
    else
        if whiptail --title "Advanced Options" --yesno "Desktop GUI is DISABLED (CLI Mode). Enable it?" 10 60; then
            systemctl set-default graphical.target && REBOOT_REQUIRED=1 [cite: 384]
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
        A2) VERSION=$(cat /etc/nv_tegra_release) && whiptail --title "L4T" --msgbox "$VERSION" 12 70[cite: 262];;
        A3) apt-get update && apt-get install -y nvidia-container-toolkit && REBOOT_REQUIRED=1 && whiptail --msgbox "Container Runtime installed!" 8 45[cite: 263, 315];;
        A4) clear && apt-get update && apt-get upgrade -y && REBOOT_REQUIRED=1[cite: 264];;
    esac
}

# --- Exit Logic [cite: 233, 250, 310] ---

final_exit() {
    if [ "$REBOOT_REQUIRED" -eq 1 ]; then
        if whiptail --title "Reboot Required" --yesno "Major changes were made. Reboot now?" 8 45; then
            reboot [cite: 248]
        fi
    fi
    clear
    exit 0
}

# --- Main Program Loop ---

while true; do
    CHOICE=$(whiptail --title "Orin Toolbox: Jetson Config" --menu "Select an Option" 20 75 10 \
    "1" "System Options (Host/Wifi/BT)" \
    "2" "Toggle jetson_clocks Performance" \
    "3" "Set Power Mode (nvpmodel)" \
    "4" "Fan Management (Quiet/Cool/Auto)" \
    "5" "AI Setup & LLM Lab" \
    "6" "Configure 40-Pin Header (jetson-io)" \
    "7" "Advanced Options (GUI/OTA/Docker)" \
    "8" "View Detailed Verbose Config" \
    "9" "Backup Configs (.bak)" \
    "10" "Exit" 3>&1 1>&2 2>&3)

    [ $? -ne 0 ] && final_exit # Handles Cancel/Esc [cite: 234]

    case $CHOICE in
        1) system_options_menu ;;
        2) toggle_clocks_service ;;
        3) 
            MODE=$(whiptail --title "Power Modes" --menu "Select profile:" 15 60 5 $(grep "^< POWER_MODEL" /etc/nvpmodel.conf | grep -v "id_num" | sed -E 's/.*ID=([0-9]+) NAME=([^ >]+).*/\1 \2/') 3>&1 1>&2 2>&3) [cite: 146, 161]
            [ ! -z "$MODE" ] && nvpmodel -m $MODE && REBOOT_REQUIRED=1[cite: 101, 252];;
        4) 
            ensure_auto_profile
            FAN_MODE=$(whiptail --title "Fan Management" --menu "Profile:" 15 60 3 "quiet" "Quiet" "cool" "Cool" "auto" "Custom Auto" 3>&1 1>&2 2>&3) [cite: 215, 226]
            [ ! -z "$FAN_MODE" ] && sed -i "s/FAN_DEFAULT_PROFILE .*/FAN_DEFAULT_PROFILE $FAN_MODE/" /etc/nvfancontrol.conf && systemctl restart nvfancontrol[cite: 187, 219];;
        5) ai_lab_menu ;;
        6) /opt/nvidia/jetson-io/jetson-io.py && REBOOT_REQUIRED=1[cite: 15, 254];;
        7) advanced_menu ;;
        8) nvpmodel -q --verbose > /tmp/nvp.txt && whiptail --textbox /tmp/nvp.txt 20 75[cite: 105, 147];;
        9) cp /etc/nvpmodel.conf /etc/nvpmodel.conf.bak && cp /etc/nvfancontrol.conf /etc/nvfancontrol.conf.bak && whiptail --msgbox "Backups created!" 8 45[cite: 148, 244];;
        10) final_exit ;;
    esac
done
