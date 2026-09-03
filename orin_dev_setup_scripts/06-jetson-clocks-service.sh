#!/bin/bash

# Define file paths
INIT_SCRIPT="/usr/local/bin/jetson_init_clocks.sh"
SERVICE_FILE="/etc/systemd/system/jetson-clocks.service"

echo "--- Starting Jetson Clock Service Setup ---"

# 1. Create the hardware initialization script
# Using absolute paths to ensure it runs correctly at boot
sudo bash -c "cat << 'EOF' > $INIT_SCRIPT
#!/bin/bash
# Set Power Mode to MAXN
/usr/sbin/nvpmodel -m 2
# Lock Clocks and Fan to Max
/usr/bin/jetson_clocks
EOF"

# 2. Set permissions
sudo chmod +x $INIT_SCRIPT

# 3. Create the systemd service file
sudo bash -c "cat << 'EOF' > $SERVICE_FILE
[Unit]
Description=Lock Jetson Clocks to Max on Boot
After=multi-user.target

[Service]
Type=oneshot
ExecStart=$INIT_SCRIPT
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF"

# 4. Enable and Start the service
echo "Registering and starting service..."
sudo systemctl daemon-reload
sudo systemctl enable jetson-clocks.service
sudo systemctl start jetson-clocks.service

echo "------------------------------------------------"
echo "Success! jetson_clocks is now a boot service."
echo "------------------------------------------------"
