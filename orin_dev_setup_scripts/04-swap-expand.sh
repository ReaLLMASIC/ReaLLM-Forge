#!/bin/bash

# 1. Recreate the Swapfile at 28GB
sudo swapoff /swapfile 2>/dev/null
sudo fallocate -l 28G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 2. Add to fstab ONLY if it's not already there
if ! grep -q "/swapfile" /etc/fstab; then
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# 3. Add swappiness ONLY if it's not already there
if ! grep -q "vm.swappiness=100" /etc/sysctl.conf; then
  echo 'vm.swappiness=100' | sudo tee -a /etc/sysctl.conf
fi

# 4. Apply changes immediately
sudo sysctl -p
