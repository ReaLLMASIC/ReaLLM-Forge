#!/bin/bash

sudo cp -r nvfancontrol.conf /etc/
sudo systemctl restart nvfancontrol.service
sudo pip3 install -U jetson-stats

## Jtop message
Green='\033[0;32m'
echo -e "${Green}After you can run 'jtop' for system stats"
