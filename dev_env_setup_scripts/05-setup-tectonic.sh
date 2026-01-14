#!/bin/bash

# download dependencies
sudo apt update -y
sudo apt upgrade -y
sudo apt install libgraphite2-3 -y

# obtain appimage
wget https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64.AppImage

# move to folder within path
chmod +x tectonic-0.15.0-x86_64.AppImage
sudo mv tectonic-0.15.0-x86_64.AppImage /usr/local/bin
