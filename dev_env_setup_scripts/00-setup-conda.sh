#!/bin/bash

set -euo pipefail

MINICONDA_DIR="$HOME/miniconda3"
INSTALLER_PATH="$MINICONDA_DIR/miniconda.sh"
ENV_NAME="reallmforge"

mkdir -p "$MINICONDA_DIR"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$INSTALLER_PATH"
bash "$INSTALLER_PATH" -b -u -p "$MINICONDA_DIR"
rm "$INSTALLER_PATH"

source "$MINICONDA_DIR/bin/activate"

conda init --all

if ! conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create --name "$ENV_NAME" python=3.10 -y
fi

if ! grep -Fq "conda activate $ENV_NAME" ~/.zshrc 2>/dev/null; then
  echo "conda activate $ENV_NAME" >> ~/.zshrc
fi

if ! grep -Fq "conda activate $ENV_NAME" ~/.bashrc 2>/dev/null; then
  echo "conda activate $ENV_NAME" >> ~/.bashrc
fi

sudo apt update
sudo apt install -y build-essential
sudo apt install -y python3-pip
