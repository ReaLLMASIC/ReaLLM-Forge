#!/bin/bash

## Update Apt list and install python-pip
sudo apt update
sudo apt install python3-pip -y

## Install pip dependencies
pip install sentencepiece tiktoken tqdm rich torchinfo plotly seaborn tensorboard pyyaml textual
pip install "triton==3.5.1"

## Get dataset
RED='\033[0;31m'
NC='\033[0m' # No Color
echo -e "${RED}After you can run bash data/shakespeare_char/get_dataset.sh"

