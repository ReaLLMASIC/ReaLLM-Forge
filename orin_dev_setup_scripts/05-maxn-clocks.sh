#!/bin/bash
# 1. Set Power Mode to MAXN (Your Mode 2)
sudo nvpmodel -m 2

# 2. Lock Clocks & Max Fan
sudo jetson_clocks

# 3. Optimize Memory for LLM Inference
sudo sysctl -w vm.swappiness=100
sudo sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
