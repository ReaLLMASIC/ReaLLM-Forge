#!/bin/bash

# Define Green Color and Reset
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Updating package lists...${NC}"
sudo apt-get update

echo -e "${GREEN}Installing dependencies...${NC}"
sudo apt-get install -y libcurl4-openssl-dev libssl-dev cmake build-essential git

# 1. Repository Setup
if [ ! -d "llama.cpp" ]; then
    echo -e "${GREEN}Cloning llama.cpp repository...${NC}"
    git clone https://github.com/ggerganov/llama.cpp
fi
cd llama.cpp || exit

# 2. Build Configuration
echo -e "${GREEN}Creating build directory...${NC}"
mkdir -p build && cd build || exit

echo -e "${GREEN}Configuring with CUDA and CURL...${NC}"
cmake .. -DGGML_CUDA=ON -DLLAMA_CURL=ON

# 3. Compilation
echo -e "${GREEN}Starting compilation using $(nproc) cores...${NC}"
cmake --build . --config Release -j$(nproc)

echo -e "${GREEN}--------------------------------------------------${NC}"
echo -e "${GREEN}Build complete! Binaries are in llama.cpp/build/bin/${NC}"
echo -e "${GREEN}--------------------------------------------------${NC}"
