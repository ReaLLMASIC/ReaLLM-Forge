#!/usr/bin/env bash

###############################################################################
# Bash settings to exit on errors (-e), treat unset variables as errors (-u),
# and fail on any command in a pipeline (-o pipefail).
###############################################################################
set -euo pipefail
set -x

###############################################################################
# Colors for printouts
###############################################################################
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

###############################################################################
# Helper functions
###############################################################################

print_step() {
    local step_message="$1"
    echo -e "${BLUE}=== ${step_message} ===${NC}"
}

check_file_exists() {
    local file_path="$1"
    if [[ ! -f "$file_path" ]]; then
        echo -e "${RED}ERROR: Expected file '${file_path}' not found.${NC}"
        exit 1
    else
        echo -e "${GREEN}File '${file_path}' confirmed.${NC}"
    fi
}

###############################################################################
# Step 1: Create directory et-nanogpt and virtual environment
###############################################################################
print_step "Step 1: Creating 'et-nanogpt' directory and Python virtual environment"

mkdir -p et-nanogpt
cd et-nanogpt

echo -e "${YELLOW}Updating pip and creating virtual environment...${NC}"
python3 -m pip install --upgrade pip
python3 -m venv venv
source venv/bin/activate

###############################################################################
# Step 2: Clone ExecuTorch repository and submodules
###############################################################################
print_step "Step 2: Cloning ExecuTorch repository and initializing submodules"

mkdir -p third-party
git clone -b release/0.4 https://github.com/pytorch/executorch.git third-party/executorch
cd third-party/executorch
git submodule update --init --recursive

###############################################################################
# Step 3: Install ExecuTorch requirements
###############################################################################
print_step "Step 3: Installing ExecuTorch requirements"

PYTHON_EXECUTABLE=python3 ./install_requirements.sh

cd ../../

###############################################################################
# Step 4: Download model.py and vocab.json
###############################################################################
print_step "Step 4: Downloading 'model.py' and 'vocab.json'"

curl -fLO https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py
curl -fLO https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json

# Verify downloads
check_file_exists "model.py"
check_file_exists "vocab.json"

###############################################################################
# Step 5: Copy export_nanogpt.py into current folder
###############################################################################
print_step "Step 5: Copying 'export_nanogpt.py'"

cp ../export_nanogpt.py .
check_file_exists "export_nanogpt.py"

###############################################################################
# Step 6: Export the model to nanogpt.pte
###############################################################################
print_step "Step 6: Exporting NanoGPT model"

python3 export_nanogpt.py

# Check that nanogpt.pte was created
check_file_exists "nanogpt.pte"

###############################################################################
# Step 7: Copy main.cpp into current folder
###############################################################################
print_step "Step 7: Copying 'main.cpp'"

cp ../main.cpp .
check_file_exists "main.cpp"

###############################################################################
# Step 8: Download basic_sampler.h and basic_tokenizer.h
###############################################################################
print_step "Step 8: Downloading 'basic_sampler.h' and 'basic_tokenizer.h'"

curl -fLO https://raw.githubusercontent.com/pytorch/executorch/main/examples/llm_manual/basic_sampler.h
curl -fLO https://raw.githubusercontent.com/pytorch/executorch/main/examples/llm_manual/basic_tokenizer.h

# Verify downloads
check_file_exists "basic_sampler.h"
check_file_exists "basic_tokenizer.h"

###############################################################################
# Step 9: Copy CMakeLists.txt
###############################################################################
print_step "Step 9: Copying 'CMakeLists.txt'"

cp ../CMakeLists.txt .
check_file_exists "CMakeLists.txt"

###############################################################################
# Step 10: Confirm expected files in current directory
###############################################################################
print_step "Step 10: Validating all required files"

required_files=(
  "CMakeLists.txt"
  "main.cpp"
  "basic_tokenizer.h"
  "basic_sampler.h"
  "export_nanogpt.py"
  "model.py"
  "vocab.json"
  "nanogpt.pte"
)

for f in "${required_files[@]}"; do
    check_file_exists "$f"
done

###############################################################################
# Step 11: Install ExecuTorch requirements (clean) and build
###############################################################################
print_step "Step 11: Re-install ExecuTorch requirements (clean) and build"

cd third-party/executorch
PYTHON_EXECUTABLE=python3 ./install_requirements.sh --clean

cd ../../

echo -e "${YELLOW}Running CMake...${NC}"
mkdir -p cmake-out
cd cmake-out
cmake ..

echo -e "${YELLOW}Building project...${NC}"
cmake --build . -j"$(nproc)"

echo -e "${GREEN}Build completed successfully!${NC}"

cp nanogpt_runner ../

cd ../


