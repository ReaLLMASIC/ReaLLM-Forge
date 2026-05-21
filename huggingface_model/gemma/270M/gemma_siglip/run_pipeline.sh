#!/bin/bash
# Exit immediately if any command fails
set -e

echo "=========================================="
echo "🚀 STARTING GEMMA-SIGLIP2 PIPELINE"
echo "=========================================="

# 1. Run the training script (using the default 'both' mode to train and validate)
echo -e "\n\n>>> 1/4: RUNNING TRAINING <<<"
python gemma_siglip2_finetune.py --train_mode map_only

# 2. Setup a test image
TEST_IMAGE="test_image.jpg"
if [ ! -f "$TEST_IMAGE" ]; then
    echo -e "\n📥 Downloading a sample image for testing..."
    # Downloads a standard picture of a dog
    wget -qO $TEST_IMAGE "https://images.unsplash.com/photo-1543466835-00a7907e9de1?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"
fi

# 3. Test Mode 1: Zero-Shot Category
echo -e "\n\n>>> 2/4: TESTING CATEGORY MODE <<<"
python gemma_siglip2_inference.py $TEST_IMAGE --mode category --concepts dog cat bird car

# 4. Test Mode 2: Dense Captioning
echo -e "\n\n>>> 3/4: TESTING CAPTION MODE (DENSE PATCHES) <<<"
python gemma_siglip2_inference.py $TEST_IMAGE --mode caption --query "Describe the main subject of this image in detail."

# 5. Test Mode 3: MAP Query
echo -e "\n\n>>> 4/4: TESTING MAP QUERY MODE (GLOBAL TOKEN) <<<"
python gemma_siglip2_inference.py $TEST_IMAGE --mode map_query --query "Describe the main subject of this image in detail."

echo -e "\n=========================================="
echo "✅ PIPELINE COMPLETE!"
echo "=========================================="
