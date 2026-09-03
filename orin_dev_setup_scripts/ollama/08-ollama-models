#!/bin/bash

# 1. Scrape trending models from the Ollama library page
echo ">>> Scraping available models from Ollama Library..."
scraped_models=$(curl -s https://ollama.com/library | grep -oP 'href="/library/\K[^"]+' | head -n 15)

# 2. Add your specific requests (Gemma 3, 4 and Qwen 3.5)
# Using 'sort -u' to remove any duplicates between the scrape and manual list
all_models=$(echo -e "$scraped_models\nqwen3.5:4b\nqwen3.5:7b\ngemma4:e2b\ngemma4:e4b\ngemma3:1b\ngemma3:4b\ngemma3:12b\ngemma3:27b\nllama4:8b" | sort -u)

# Convert to an array
mapfile -t model_array <<< "$all_models"

echo "------------------------------------------"
echo "   OLLAMA AUTO-SCRAPER & PULLER (2026)   "
echo "------------------------------------------"

# Display Disk Space
free_space=$(df -h . | awk 'NR==2 {print $4}')
echo "Available Disk Space: $free_space"
echo "------------------------------------------"

# Generate Menu in a clean grid
for i in "${!model_array[@]}"; do
    printf "%2s) %-18s" "$((i+1))" "${model_array[$i]}"
    if (( (i+1) % 2 == 0 )); then echo ""; fi
done

echo -e "\n------------------------------------------"
read -p "Select number to pull (or 'q' to quit): " choice

if [[ "$choice" == "q" ]]; then
    exit 0
elif [[ "$choice" -ge 1 && "$choice" -le "${#model_array[@]}" ]]; then
    selected_model="${model_array[$((choice-1))]}"
    
    # Check for specific tags (if it's just a name like 'gemma', it pulls 'latest')
    echo ">>> Pulling $selected_model..."
    ollama pull "$selected_model"
    
    echo "------------------------------------------"
    echo "Success! Run it with: ollama run $selected_model"
else
    echo "Invalid selection."
fi
