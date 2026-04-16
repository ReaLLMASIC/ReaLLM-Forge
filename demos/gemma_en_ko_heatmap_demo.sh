#!/bin/bash
# demos/gemma_en_ko_heatmap_demo.sh
#
# Demonstrates all colour modes of test_gemma_en_ko_heatmap.py.
# Fetches EN-KO translation pairs from Helsinki-NLP/opus-100 and
# produces side-by-side tokenization heatmaps.
#
# Modes 1-3 are tokenization-only (fast, no model download).
# Modes 4-6 require --inference (downloads the Gemma 270M model).

SCRIPT="huggingface_model/gemma/270M/test_gemma_en_ko_heatmap.py"
PAIRS=5
OUTDIR="out/gemma_en_ko_demo"
mkdir -p "$OUTDIR"

echo "=== Gemma EN-KO Heatmap Demo ==="
echo ""

# -------------------------------------------------------
# 1. byte_length – colour by UTF-8 byte count per token
# -------------------------------------------------------
echo "--- Mode: byte_length (no model needed) ---"
python "$SCRIPT" \
  --num_pairs   "$PAIRS" \
  --colour_mode byte_length \
  --output_file "$OUTDIR/byte_length_terminal.txt" \
  --html         "$OUTDIR/byte_length.html"

# -------------------------------------------------------
# 2. char_length – colour by character count per token
# -------------------------------------------------------
echo "--- Mode: char_length (no model needed) ---"
python "$SCRIPT" \
  --num_pairs   "$PAIRS" \
  --colour_mode char_length \
  --output_file "$OUTDIR/char_length_terminal.txt" \
  --html         "$OUTDIR/char_length.html"

# -------------------------------------------------------
# 3. token_id – deterministic colour per token ID
# -------------------------------------------------------
echo "--- Mode: token_id (no model needed) ---"
python "$SCRIPT" \
  --num_pairs   "$PAIRS" \
  --colour_mode token_id \
  --output_file "$OUTDIR/token_id_terminal.txt" \
  --html         "$OUTDIR/token_id.html"

# -------------------------------------------------------
# 4. rank – green = rank 1, red = rank >= rank_red
# -------------------------------------------------------
echo "--- Mode: rank (inference) ---"
python "$SCRIPT" \
  --num_pairs   "$PAIRS" \
  --inference \
  --colour_mode rank \
  --rank_red 50 \
  --output_file "$OUTDIR/rank_terminal.txt" \
  --html         "$OUTDIR/rank.html"

# -------------------------------------------------------
# 5. probability – green = 1.0, red = 0.0 (absolute)
# -------------------------------------------------------
echo "--- Mode: probability (inference) ---"
python "$SCRIPT" \
  --num_pairs   "$PAIRS" \
  --inference \
  --colour_mode probability \
  --output_file "$OUTDIR/probability_terminal.txt" \
  --html         "$OUTDIR/probability.html"

# -------------------------------------------------------
# 6. minmax – relative min/max normalised probability
# -------------------------------------------------------
echo "--- Mode: minmax (inference) ---"
python "$SCRIPT" \
  --num_pairs   "$PAIRS" \
  --inference \
  --colour_mode minmax \
  --output_file "$OUTDIR/minmax_terminal.txt" \
  --html         "$OUTDIR/minmax.html"

echo ""
echo "=== All outputs saved to $OUTDIR ==="
echo "  Terminal: *_terminal.txt"
echo "  HTML:     *.html (open in browser)"
