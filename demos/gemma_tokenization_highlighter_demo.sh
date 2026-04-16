#!/bin/bash
# demos/gemma_tokenization_highlighter_demo.sh
#
# Demonstrates all colour modes of gemma_tokenization_highlighter.py.
# Modes 1-3 are tokenization-only (fast, no model download).
# Modes 4-6 require --inference (downloads the Gemma 270M model).

SCRIPT="huggingface_model/gemma/270M/gemma_tokenization_highlighter.py"
TEXT="The quick brown fox jumps over the lazy dog. 안녕하세요, 세계! Héllo wörld."
OUTDIR="out/gemma_highlighter_demo"
mkdir -p "$OUTDIR"

echo "=== Gemma Tokenization Highlighter Demo ==="
echo ""

# -------------------------------------------------------
# 1. token_id – deterministic colour per token ID
# -------------------------------------------------------
echo "--- Mode: token_id (no model needed) ---"
python "$SCRIPT" \
  --text "$TEXT" \
  --colour_mode token_id \
  --output_file "$OUTDIR/token_id_terminal.txt" \
  --html         "$OUTDIR/token_id.html"

# -------------------------------------------------------
# 2. char_length – colour by character count per token
# -------------------------------------------------------
echo "--- Mode: char_length (no model needed) ---"
python "$SCRIPT" \
  --text "$TEXT" \
  --colour_mode char_length \
  --output_file "$OUTDIR/char_length_terminal.txt" \
  --html         "$OUTDIR/char_length.html"

# -------------------------------------------------------
# 3. byte_length – colour by UTF-8 byte count per token
# -------------------------------------------------------
echo "--- Mode: byte_length (no model needed) ---"
python "$SCRIPT" \
  --text "$TEXT" \
  --colour_mode byte_length \
  --output_file "$OUTDIR/byte_length_terminal.txt" \
  --html         "$OUTDIR/byte_length.html"

# -------------------------------------------------------
# 4. rank – green = rank 1, red = rank >= rank_red
# -------------------------------------------------------
echo "--- Mode: rank (inference) ---"
python "$SCRIPT" \
  --text "$TEXT" \
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
  --text "$TEXT" \
  --inference \
  --colour_mode probability \
  --output_file "$OUTDIR/probability_terminal.txt" \
  --html         "$OUTDIR/probability.html"

# -------------------------------------------------------
# 6. minmax – relative min/max normalised probability
# -------------------------------------------------------
echo "--- Mode: minmax (inference) ---"
python "$SCRIPT" \
  --text "$TEXT" \
  --inference \
  --colour_mode minmax \
  --output_file "$OUTDIR/minmax_terminal.txt" \
  --html         "$OUTDIR/minmax.html"

echo ""
echo "=== All outputs saved to $OUTDIR ==="
echo "  Terminal: *_terminal.txt"
echo "  HTML:     *.html (open in browser)"
