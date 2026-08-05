#!/bin/bash

FILE="$1"

NFC_90="p90_nfc.txt"
NFC_10="p10_nfc.txt"

NFD_90="p90_nfd.txt"
NFD_10="p10_nfd.txt"

TOTAL=$(wc -l < "$FILE")
PCT=$((TOTAL * 10 / 100))
head -n -$PCT "$FILE" > "$NFC_90"
tail -n $PCT "$FILE" > "$NFC_10"

python3 hangul_nfc_to_nfd.py "$NFC_90" "$NFD_90"
python3 hangul_nfc_to_nfd.py "$NFC_10" "$NFD_10"

