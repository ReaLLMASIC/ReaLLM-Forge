# OOV & Unicode Stress-Test Prompts

This directory contains evaluation prompts specifically curated to test tokenizer robustness, out-of-vocabulary (OOV) handling, character/BPE byte fallback mechanisms, and Hangul multicontext factorization models.

## Files

| File | Description | Count |
| :--- | :--- | :--- |
| [`all_oov_prompts.txt`](./all_oov_prompts.txt) | Consolidated text file containing all 20 unique stress-test prompts | 20 |
| [`oov_prompts.json`](./oov_prompts.json) | Structured JSON dataset with rich metadata (categories, target phenomena, description) | 20 |
| [`oov_prompts.txt`](./oov_prompts.txt) | Original Character Baseline test suite (Foreign loanwords, 2000s slang, Old Hangul, math) | 5 |
| [`bpe_oov_prompts.txt`](./bpe_oov_prompts.txt) | Byte-Fallback BPE test suite (Rare Hanja, comic onomatopoeia, Maxwell equations, emojis) | 5 |
| [`all_baseline_oov_prompts.txt`](./all_baseline_oov_prompts.txt) | Combined baseline test suite (Historical texts, Phoenician script, vector calculus, AI announcements) | 5 |
| [`pure_byte_prompts.txt`](./pure_byte_prompts.txt) | Pure Byte test suite (Egyptian hieroglyphs, complex multi-person ZWJ family emojis, flags) | 5 |

## Evaluated Categories & Phenomena

1. **Foreign Loanwords & Latin Script**: Tests loanword transliteration (e.g. Thai dishes `똠얌꿍`, `팟타이`) and inline English glosses in parentheses.
2. **Internet Slang & Rare Syllables**: 2000s Korean web culture slang (e.g., `아햏햏`, `즐드삼 쀍`) and non-standard syllables with irregular codepoints.
3. **Archaic / Middle Korean (Old Hangul)**: 15th-century Hunminjeongeum characters including archaic consonants (`ㅿ` Bansiot, `ᄠ` Ssangbieup-digeut) and the arae-a vowel (`ᆞ`).
4. **Mathematical & Scientific Notation**: Calculus operators and formulas (`∫`, `∇`, `∂`, `∮`, `∑`, `∞`, `≈`, `≠`, `≥`).
5. **Rare Hanja (CJK Ideographs)**: Ultra-rare, high-stroke ideographs (e.g. `龘` [48 strokes], `爨` [29 strokes]) absent from typical subword sets.
6. **Ancient SMP Scripts**: Supplementary Multilingual Plane 4-byte UTF-8 scripts including ancient Phoenician (`𐤀𐤁𐤂`) and Egyptian Hieroglyphs (`𓀀𓀁𓀂`).
7. **Complex Emojis & ZWJ Sequences**: Regional indicator flag ligatures (`🇰🇷`) and multi-codepoint Zero-Width Joiner family sequences (`👨‍👩‍👧‍👦`).
8. **Modern Tech Announcements**: Alphanumeric model and version strings (`ReaLLM v2.0`) combined with trending emojis.

## Usage

To evaluate all models against these prompts:
```bash
python3 benchmarks/run_oov_evaluations.py
```
Or specify a custom prompt file:
```bash
python3 benchmarks/run_oov_evaluations.py --prompts_file benchmarks/prompts/all_oov_prompts.txt
```
