# Open Source CJK Analysis Tool

## Gemma 3 270M semantic factorization

`semantic_factorization.py` estimates how many embedding/LM-head rows can be
replaced by a base row plus an orthographic feature. It applies these rules:

1. Replace `▁token` with `SPACE + token` when `token` exists. Gemma's tokenizer
   normalizer uses `▁` (U+2581) for a space; the vocabulary does not store a
   literal leading ASCII space.
2. Replace a token beginning with an uppercase Unicode character with
   `CAPITALIZE + lower-first(token)` when that counterpart exists. A leading
   `▁` is ignored while identifying the first character.
3. Replace a token whose cased characters are all uppercase with
   `ALL_CAPS + lower(token)` when that counterpart exists.

The rule sets overlap, so savings use their **union**, not their sum. The CLI
compares these defaults: `google/gemma-3-270m`, `Qwen/Qwen3-0.6B` (the released
model is 0.6 **billion**, not 0.6 million), and tiktoken's `o200k_base`.

| Vocabulary/head | Before | Unique removed | After | Reduction |
|---|---:|---:|---:|---:|
| Gemma 3 270M | 262,144 | 65,035 | 197,109 | 24.81% |
| Qwen 3 0.6B | 151,936 | 32,799 | 119,137 | 21.59% |
| tiktoken `o200k_base` | 200,019 | 47,022 | 152,997 | 23.51% |

Gemma's per-rule detail is:

| Rule | Matches | Newly removed in rule order |
|---|---:|---:|
| Leading space | 40,893 | 40,893 |
| Initial capital | 27,786 | 18,530 |
| All caps | 8,859 | 5,612 |
| **Unique total** | **65,035** | **65,035** |

That reduces the estimated vocabulary/head from **262,144 to 197,109 rows**, a
**24.81%** reduction. The estimate does not count three new feature vectors; if
they occupy ordinary vocabulary rows, the physical total is 197,112. It also
does not claim an immediately usable tokenizer: BPE merges referring to removed
tokens must be rewritten, encoding must emit base-plus-feature structure, and
the model must be trained or fine-tuned with compositional input/output heads.

Install the repository requirements, accept the Gemma license on Hugging Face,
set `HF_TOKEN`, and run all defaults:

```bash
python analysis/tokenizer_analysis/semantic_factorization.py
```

Add any number of Hugging Face models or tiktoken encodings; one inaccessible
model produces a warning without preventing the other comparisons:

```bash
python analysis/tokenizer_analysis/semantic_factorization.py \
  --hf-model meta-llama/Llama-3.2-1B \
  --hf-model Qwen/Qwen3-4B \
  --tiktoken-encoding cl100k_base
```

Use `--no-defaults` to analyze only the explicitly supplied sources. Local
Hugging Face tokenizer JSON files can also be positional arguments:

```bash
python analysis/tokenizer_analysis/semantic_factorization.py \
  --no-defaults /path/to/tokenizer.json
```

For Hugging Face model IDs, `config.vocab_size` defines the current head size;
unused/padded IDs remain in the estimate and added tokenizer IDs beyond the head
are excluded. For tiktoken, `n_vocab` defines the extent, including reserved
holes. Invalid UTF-8 byte fragments are retained but conservatively considered
non-factorizable. A standalone tokenizer JSON has no model config, so its
`model.vocab` length is used.

A practical implementation can represent a factored token as
`(base_id, feature_mask)`. For input embeddings, add learned feature vectors to
the base embedding. For the output head, score the base rows once and predict
the three feature bits with small auxiliary heads (or score only valid
base/feature combinations). Retokenize training data and fine-tune before using
the smaller head; simply deleting rows changes sequence boundaries and cannot
reproduce the original BPE model.

This repository contains the **Open Source CJK Analysis Tool**, a Python script (`open_source_cjk_analysis.py`) that provides detailed analysis of tokenizers with respect to Chinese (C), Japanese (J), and Korean (K) characters. The tool can analyze token coverage, symbol representation, and overlaps among these languages in tokenizer vocabularies.

## Features

- **Subcategory Analysis**: 
  - For each subcategory (e.g., Hiragana, Unified Ideographs, etc.), the script calculates:
    - Total tokens in the vocabulary containing characters from the subcategory.
    - Number of unique symbols (characters) found in the vocabulary.
    - Total possible characters in the subcategory.
    - Percentage of symbols represented in the vocabulary.
    
- **Total Tokens in Categories**:
  - Calculates the total number of tokens falling into any of the C, J, or K categories without double-counting.

- **Overlap Analysis**:
  - Analyzes overlaps between the C, J, and K categories, providing counts for:
    - Tokens in multiple categories (e.g., CJ, CK, JK, CJK).

## Requirements

### Python Libraries

- `transformers`
- `rich`
- `tiktoken`

Install dependencies using:
```bash
pip install transformers rich tiktoken
```

## Usage

1. Place the `open_source_cjk_analysis.py` script in your working directory.
2. Run the script using Python:
   ```bash
   python open_source_cjk_analysis.py
   ```

3. The script analyzes the tokenizers specified in the `tokenizers` list:
   ```python
   tokenizers = [
       {"name": "google/gemma-7b", "is_tiktoken": False},
       {"name": "o200k_base", "is_tiktoken": True},
       {"name": "mistralai/Mistral-7B-Instruct-v0.3", "is_tiktoken": False},
   ]
   ```
   You can modify this list to include additional tokenizers or replace existing ones.

## Output

The script provides the following outputs:

### 1. **Subcategory Analysis Table**

| Language | Subcategory       | Token Count | Unique Symbols Found | Total Possible Characters | % of Symbols in Range |
|----------|-------------------|-------------|-----------------------|---------------------------|------------------------|
| C        | Unified Ideographs| 1234        | 892                  | 20000                     | 4.46%                 |
| ...      | ...               | ...         | ...                  | ...                       | ...                   |
| **TOTAL**|                   | 5678        | 2345                 | 67890                     | 12.34%                |

### 2. **Total Tokens in Categories Table**

| Category | Token Count |
|----------|-------------|
| C        | 4567        |
| J        | 1234        |
| K        | 890         |
| **TOTAL (Any C, J, K)** | 5678 |

### 3. **Overlap Analysis Table**

| Overlap  | Token Count |
|----------|-------------|
| CJ       | 456         |
| CK       | 234         |
| JK       | 123         |
| CJK      | 45          |

### 4. **Summary**

- Total tokens in the tokenizer: `256000`
- Total tokens in any C, J, or K category (no double-counting): `5678`.

## Contributing

Contributions to improve the analysis or add features are welcome! Please submit a pull request or open an issue if you encounter any problems.
