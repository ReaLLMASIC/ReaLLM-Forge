Here’s a rewritten **README.md** that matches what you actually have now: `get_dataset.sh`, `organize_datasets.py`, `verilog_ts_colorize.py`, and the Tree-sitter highlight workflow (including the “filter the Neovim query file to what the Python grammar supports” behavior).

You can paste this over your current `README.md`.

````md
# PyraNet-Verilog (helpers)

This folder contains small utilities for working with the
**PyraNet-Verilog** dataset on Hugging Face:

- Dataset: `bnadimi/PyraNet-Verilog`
- Contents: Verilog code samples + metadata fields (rankings, descriptions, compile info, etc.)
- We primarily extract the `code` field into standalone `.v` files, and (optionally) generate
  a **Tree-sitter syntax-highlight “shadow” file** aligned byte-for-byte with each source file.

---

## Files

- **`get_dataset.sh`**
  - Convenience wrapper that runs `organize_datasets.py` with defaults.

- **`organize_datasets.py`**
  - Downloads the dataset (streaming by default) and writes each sample’s `code` into:
    - `orig/orig_<index>.v` (7-digit zero-padded)
  - Supports limiting rows, disabling streaming, and overwriting existing outputs.

- **`verilog_ts_colorize.py`**
  - Runs Tree-sitter parsing + Tree-sitter highlight queries and emits a per-byte “color symbol” mask.
  - Output file name defaults to:
    - `<input>.tscolors.txt`
  - Output is **exactly the same length** as the input file and preserves newlines.

---

## 1) Download + extract the Verilog sources

### Quick start (recommended)
```bash
bash get_dataset.sh
````

By default, files are written to:

```
orig/orig_<index>.v
```

### Advanced usage

```bash
python3 organize_datasets.py --help
```

Notable flags:

* `--output-dir` : output directory for extracted files (default: `./orig`)
* `--max-rows N` : extract only the first `N` samples (handy for testing)
* `--no-streaming` : disable streaming (downloads the whole dataset; can be large)
* `--overwrite` : overwrite existing `orig_<index>.v` files

Example:

```bash
python3 organize_datasets.py --output-dir orig --max-rows 1000
```

---

## 2) (Optional) Generate Tree-sitter syntax-highlight masks

### What is produced?

For an input Verilog file:

```
orig/orig_0000000.v
```

The script writes:

```
orig/orig_0000000.v.tscolors.txt
```

This output is a **byte-aligned mask**: every byte in the source is replaced by a
single ASCII symbol indicating the highlight category (keyword/comment/string/etc.).
Newlines are preserved so the file lines up visually with the original.

### Dependencies

This is known to work with:

* `tree_sitter==0.25.2`
* `tree-sitter-verilog==1.0.3`

Install:

```bash
pip install tree_sitter tree-sitter-verilog
```

### Highlights query (`highlights.scm`)

Tree-sitter parsing does not include highlight rules by default; you must provide a
highlight query file. The easiest option is to reuse Neovim’s `nvim-treesitter`
queries:

```bash
curl -L \
  https://raw.githubusercontent.com/nvim-treesitter/nvim-treesitter/master/queries/verilog/highlights.scm \
  -o highlights.scm
```

⚠️ **Important compatibility note:**
Neovim’s `highlights.scm` may reference node types / field names that do not exist
in the Python `tree-sitter-verilog` grammar. `verilog_ts_colorize.py` automatically:

1. Splits the query into top-level rules
2. Drops rules referencing unknown node types
3. Drops rules referencing unknown field names
4. Drops rules that fail to compile (e.g. “Impossible pattern”)

This makes the query usable even when grammar versions differ.

### Run on one file

```bash
python3 verilog_ts_colorize.py orig/orig_0000000.v \
  --highlights ./highlights.scm \
  --prefer-longest \
  --verbose-filter
```

### Useful flags

```bash
python3 verilog_ts_colorize.py --help
```

Common options:

* `--highlights PATH` : required highlight query file (`highlights.scm`)
* `--prefer-longest` : overlap heuristic (longer spans fill first)
* `--symbol-map JSON` : override mapping from capture families to 1-char symbols
* `--keep-unfiltered-query` : skip filtering (may fail if query/grammar mismatch)
* `--verbose-filter` : print which query rules were dropped and why

---

## License

PyraNet-Verilog is licensed under **CC BY-NC-SA 4.0**.
Refer to the dataset card for full terms and ensure your usage complies with the license.

