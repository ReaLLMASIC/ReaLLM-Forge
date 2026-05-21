# Readable symbolic compact format for reversible Han files

This format is the human-readable sibling of the earlier compact container.

It is designed to be:

- **reversible**: you can recover the exact original `input.txt` bytes
- **readable**: the file stays line-oriented text with a legend and a visible body
- **compact-ish**: repeated Han characters become short symbol IDs instead of full JSON token records

## What the format looks like

A file starts with metadata, then a legend, then a wrapped symbol body:

```text
HAN-READABLE-SYMBOLIC/v1
meta	{"schema_version":"han-readable-symbolic/v1",...}
stats	{"token_count":1234,...}
legend_begin
symbol	{"symbol_id":"←","original_text":"河","serialized_text":"水 丁 口",...}
symbol	{"symbol_id":"↑","original_text":"語","serialized_text":"言 五 口",...}
legend_end
body_begin
|¤←¤↑A ¤←¤~n¤↑
```

Interpretation:

- `¤←` means “token mapped by symbol ID `←`”
- the legend tells you that `←` maps back to original `河`
- `serialized_text` shows the transformed/decomposition rendering for that symbol
- `¤~n`, `¤~r`, `¤~t`, and `¤~s` are visible escapes for newline, carriage return, tab, and the sentinel itself

## Files

- `han_file_symbolic_serialize.py` — forward transform into the readable symbolic format
- `han_file_symbolic_reverse.py` — reverse recovery, inspection, and compare
- `han_file_symbolic_common.py` — shared format/parsing helpers

## Typical workflow

### 1) Build the Han decomposition dataset

```bash
python3 han_main_block_decomp.py build \
  --data-dir data \
  --download-missing \
  --decomp-source cjkvi \
  --normalization conservative \
  --output data/han_main_block.jsonl
```

### 2) Create the readable symbolic file directly from `input.txt`

```bash
python3 han_file_symbolic_serialize.py build \
  --input data/input.txt \
  --dataset data/han_main_block.jsonl \
  --output data/input.readable.hsym
```

Useful options:

```bash
python3 han_file_symbolic_serialize.py build \
  --input data/input.txt \
  --dataset data/han_main_block.jsonl \
  --output data/input.readable.hsym \
  --mode decomp-normalized \
  --component-separator " " \
  --sentinel "¤" \
  --wrap-width 120
```

### 3) Or create it from an existing mapped file

```bash
python3 han_file_symbolic_serialize.py from-map \
  --map data/mapped.json \
  --output data/input.readable.hsym
```

### 4) Recover the exact original file

```bash
python3 han_file_symbolic_reverse.py recover \
  --symbolic data/input.readable.hsym \
  --mode original-bytes \
  --output data/recovered_input.txt
```

### 5) Compare recovered output against the original

```bash
python3 han_file_symbolic_reverse.py compare \
  --symbolic data/input.readable.hsym \
  --mode original-bytes \
  --compare-to data/input.txt \
  --report-json data/readable_compare_report.json
```

## Recovery modes

`han_file_symbolic_reverse.py recover` supports:

- `original-bytes` — exact original file bytes
- `original-text` — reconstructed original Unicode text
- `serialized-text` — the transformed/decomposition text view
- `symbolic-body` — the raw visible symbol stream stored in the container

Examples:

```bash
python3 han_file_symbolic_reverse.py recover \
  --symbolic data/input.readable.hsym \
  --mode serialized-text \
  --output data/serialized_view.txt

python3 han_file_symbolic_reverse.py inspect \
  --symbolic data/input.readable.hsym \
  --include-previews
```

## Notes

- By default, **Han main-block tokens are symbolized**, while safe non-Han literals stay inline for readability.
- If needed, you can also symbolize non-Han tokens with `--symbolize-non-han`.
- The format stores the source UTF encoding plan, so the original byte stream can be reconstructed exactly.
- If you build from a pre-edited mapped file, any token whose serialized/current text differs from the original is automatically moved into the legend so reversibility is preserved.

## Quick validation

```bash
python3 han_file_symbolic_serialize.py self-test
python3 han_file_symbolic_reverse.py self-test
```
