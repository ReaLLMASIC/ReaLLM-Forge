# Hanzi Factor: setup and end-to-end demo

This demo starts from a fresh Python environment and exercises the complete
document workflow:

1. Create an isolated virtual environment.
2. Install `hanzi-factor` with the OpenCC normalization extra.
3. Download and SHA-512-verify the pinned CCD decomposition catalogue.
4. Normalize one mixed Chinese document to Simplified Chinese.
5. Normalize it separately to Taiwan Traditional Chinese with phrase vocabulary.
6. Replace every covered Han character with recursively expanded prefix IDS.
7. Parse both mixed IDS/plain-text streams back into ordinary documents.
8. Verify both document round trips are byte-identical.
9. Save forward and inverse statistics for both converted documents.
10. Demonstrate Hanzi → binary tree → Hanzi reconstruction.
11. Run the complete automated test suite.

## Requirements

- Python 3.10 or newer with the `venv` module.
- Internet access on the first run for the Python dependency and pinned npm
  catalogue. Subsequent runs reuse the downloaded catalogue and virtual
  environment.

## One-command demo

From the extracted `hanzi-factor` source directory:

```bash
bash demos/setup_and_demo.sh
```

The default workspace is `out/hanzi_factor_demo`. To put it elsewhere, pass a
directory as the first argument:

```bash
bash demos/setup_and_demo.sh /tmp/my_hanzi_demo
```

You can also choose a Python interpreter or virtual-environment location:

```bash
PYTHON_BIN="$HOME/miniconda3/envs/torch/bin/python" \
VENV_DIR=/tmp/hanzi-factor-venv \
bash demos/setup_and_demo.sh /tmp/my_hanzi_demo
```

## Input document

The included [`sample_chinese.txt`](sample_chinese.txt) deliberately contains a
known Simplified Chinese source together with English, numbers, punctuation,
and an emoji. Starting from one known source profile avoids ambiguous treatment
of a mixed-orthography input while still showing phrase-sensitive conversion
and preservation of non-Chinese text.

## Generated results

The script writes these files beneath `out/hanzi_factor_demo/results/`:

| File | Contents |
|---|---|
| `sample.simplified.txt` | Entire sample normalized to Simplified Chinese |
| `sample.traditional-tw.txt` | Entire sample normalized to Taiwan Traditional Chinese and vocabulary |
| `sample.simplified.ids.txt` | Simplified document with expanded prefix IDS |
| `sample.traditional-tw.ids.txt` | Traditional document with expanded prefix IDS |
| `sample.simplified.restored.txt` | Simplified IDS stream restored to text |
| `sample.traditional-tw.restored.txt` | Traditional IDS stream restored to text |
| `*.ids.report.json` | Match, change, source-issue, and uncovered-character counts |
| `*.restored.report.json` | Decoded root, escape, ambiguity, and pass-through counts |
| `roundtrip.txt` | Structural/binary round trips for `汉 国 语 清 森` |

The IDS converter uses `--on-uncovered escape`, so a missing decomposition is
visible as `<U+XXXX>` instead of being silently retained. A graphical primitive
may remain the same scalar—for example, an atomic leaf is already a valid
one-node IDS. Every multi-component expression is serialized in prefix order;
operator arity supplies its tree boundaries without parentheses.

## Run individual stages manually

After the demo has created its environment:

```bash
DEMO_PY=out/hanzi_factor_demo/.venv/bin/python
CCD=out/hanzi_factor_demo/data/ccd.json

"$DEMO_PY" scripts/normalize_chinese.py demos/sample_chinese.txt \
  --to traditional --variant taiwan-phrases -o /tmp/sample.tw.txt

"$DEMO_PY" scripts/text_to_ids.py /tmp/sample.tw.txt \
  --ccd "$CCD" --format expanded --on-uncovered escape \
  -o /tmp/sample.tw.ids.txt

"$DEMO_PY" scripts/ids_to_text.py /tmp/sample.tw.ids.txt \
  --ccd "$CCD" -o /tmp/sample.tw.restored.txt

cmp /tmp/sample.tw.txt /tmp/sample.tw.restored.txt
```

Simplified/Traditional conversion is linguistic and not bijective: distinct
traditional forms can collapse to one simplified form. Keep the original when
exact editorial identity matters. IDS factorization is a separate graphical
operation and uses the pinned catalogue as its structural contract.
