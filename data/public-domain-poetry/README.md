# Scripts compatible with Public-domain-poetry

This directory contains scripts compatible with public-domain-poetry.

## Dataset license

cc0-1.0

## Huggingface Website

* https://huggingface.co/datasets/DanFosing/public-domain-poetry

The raw downloaded JSON retains `Author`, `Title`, and `text`; `include_keys`
in `get_dataset.sh` selects fields for emitted text only.

`bash data/public-domain-poetry/run_extract.sh [INPUT_JSON] [OUTPUT_TEXT]`
creates one exact output filename and fails if no records are accepted. For
sharded output, invoke `json_poetry_to_espeak_text.py` directly; its default
10 MB outputs are named `poems_espeak_0001.txt`, etc. Use a fresh output prefix
for each sharded run so older shards cannot be mistaken for current output.
`--reject-report` explains rejected records; `--allow-empty` is explicit opt-in.

Use `inspect_characters.py FILE` to list characters. `split_wav.sh INPUT_WAV`
segments into approximately ten pieces at FFmpeg packet boundaries; failures
return nonzero. The separate Shakespeare FLAC splitter takes an input argument
and checks resulting byte sizes rather than promising exact compressed sizes.
