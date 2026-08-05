# Korean multicontext Hangul factor dataset

This dataset layout is intended for English-to-Korean experiments using the Korean side of OPUS-100 English-Korean parallel data.

## Source and license

`get_dataset.sh` downloads OPUS-100 English-to-Korean data and extracts the Korean target text into `input.txt`. OPUS-100 is a multilingual parallel corpus derived from OPUS collections; users should review the OPUS-100 dataset card and the licenses of the underlying OPUS corpora before redistribution or commercial use.

## Pipeline

1. Download English-Korean OPUS-100 parallel data.
2. Extract Korean target segments into `input.txt`.
3. Run `../template/utils/korean/extract_multicontext_streams.py input.txt .`.
4. The extractor streams through `input.txt` in chunks, so it does not keep the full corpus, all lane streams, or all per-character metadata in memory.
5. The extractor writes one aligned `input.txt` stream for each of the 23 Hangul factor lanes plus `char/input.txt` containing the original character stream.
6. Run `../template/prepare.py --method char -s -S <lane_name>` for every lane directory, then copy the generated `char_<lane_name>/meta.pkl`, `train.bin`, and `val.bin` to the lane directory root because multicontext loading expects `data/korean_mc/<lane>/meta.pkl`.

Non-Hangul characters are preserved in `char/input.txt` and the metadata sidecars. Their feature lanes use `NON_HANGUL` in the `script` lane and `PAD` markers elsewhere. For very large corpora, pass `--metadata-json '' --metadata-yaml ''` to skip full per-character sidecars and keep only `lane_metadata.json`.

## Sampling prompts

Lane datasets use private-use factor tokens, so rendered text such as `English: Hello Korean: ` should not be passed directly to every lane with `--multicontext_start`. Use `../template/utils/korean/make_multicontext_prompt.py` after running `get_dataset.sh` to encode the rendered prompt into per-lane `.bin` start files, then pass those files to `sample.py` with `--multicontext_start_files`.
