# Korean multicontext Hangul factor dataset with POS tagging (24 lanes)

This dataset layout is intended for English-to-Korean experiments using the Korean side of OPUS-100 English-Korean parallel data, factorizing Korean Hangul syllables into 24 lanes (the 23 structural features plus a part-of-speech tag lane from `kiwipiepy`).

## Source and license

`get_dataset.sh` downloads OPUS-100 English-to-Korean data and extracts the Korean target text into `input.txt`. OPUS-100 is a multilingual parallel corpus derived from OPUS collections; users should review the OPUS-100 dataset card and the licenses of the underlying OPUS corpora before redistribution or commercial use.

## Pipeline

1. Download English-Korean OPUS-100 parallel data (or use fallback corpus).
2. Extract Korean target segments into `input.txt`.
3. Run `../template/utils/korean/extract_multicontext_streams.py input.txt . --use-pos --metadata-json '' --metadata-yaml ''`.
4. The extractor streams through `input.txt` in chunks, using `kiwipiepy` to tag part-of-speech (POS) and factorizing each syllable into 24 lanes.
5. The extractor writes one aligned `input.txt` stream for each of the 24 Hangul factor lanes plus `char/input.txt` containing the original character stream.
6. Run `../template/prepare.py --method char -s -S <lane_name>` for every lane directory, then copy the generated `char_<lane_name>/meta.pkl`, `train.bin`, and `val.bin` to the lane directory root because multicontext loading expects `data/korean_pos_mc/<lane>/meta.pkl`.

## The 24 Lanes

The 24 factor lanes are:
`script`, `choseong`, `jungseong`, `jongseong`, `jung_base1`, `jung_base2`, `jung_has_w`, `jung_has_y`, `jung_has_i`, `jong_base1`, `jong_base2`, `jong_base3`, `choseong_tense`, `choseong_aspirated`, `choseong_nasal_liquid`, `choseong_place`, `jung_height`, `jung_backness`, `jung_round`, `jong_complex`, `has_batchim`, `syllable_index_mod`, `codepoint_mod`, and `pos`.

Non-Hangul characters are preserved in `char/input.txt` and the metadata sidecars. Their feature lanes use `NON_HANGUL` in the `script` lane and `PAD` markers elsewhere (except in `pos`, where Kiwi tags punctuation/alphanumeric characters).

## Sampling prompts

Lane datasets use private-use factor tokens, so rendered text such as `English: Hello Korean: ` should not be passed directly to every lane with `--multicontext_start`. Use `../template/utils/korean/make_multicontext_prompt.py` with `--use-pos` after running `get_dataset.sh` to encode the rendered prompt into per-lane `.bin` start files, then pass those files to `sample.py` with `--multicontext_start_files`.
