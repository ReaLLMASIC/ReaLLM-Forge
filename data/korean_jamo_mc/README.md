# Korean lightweight jamo multicontext dataset

This is a smaller Korean multicontext dataset layout intended for quick experiments. Instead of the full 23-lane Hangul factorization, it uses four aligned lanes:

1. `char`: the original rendered character stream.
2. `first_jamo`: the first jamo of each modern NFC Hangul syllable, or `_` for non-Hangul/non-NFC characters.
3. `last_jamo`: the final jamo when a syllable has batchim, otherwise the vowel jamo, or `_` for non-Hangul/non-NFC characters.
4. `eun_neun`: `은` when the Hangul syllable has a final consonant, `는` when it does not, or `_` for non-Hangul/non-NFC characters.

`get_dataset.sh` downloads OPUS-100 English-Korean data through Hugging Face `datasets` when available, falls back to the included tiny corpus otherwise, extracts the four lanes with `../template/utils/korean/extract_jamo_lite_streams.py`, and prepares each lane for multicontext training.

For sampling, rendered prompts must be converted into per-lane `.bin` start files with `../template/utils/korean/make_jamo_lite_prompt.py`; the special lanes are not raw English text lanes.
