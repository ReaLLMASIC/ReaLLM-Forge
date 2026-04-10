# Tokenization

This folder is a template data folder, with a script provides a utility for
tokenizing text data using alternative methods.

Currently SentencePiece, TikToken, and character-level tokenizations are
supported, with more tokenization method support planned.

Additionally, phonemization using espeak via a shell script is supported to
preprocess text data into phoneme representations.

Whisper-style mel spectrogram CSV export is also supported for audio inputs.

## Currently Supported Tokenizations

- **SentencePiece Tokenization**
- **TikToken Tokenization**
- **Character-Level Tokenization**
- **Whisper-style Mel Spectrogram CSV Export**

## Usage

### Prerequisites

Ensure you have Python installed on your system along with the necessary
libraries: `numpy`, `pickle`, `sentencepiece`, and `tiktoken`.

Also for phonemization ensure that `espeak` and `GNU Parallel` are installed.

For Whisper-style mel CSV export and visualization, install `torch`,
`torchaudio`, and `matplotlib`.

##### 1. Create a New Data Folder

```bash
# from the ./data directory
bash create_new_dataset.sh <name-of-dataset-folder>
```

##### 2. Add data to folder

Obtain a text format of the data for training.

Note: make sure not to check in only the scripts and instructions, not the dataset.

Note: For parquet datasets from urls see the `get_dataset.sh` script, which may
simplify the process of downloading the dataset.

##### 2a. (if parquet dataset) get_dataset.sh setup

Use an editor follow the instrudtions within the `get_dataset.sh`.

After modification to the proper values for the dataset, run:

```bash
bash get_dataset.sh
```

Which should bring necessary files (parquet -> json -> text) into the present folder.

Again, feel free to modify or even fully replace the `get_dataset.sh` script with your own script.

Other scripts may expect the file to be called `get_dataset.sh` so please title
your script with this naming convention.

### 3. Run tokenization script

Finally, run `prepare.py` script to process the dataset for training.

#### Examples:

##### SentencePiece

```bash
python3 prepare.py -t input.txt --method sentencepiece --vocab_size 1000
```

##### TikToken

```bash
python3 prepare.py -t input.txt --method tiktoken
```

##### Character Level Tokenization

This command will tokenize the text in from the input file at the character level.

```bash
python3 prepare.py -t input.txt --method char
```

##### Custom

```bash
python3 prepare.py -t input.txt --method custom --tokens_file phoneme_list.txt
```

##### Custom with Byte Fallback

```bash
python3 prepare.py -t input.txt --method custom_char_byte_fallback --custom_chars_file tokens.txt
```


##### JSON-Constrained BPE with Byte Fallback

```bash
python3 prepare.py -t input.txt --method json_bpe_byte_fallback --json_tokens_file tokens.json --vocab_size 1200
```

`--vocab_size` sets the target total vocab size (including 256 byte tokens). Merges are added until this target is reached or no additional merge pairs exist where both sides are in the current merge vocabulary (which starts from the JSON list and grows with learned merges). Newly-created merged tokens do **not** need to already exist in the JSON array.

After tokenization this writes inspection artifacts similar to `char_bpe`:
- `json_bpe_vocab.json` (token list by ID, including `<byte:N>` entries)
- `json_bpe_token_counts.json` (when `-T/--track_token_counts` is enabled)

##### Whisper-style Mel Spectrogram CSV Export

This emits a CSV file where each row is a time frame and each column is a mel
channel. Defaults match Whisper/whisper.cpp (16 kHz, 80 mel channels, 25 ms
window, 10 ms hop).

```bash
python3 prepare.py \
  --method whisper_mel_csv \
  --train_input sample.wav \
  --train_output sample.csv
```

For convenience, see `run_whisper_mel_csv_examples.sh` for mp3/wav/flac
examples.

To visualize the CSV output:

```bash
python3 visualize_whisper_mel_csv.py sample.csv --output sample.png
```

To reconstruct a WAV file from the CSV (approximate inversion):

```bash
python3 mel_csv_to_wav.py sample.csv --output reconstructed.wav
```

#### Expected Value Range

When `--mel_normalize` is enabled (default), values follow Whisper's log-mel
normalization:

1. Convert mel spectrogram to log10.
2. Clamp to `max - 8`.
3. Scale as `(log_mel + 4) / 4`.

This yields values typically in the `~[0, 1]` range after normalization (values
can be slightly outside this range depending on the audio content and scale).

##### Converting to int16 Range

If you need a fixed-point representation, you can linearly scale and clamp the
normalized values:

```python
scaled = np.clip(mel, 0.0, 1.0)
int16 = (scaled * 32767).astype(np.int16)
```

To recover approximate floats later:

```python
mel = int16.astype(np.float32) / 32767.0
```

Note that this is a lossy quantization step and will reduce precision.

### Additional details about the `prepare.py` script

#### `prepare.py` Command Line Arguments

This script takes in a text file as its argument for tokenization, and
additional parameters for tokenization strategy and their parameters.

- `input_file`: Path to the input text file.
- `--method`: Tokenization method (`sentencepiece`, `tiktoken`, `char`). Default is `sentencepiece`.
- `--vocab_size`: Vocabulary size for the SentencePiece model. Default is 500.

#### `prepare.py` Generated File Descriptions

Afterward it produces the train.bin and val.bin (and meta.pkl if not tiktoken)
* `train.bin` - training split containing 90% of the input file
* `val.bin` - validation split containing 10% of the input file
* `meta.pkl` - additonal file specifying tokenization needed for training and inference (Note: not produced/required for tiktoken)

These files above are then utilized to train the model via the `train.py` or
`run_experiments.py` wrapper scripts.

### Compare vocabularies from meta.pkl files

Use the Textual-based TUI to compare two token vocabularies side-by-side and
sort both lists by byte length or tracked frequency (requires `-T` when running
`prepare.py`).

```bash
python3 compare_meta_vocab_tui.py /path/to/first/meta.pkl /path/to/second/meta.pkl
```

### (Optional) Pre-processing of input.txt

There are a number of methods to preprocess data before tokenization.

#### Phonemization

To experiment with utilization of a phonemized version of the dataset, first
convert the dataset into text format (e.g. `input.txt`), and run the
phonemization script on the text file:

```bash
bash txt_to_phonemes.sh input.txt phonemized_output.txt
```

The above by default utilizes all of the cores of one's machine to speed up the
process.

To specify the number of cores to utilize, use the following invocation, setting
the number of cores to limit the script to:

```bash
bash txt_to_phonemes.sh -n 8 input.txt phonemized_output.txt
```

The above command will limit the script to utilize only 8 cores at any one time.

## Relevant Tokenization Resources and References

This section provides links to research papers and GitHub repositories related to the tokenization methods used in this script. These resources can offer deeper insights into the algorithms and their implementations.

### SentencePiece

- [Read the Paper](https://arxiv.org/abs/1808.06226)
- [SentencePiece Github Repository](https://github.com/google/sentencepiece)

### TikToken

- [General Usage Guide](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)
- [TikToken Github Repository](https://github.com/openai/tiktoken)

## Open to Contributions

- [ ] Add feature to take in a file with set of multi-character tokens for custom tokenization (e.g. char level tokenization but custom word-level tokenization list)
- [ ] Add byte-level tokenization options
- [ ] Add argparse arguments for more features of SentencePiece and TikToken
