# prepare.py
import json
import os
import argparse
import numpy as np
from tokenizers import (
    SentencePieceTokenizer,
    TiktokenTokenizer,
    CustomTokenizer,
    ByteTokenizer,
    CharTokenizer,
    CustomCharTokenizerWithByteFallback,
    JsonByteTokenizerWithByteFallback,
    PythonProgrammingTokenizer,
    SineWaveTokenizer,
)
from tqdm import tqdm
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser(description="Tokenize text data using different methods.")

    # Input/output arguments
    parser.add_argument("-t", "--train_input", type=str, required=True, help="Path to the input text file")
    parser.add_argument("-v", "--val_input", type=str, help="Path to validation input file. If not provided, train_input will be split using percentage_train")
    parser.add_argument("--train_output", type=str, default="train.bin", help="Path to save the training output file")
    parser.add_argument("--val_output", type=str, default="val.bin", help="Path to save the validation output file")
    parser.add_argument("-p", "--percentage_train", type=float, default=0.9, help="Percentage of data to use for training (between 0 and 1) when val_input is not provided")

    # Tokenizer selection and configuration
    parser.add_argument("--method", type=str,
                       choices=["sentencepiece", "tiktoken", "char", "custom", "byte", "custom_char_byte_fallback", "json_byte_fallback", "python_programming", "sinewave"],
                       default="tiktoken", help="Tokenization method")

    # Sine wave tokenizer arguments
    parser.add_argument("--sine_period", type=float, default=1.0,
                        help="Period multiplier applied to the sine wave (in radians)")
    parser.add_argument("--sine_points_per_period", type=int, default=64,
                        help="Number of discrete points sampled per sine wave period")
    parser.add_argument("--sine_num_periods", type=int, default=10,
                        help="Total number of periods to generate")
    parser.add_argument("--sine_amplitude", type=float, default=50.0,
                        help="Amplitude of the generated sine wave prior to clamping")

    # SentencePiece arguments
    parser.add_argument("--vocab_size", type=int, default=500, help="Vocabulary size for SentencePiece model")
    parser.add_argument("--spm_model_file", type=str, default=None, help="Path to the pre-trained SentencePiece model file")
    parser.add_argument("--spm_vocab_file", type=str, default=None, help="Path to the SentencePiece vocabulary file")
    parser.add_argument("--skip_tokenization", action="store_true", help="Skip creation of .bin files")

    # Tiktoken arguments
    parser.add_argument("-e", "--tiktoken_encoding",
                       choices=["gpt2", "r50k_base", "p50k_base", "cl100k_base"],
                       default="gpt2", help="Version of tiktoken encoding to utilize")
    parser.add_argument("--additional_tokens_file", type=str, default=None,
                       help="Path to JSON file containing additional special tokens for tiktoken (format: {'token': id})")

    # Char tokenizer arguments
    parser.add_argument("--reuse_chars", action="store_true", help="Reuse character list from meta.pkl")

    # Custom tokenizer arguments
    parser.add_argument("--tokens_file", type=str, default=None, help="Path to the file containing newline-separated tokens for tokenization")
    parser.add_argument("--custom_chars_file", type=str, default=None, help="Path to the file containing custom characters for the tokenizer")
    parser.add_argument("--json_tokens_file", type=str, default=None, help="Path to JSON file containing tokens for json_byte_fallback tokenizer")

    # Additional options
    parser.add_argument("-T", "--track_token_counts", action="store_true", help="Track how often each token appears and store in meta.pkl")
    parser.add_argument("-E", "--extract-vocab", action="store_true", help="Export the tokenizer vocabulary to a JSON file")

    return parser.parse_args()

def save_tokens(ids, output_file, dtype):
    """Save tokenized data to a binary file with progress bar."""
    total = len(ids)
    batch_size = 1024 * 1024  # 1 million tokens per batch
    with open(output_file, 'wb') as f_out:
        for i in tqdm(range(0, total, batch_size), desc=f"Saving {output_file}"):
            batch = ids[i:i+batch_size]
            np.array(batch, dtype=dtype).tofile(f_out)

def _read_input_data(path):
    if os.path.isdir(path):
        collected = []
        for root, _, files in os.walk(path):
            for name in sorted(files):
                file_path = os.path.join(root, name)
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    collected.append(f.read())
        return "\n".join(collected)
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def main():
    args = parse_arguments()

    # Load training/validation data depending on tokenizer method
    if args.method == "sinewave":
        train_data = None
        val_data = None
    else:
        train_data = _read_input_data(args.train_input)

        if args.val_input:
            val_data = _read_input_data(args.val_input)
        else:
            n = len(train_data)
            train_data, val_data = train_data[:int(n * args.percentage_train)], train_data[int(n * args.percentage_train):]
            if args.percentage_train == 1.0:
                val_data = None

    # Initialize tokenizer based on method
    if args.method == "sentencepiece":
        tokenizer = SentencePieceTokenizer(args, input_files=args.train_input)
    elif args.method == "tiktoken":
        tokenizer = TiktokenTokenizer(args)
    elif args.method == "custom":
        tokenizer = CustomTokenizer(args)
    elif args.method == "byte":
        tokenizer = ByteTokenizer(args)
    elif args.method == "char":
        tokenizer = CharTokenizer(args, train_data, val_data)
    elif args.method == "custom_char_byte_fallback":
        tokenizer = CustomCharTokenizerWithByteFallback(args)
    elif args.method == "json_byte_fallback":
        tokenizer = JsonByteTokenizerWithByteFallback(args)
    elif args.method == "python_programming":
        tokenizer = PythonProgrammingTokenizer(args)
    elif args.method == "sinewave":
        tokenizer = SineWaveTokenizer(args)
    else:
        raise ValueError(f"Unknown tokenization method: {args.method}")

    # Tokenize data
    train_ids = tokenizer.tokenize(train_data)
    if args.method == "tiktoken":
        print(f"[tiktoken] Total train tokens: {tokenizer.last_token_count:,}")
    if args.method == "sinewave" and args.val_input is None:
        split_point = int(len(train_ids) * args.percentage_train)
        val_ids = train_ids[split_point:]
        train_ids = train_ids[:split_point]
    elif val_data is not None:
        val_ids = tokenizer.tokenize(val_data)
        if args.method == "tiktoken":
            print(f"[tiktoken] Total val tokens: {tokenizer.last_token_count:,}")
    else:
        val_ids = None

    # Determine dtype based on vocabulary size from meta.pkl
    if args.method == "sinewave":
        dtype = np.uint16
    else:
        with open("meta.pkl", "rb") as f:
            meta = pickle.load(f)
        vocab_size = meta["vocab_size"]
        dtype = np.uint32 if vocab_size > 65535 else np.uint16

    # Ensure output directories exist if paths include folders
    for output_path in [args.train_output, args.val_output]:
        if output_path:
            out_dir = os.path.dirname(output_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

    # Save tokenized data
    save_tokens(train_ids, args.train_output, dtype)
    if val_ids is not None:
        save_tokens(val_ids, args.val_output, dtype)

    if args.extract_vocab:
        if not hasattr(tokenizer, "get_vocabulary"):
            raise ValueError(f"Vocabulary extraction is not supported for tokenizer method '{args.method}'.")

        vocab = tokenizer.get_vocabulary()
        # Ensure string representations for all tokens
        vocab_strings = [token if isinstance(token, str) else str(token) for token in vocab]
        vocab_strings.sort(key=lambda token: (-len(token), token))

        vocab_filename = f"{args.method}_vocab.json"
        with open(vocab_filename, 'w', encoding='utf-8') as f:
            json.dump(vocab_strings, f, ensure_ascii=False, indent=2)
        print(f"Saved vocabulary to {vocab_filename}")

    if args.method == "sinewave":
        meta = {
            "tokenizer": "sinewave",
            "vocab_size": 256,
            "sine_period": args.sine_period,
            "sine_points_per_period": args.sine_points_per_period,
            "sine_num_periods": args.sine_num_periods,
            "sine_amplitude": args.sine_amplitude,
        }
        with open("meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    # Save additional metadata for tiktoken if needed
    if args.method == "tiktoken" and args.additional_tokens_file:
        with open(args.additional_tokens_file, 'r') as f:
            additional_tokens = json.load(f)
        with open("meta.pkl", "rb") as f:
            meta = pickle.load(f)
        meta.update({
            "has_additional_tokens": True,
            "special_tokens": additional_tokens,
            "tokenizer": "tiktoken",
            "tiktoken_encoding": args.tiktoken_encoding
        })
        with open("meta.pkl", "wb") as f:
            pickle.dump(meta, f)

if __name__ == "__main__":
    main()

