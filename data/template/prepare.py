# prepare.py
import json
import os
import pickle
import argparse
import numpy as np
from tokenizer_options import (
    NumericRangeTokenizer,
    SentencePieceTokenizer,
    TiktokenTokenizer,
    CustomTokenizer,
    CharTokenizer,
    CustomCharTokenizerWithByteFallback,
    Qwen2Tokenizer,
    JsonByteTokenizerWithByteFallback,
)
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Tokenize text data using different methods.")
    parser.add_argument("--tokens_file", type=str, default=None, help="Path to the file containing newline-separated tokens for tokenization")
    parser.add_argument("--method", type=str, choices=["sentencepiece", "tiktoken", "char", "custom", "custom_char_byte_fallback", "numeric_range", "json_byte_fallback", "qwen2"], default="tiktoken", help="Tokenization method")
    # SentencePiece only arguments
    parser.add_argument("--vocab_size", type=int, default=500, help="Vocabulary size for SentencePiece model")
    parser.add_argument("--spm_model_file", type=str, default=None, help="Path to the pre-trained SentencePiece model file")
    parser.add_argument("--spm_vocab_file", type=str, default=None, help="Path to the SentencePiece vocabulary file")
    parser.add_argument("--skip_tokenization", action="store_true", help="Skip creation of .bin files")
    # Tiktoken only argument
    parser.add_argument("-e", "--tiktoken_encoding", choices=["gpt2", "r50k_base", "p50k_base", "cl100k_base"], default="gpt2", help="Version of tiktoken encoding to utilize")
    # Char only arguments
    parser.add_argument("--reuse_chars", action="store_true", help="Reuse character list")
    # Add argument for custom characters file
    parser.add_argument("--custom_chars_file", type=str, default=None, help="Path to the file containing custom characters for the tokenizer")
    # Customize output names for bins
    parser.add_argument("--train_output", type=str, default="train.bin", help="Output file for tokenized training data")
    parser.add_argument("--val_output", type=str, default="val.bin", help="Output file for tokenized validation data")
    # Options for using separate training and validation input files
    parser.add_argument("-s", "--use_separate_files", action="store_true", help="Use separate files for training and validation input")
    parser.add_argument("-t", "--train_input", type=str, help="Path to the training input text file")
    parser.add_argument("-v", "--val_input", type=str, help="Path to the validation input text file")
    parser.add_argument("-p", "--percentage_train", type=float, default=0.9, help="Value between 0 and 1.0 for train percentage split")
    # Numeric range tokenizer arguments
    parser.add_argument("--numeric_range", action="store_true", help="Use numeric range tokenization method")
    parser.add_argument("--min_token", type=int, default=0, help="Minimum value for numeric tokens")
    parser.add_argument("--max_token", type=int, default=65535, help="Maximum value for numeric tokens")
    # Qwen2 tokenizer arguments
    parser.add_argument("--qwen2_model",
        type=str,
        choices=['qwen2_0p5b', 'qwen2_1p5b', 'qwen2_7b'],
        default="qwen2_1p5b",
        help="Model for qwen2 tokenization",
    )
    # tokenizer counts
    parser.add_argument("--json_tokens_file", type=str, default=None, help="Path to the JSON file containing an array of tokens for json_byte_fallback mode")
    parser.add_argument("-T", "--track_token_counts", action="store_true", help="Track how often each token appears and store in meta.pkl")
    return parser.parse_args()


def save_args(args, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

def main():
    args = parse_arguments()
    os.makedirs('out', exist_ok=True)
    save_args(args, "out")

    # Read data
    if args.use_separate_files:
        if not args.train_input or not args.val_input:
            raise ValueError(
                "Both --train_input and --val_input must be provided when using --use_separate_files."
            )
        with open(args.train_input, "r") as f:
            train_data = f.read()
        with open(args.val_input, "r") as f:
            val_data = f.read()
    else:
        if not args.train_input:
            raise ValueError(
                "You must provide --train_input when not using --use_separate_files."
            )
        with open(args.train_input, "r") as f:
            data = f.read()
        n = len(data)
        split_idx = int(n * args.percentage_train)
        train_data = data[:split_idx]
        val_data = data[split_idx:] if args.percentage_train < 1.0 else None

    # Select tokenizer
    tokenizer = None
    if args.method == "numeric_range":
        tokenizer = NumericRangeTokenizer(args)
    elif args.method == "sentencepiece":
        tokenizer = SentencePieceTokenizer(args, input_files=args.train_input)
    elif args.method == "tiktoken":
        tokenizer = TiktokenTokenizer(args)
    elif args.method == "custom":
        tokenizer = CustomTokenizer(args)
    elif args.method == "char":
        tokenizer = CharTokenizer(args, train_data, val_data)
    elif args.method == "custom_char_byte_fallback":
        tokenizer = CustomCharTokenizerWithByteFallback(args)
    elif args.method == "qwen2":
        tokenizer = Qwen2Tokenizer(args)
    elif args.method == "json_byte_fallback":
        tokenizer = JsonByteTokenizerWithByteFallback(args)
    else:
        raise ValueError(f"Unknown tokenization method: {args.method}")

    # Tokenize data
    train_ids = tokenizer.tokenize(train_data)
    if val_data:
        val_ids = tokenizer.tokenize(val_data)
    else:
        val_ids = None

    # Save tokenized data with progress bar
    def save_tokens(ids, output_file, dtype):
        total = len(ids)
        batch_size = 1024 * 1024  # 1 million tokens per batch
        with open(output_file, 'wb') as f_out:
            for i in tqdm(range(0, total, batch_size), desc=f"Saving {output_file}"):
                batch = ids[i:i+batch_size]
                np.array(batch, dtype=dtype).tofile(f_out)

    if os.path.exists("meta.pkl"):
        with open("meta.pkl", "rb") as f:
            meta = pickle.load(f)

        # Check the vocab size in the loaded meta
        if meta["vocab_size"] > 65536:
            dtype = np.uint32
        else:
            dtype = np.uint16
    else:
        raise FileNotFoundError("A meta.pkl file was not created properly. Please ensure it exists in the current directory.")

    save_tokens(train_ids, args.train_output, dtype)
    if val_data and val_ids:
        save_tokens(val_ids, args.val_output, dtype)


if __name__ == "__main__":
    main()

