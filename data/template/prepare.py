# prepare.py
import json
import os
import argparse
import numpy as np
from tokenizers import (
    NumericRangeTokenizer,
    SentencePieceTokenizer,
    TiktokenTokenizer,
    CustomTokenizer,
    CharTokenizer,
    CustomCharTokenizerWithByteFallback,
    JsonByteTokenizerWithByteFallback,
)
from tqdm import tqdm
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser(description="Tokenize text data using different methods.")
    parser.add_argument("--tokens_file", type=str, default=None, help="Path to the file containing newline-separated tokens for tokenization")
    parser.add_argument("--method", type=str, choices=["sentencepiece", "tiktoken", "char", "custom", "custom_char_byte_fallback", "numeric_range", "json_byte_fallback"], default="tiktoken", help="Tokenization method")
    parser.add_argument("--train_input", type=str, required=True, help="Path to the training input file")
    parser.add_argument("--train_output", type=str, required=True, help="Path to save the training output file")
    parser.add_argument("--val_input", type=str, help="Path to the validation input file")
    parser.add_argument("--val_output", type=str, help="Path to save the validation output file")
    # SentencePiece only arguments
    parser.add_argument("--vocab_size", type=int, default=500, help="Vocabulary size for SentencePiece model")
    parser.add_argument("--spm_model_file", type=str, default=None, help="Path to the pre-trained SentencePiece model file")
    parser.add_argument("--spm_vocab_file", type=str, default=None, help="Path to the SentencePiece vocabulary file")
    parser.add_argument("--skip_tokenization", action="store_true", help="Skip creation of .bin files")
    # Tiktoken only arguments
    parser.add_argument("-e", "--tiktoken_encoding", choices=["gpt2", "r50k_base", "p50k_base", "cl100k_base"], default="gpt2", help="Version of tiktoken encoding to utilize")
    parser.add_argument("--additional_tokens_file", type=str, default=None, help="Path to JSON file containing additional special tokens for tiktoken (format: {'token': id})")
    # Char only arguments
    parser.add_argument("--reuse_chars", action="store_true", help="Reuse character list")
    # Add argument for custom characters file
    parser.add_argument("--custom_chars_file", type=str, default=None, help="Path to the file containing custom characters for the tokenizer")
    # Add argument for tracking token counts
    parser.add_argument("--track_token_counts", action="store_true", help="Track how often each token appears and store in meta.pkl")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load training data
    with open(args.train_input, 'r') as f:
        train_data = f.read()

    # Load validation data if provided
    val_data = None
    if args.val_input:
        with open(args.val_input, 'r') as f:
            val_data = f.read()

    # Initialize tokenizer based on method
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

    # Determine dtype based on token IDs
    max_token_id = max(max(train_ids), max(val_ids) if val_ids else 0)
    dtype = np.uint32 if max_token_id > 65535 else np.uint16

    save_tokens(train_ids, args.train_output, dtype)
    if val_data and val_ids:
        save_tokens(val_ids, args.val_output, dtype)

    # Save additional metadata for tiktoken
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

