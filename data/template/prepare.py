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
    CharBPETokenizerWithByteFallback,
    CustomCharTokenizerWithByteFallback,
    JsonByteTokenizerWithByteFallback,
    PythonProgrammingTokenizer,
    SineWaveTokenizer,
    WhisperMelCsvTokenizer,
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
                       choices=["sentencepiece", "tiktoken", "char", "char_bpe", "custom", "byte", "custom_char_byte_fallback", "json_byte_fallback", "python_programming", "sinewave", "whisper_mel_csv"],
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

    # Whisper-style mel spectrogram tokenizer arguments
    parser.add_argument("--mel_sample_rate", type=int, default=16000,
                        help="Target sample rate for mel spectrogram computation")
    parser.add_argument("--mel_n_fft", type=int, default=400,
                        help="FFT size for mel spectrogram computation")
    parser.add_argument("--mel_hop_length", type=int, default=160,
                        help="Hop length between frames for mel spectrogram computation")
    parser.add_argument("--mel_win_length", type=int, default=400,
                        help="Window length for mel spectrogram computation")
    parser.add_argument("--mel_n_mels", type=int, default=80,
                        help="Number of mel filterbank channels")
    parser.add_argument("--mel_f_min", type=float, default=0.0,
                        help="Minimum frequency for mel filterbank")
    parser.add_argument("--mel_f_max", type=float, default=8000.0,
                        help="Maximum frequency for mel filterbank")
    parser.add_argument("--mel_center", action=argparse.BooleanOptionalAction, default=True,
                        help="Center frames during STFT computation")
    parser.add_argument("--mel_power", type=float, default=2.0,
                        help="Exponent for the magnitude spectrogram")
    parser.add_argument("--mel_normalize", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply Whisper-style log-mel normalization")
    parser.add_argument("--mel_csv_float_format", type=str, default="%.6f",
                        help="Float format string used when writing mel CSV files")

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
    parser.add_argument("--char_bpe_vocab_path", type=str, default=None,
                        help="Path to a char_bpe meta.pkl to reuse its vocabulary/merges")

    # Custom tokenizer arguments
    parser.add_argument("--tokens_file", type=str, default=None, help="Path to the file containing newline-separated tokens for tokenization")
    parser.add_argument("--custom_chars_file", type=str, default=None, help="Path to the file containing custom characters for the tokenizer")
    parser.add_argument("--json_tokens_file", type=str, default=None, help="Path to JSON file containing tokens for json_byte_fallback tokenizer")

    # Additional options
    parser.add_argument("-T", "--track_token_counts", action="store_true", help="Track how often each token appears and store in meta.pkl")
    parser.add_argument("--report_byte_tokenization", action="store_true",
                        help="Report byte vs non-byte token counts after tokenization (byte-fallback tokenizers only)")
    parser.add_argument("-s", "--output_tokenization_subdir", action="store_true",
                        help="Write meta.pkl/train.bin/val.bin into a subdirectory named after the selected tokenization method")
    parser.add_argument("-S", "--output_subdir_suffix", type=str, default="",
                        help="Optional suffix to append to the tokenization subdirectory name (e.g. sp_1000_suffix)")

    return parser.parse_args()

def save_tokens(ids, output_file, dtype):
    """Save tokenized data to a binary file with progress bar."""
    total = len(ids)
    batch_size = 1024 * 1024  # 1 million tokens per batch
    with open(output_file, 'wb') as f_out:
        for i in tqdm(range(0, total, batch_size), desc=f"Saving {output_file}"):
            batch = ids[i:i+batch_size]
            np.array(batch, dtype=dtype).tofile(f_out)

def save_mel_csv(frames, output_file, float_format):
    with open(output_file, "w", encoding="utf-8") as f_out:
        np.savetxt(f_out, frames, delimiter=",", fmt=float_format)

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

def _reset_byte_token_report(tokenizer):
    if getattr(tokenizer, "has_byte_tokens", False) and hasattr(tokenizer, "reset_byte_token_counts"):
        tokenizer.reset_byte_token_counts()

def _print_byte_token_report(tokenizer, label):
    if getattr(tokenizer, "has_byte_tokens", False) and hasattr(tokenizer, "print_byte_token_report"):
        tokenizer.print_byte_token_report(label)

def _write_byte_token_report(tokenizer, label, report_dir):
    if not report_dir:
        return
    if not (getattr(tokenizer, "has_byte_tokens", False) and hasattr(tokenizer, "get_byte_token_report")):
        return
    report = tokenizer.get_byte_token_report()
    if report is None:
        return
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"byte_token_report_{label}.yaml")
    report_payload = {
        "label": label,
        "byte_tokens": report["byte_tokens"],
        "non_byte_tokens": report["non_byte_tokens"],
        "total_tokens": report["total_tokens"],
        "byte_percentage": round(report["byte_percentage"], 4),
        "non_byte_percentage": round(report["non_byte_percentage"], 4),
    }
    lines = [f"{key}: {value}" for key, value in report_payload.items()]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def main():
    args = parse_arguments()
    output_dir = None
    if args.output_tokenization_subdir:
        if args.method == "json_byte_fallback" and args.json_tokens_file:
            output_dir = os.path.splitext(os.path.basename(args.json_tokens_file))[0]
        elif args.method == "sentencepiece":
            output_dir = f"sp_{args.vocab_size}"
        else:
            output_dir = args.method
        if args.output_subdir_suffix:
            output_dir = f"{output_dir}_{args.output_subdir_suffix}"
    if output_dir:
        args.meta_output_path = os.path.join(output_dir, "meta.pkl")
        args.train_output = os.path.join(output_dir, os.path.basename(args.train_output))
        if args.val_output:
            args.val_output = os.path.join(output_dir, os.path.basename(args.val_output))
    else:
        args.meta_output_path = "meta.pkl"
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load training/validation data depending on tokenizer method
    if args.method in {"sinewave", "whisper_mel_csv"}:
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
    elif args.method == "char_bpe":
        tokenizer = CharBPETokenizerWithByteFallback(args, train_data, val_data)
    elif args.method == "custom_char_byte_fallback":
        tokenizer = CustomCharTokenizerWithByteFallback(args)
    elif args.method == "json_byte_fallback":
        tokenizer = JsonByteTokenizerWithByteFallback(args)
    elif args.method == "python_programming":
        tokenizer = PythonProgrammingTokenizer(args)
    elif args.method == "sinewave":
        tokenizer = SineWaveTokenizer(args)
    elif args.method == "whisper_mel_csv":
        tokenizer = WhisperMelCsvTokenizer(args)
    else:
        raise ValueError(f"Unknown tokenization method: {args.method}")

    # Tokenize data
    if args.method == "whisper_mel_csv":
        train_ids = tokenizer.tokenize(args.train_input)
    else:
        _reset_byte_token_report(tokenizer)
        train_ids = tokenizer.tokenize(train_data)
        _print_byte_token_report(tokenizer, "train")
        if args.output_tokenization_subdir:
            _write_byte_token_report(tokenizer, "train", output_dir)
    if args.method == "tiktoken":
        print(f"[tiktoken] Total train tokens: {tokenizer.last_token_count:,}")
    if args.method == "whisper_mel_csv" and args.val_input is None:
        split_point = int(len(train_ids) * args.percentage_train)
        val_ids = train_ids[split_point:]
        train_ids = train_ids[:split_point]
    elif args.method == "sinewave" and args.val_input is None:
        split_point = int(len(train_ids) * args.percentage_train)
        val_ids = train_ids[split_point:]
        train_ids = train_ids[:split_point]
    elif val_data is not None:
        if args.method == "whisper_mel_csv":
            val_ids = tokenizer.tokenize(args.val_input)
        else:
            _reset_byte_token_report(tokenizer)
            val_ids = tokenizer.tokenize(val_data)
            _print_byte_token_report(tokenizer, "val")
            if args.output_tokenization_subdir:
                _write_byte_token_report(tokenizer, "val", output_dir)
        if args.method == "tiktoken":
            print(f"[tiktoken] Total val tokens: {tokenizer.last_token_count:,}")
    else:
        val_ids = None

    # Determine dtype based on vocabulary size from meta.pkl
    if args.method == "whisper_mel_csv":
        dtype = None
    elif args.method == "sinewave":
        dtype = np.uint16
    else:
        with open(args.meta_output_path, "rb") as f:
            meta = pickle.load(f)
        vocab_size = meta["vocab_size"]
        dtype = np.uint32 if vocab_size > 65535 else np.uint16

    # Ensure output directories exist if paths include folders
    for output_path in [args.train_output, args.val_output, args.meta_output_path]:
        if output_path:
            out_dir = os.path.dirname(output_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

    # Save tokenized data
    if not args.skip_tokenization:
        if args.method == "whisper_mel_csv":
            save_mel_csv(train_ids, args.train_output, args.mel_csv_float_format)
            if val_ids is not None:
                save_mel_csv(val_ids, args.val_output, args.mel_csv_float_format)
        else:
            save_tokens(train_ids, args.train_output, dtype)
            if val_ids is not None:
                save_tokens(val_ids, args.val_output, dtype)

    if args.method == "sinewave":
        meta = {
            "tokenizer": "sinewave",
            "vocab_size": 256,
            "sine_period": args.sine_period,
            "sine_points_per_period": args.sine_points_per_period,
            "sine_num_periods": args.sine_num_periods,
            "sine_amplitude": args.sine_amplitude,
        }
        with open(args.meta_output_path, "wb") as f:
            pickle.dump(meta, f)
    elif args.method == "whisper_mel_csv":
        meta = {
            "tokenizer": "whisper_mel_csv",
            "sample_rate": args.mel_sample_rate,
            "n_fft": args.mel_n_fft,
            "hop_length": args.mel_hop_length,
            "win_length": args.mel_win_length,
            "n_mels": args.mel_n_mels,
            "f_min": args.mel_f_min,
            "f_max": args.mel_f_max,
            "center": args.mel_center,
            "power": args.mel_power,
            "normalize": args.mel_normalize,
        }
        with open("meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    # Save additional metadata for tiktoken if needed
    if args.method == "tiktoken" and args.additional_tokens_file:
        with open(args.additional_tokens_file, 'r') as f:
            additional_tokens = json.load(f)
        with open(args.meta_output_path, "rb") as f:
            meta = pickle.load(f)
        meta.update({
            "has_additional_tokens": True,
            "special_tokens": additional_tokens,
            "tokenizer": "tiktoken",
            "tiktoken_encoding": args.tiktoken_encoding
        })
        with open(args.meta_output_path, "wb") as f:
            pickle.dump(meta, f)

if __name__ == "__main__":
    main()
