"""Streaming tiktoken tokenizer for large files.
Reads input line-by-line, tokenizes in chunks, writes binary output."""

import tiktoken
import numpy as np
import pickle
import sys
import os

input_file = sys.argv[1] if len(sys.argv) > 1 else "input.txt"
train_output = "train.bin"
val_output = "val.bin"
meta_output = "meta.pkl"
val_fraction = 0.1

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
dtype = np.uint32 if vocab_size > 65535 else np.uint16

# Get total file size for progress and split point
file_size = os.path.getsize(input_file)
split_byte = int(file_size * (1 - val_fraction))

print(f"Input: {input_file} ({file_size / 1e9:.1f} GB)")
print(f"Vocab size: {vocab_size}, dtype: {dtype}")
print(f"Train/val split at byte {split_byte / 1e9:.1f} GB")

train_tokens = 0
val_tokens = 0
bytes_read = 0
in_val = False

train_f = open(train_output, "wb")
val_f = open(val_output, "wb")

with open(input_file, "r", encoding="utf-8", errors="replace") as f:
    chunk_lines = []
    chunk_size = 0
    target_chunk = 50 * 1024 * 1024  # 50MB chunks

    for line in f:
        line_bytes = len(line.encode("utf-8"))
        bytes_read += line_bytes
        chunk_lines.append(line)
        chunk_size += line_bytes

        if chunk_size >= target_chunk:
            text = "".join(chunk_lines)
            tokens = enc.encode_ordinary(text)
            arr = np.array(tokens, dtype=dtype)

            if not in_val and bytes_read >= split_byte:
                in_val = True
                print(f"  Switching to val at {bytes_read / 1e9:.1f} GB")

            if in_val:
                arr.tofile(val_f)
                val_tokens += len(tokens)
            else:
                arr.tofile(train_f)
                train_tokens += len(tokens)

            chunk_lines = []
            chunk_size = 0

            total_tokens = train_tokens + val_tokens
            pct = bytes_read / file_size * 100
            print(f"  {pct:.1f}% | {total_tokens:,} tokens | train: {train_tokens:,} val: {val_tokens:,}")

    # Flush remaining
    if chunk_lines:
        text = "".join(chunk_lines)
        tokens = enc.encode_ordinary(text)
        arr = np.array(tokens, dtype=dtype)
        if in_val:
            arr.tofile(val_f)
            val_tokens += len(tokens)
        else:
            arr.tofile(train_f)
            train_tokens += len(tokens)

train_f.close()
val_f.close()

# Save meta
meta = {
    "vocab_size": vocab_size,
    "tokenizer": "tiktoken",
    "tiktoken_encoding": "gpt2",
}
with open(meta_output, "wb") as f:
    pickle.dump(meta, f)

print(f"\nDone!")
print(f"  Train tokens: {train_tokens:,}")
print(f"  Val tokens:   {val_tokens:,}")
print(f"  Total tokens: {train_tokens + val_tokens:,}")
