import os
import argparse
import pickle
import numpy as np

def infer_dtype_from_meta(meta_path: str) -> np.dtype:
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    # Match prepare.py behavior
    if meta.get("tokenizer") == "sinewave":
        return np.dtype(np.uint16)

    vocab_size = meta.get("vocab_size")
    if vocab_size is None:
        raise ValueError("meta.pkl missing 'vocab_size'. Can't infer dtype safely.")

    return np.dtype(np.uint32) if vocab_size > 65535 else np.dtype(np.uint16)

def count_tokens_in_bin(bin_path: str, dtype: np.dtype) -> int:
    size = os.path.getsize(bin_path)
    itemsize = dtype.itemsize
    if size % itemsize != 0:
        raise ValueError(
            f"{bin_path}: file size {size} not divisible by dtype itemsize {itemsize}. "
            "Wrong dtype or corrupted file?"
        )
    return size // itemsize

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="meta.pkl", help="Path to meta.pkl")
    ap.add_argument("--train", default="train.bin", help="Path to train.bin")
    ap.add_argument("--val", default="val.bin", help="Path to val.bin")
    args = ap.parse_args()

    dtype = infer_dtype_from_meta(args.meta)
    print(f"Inferred dtype: {dtype}")

    if os.path.exists(args.train):
        n_train = count_tokens_in_bin(args.train, dtype)
        print(f"train tokens: {n_train:,}  ({args.train})")
    else:
        print(f"train.bin not found: {args.train}")

    if args.val and os.path.exists(args.val):
        n_val = count_tokens_in_bin(args.val, dtype)
        print(f"val tokens:   {n_val:,}  ({args.val})")
    else:
        print(f"val.bin not found (or not provided): {args.val}")

if __name__ == "__main__":
    main()
