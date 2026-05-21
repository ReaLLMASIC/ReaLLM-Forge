#!/usr/bin/env python3
"""Check val.bin / train.bin for spurious '!' tokens injected during prepare.

Usage:
    python check_tokens.py data/dialogsum/hf_gpt2/val.bin
    python check_tokens.py data/dialogsum/hf_HuggingFaceTB_SmolLM3-3B/val.bin -n 200
    python check_tokens.py data/dialogsum/hf_gpt2/val.bin --compare data/dialogsum/input.txt
"""
import argparse
import os
import pickle
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Check .bin for artifact '!' tokens")
    parser.add_argument("bin_file", help="Path to train.bin or val.bin")
    parser.add_argument("-n", "--num_tokens", type=int, default=100,
                        help="Number of tokens to inspect (default: 100)")
    parser.add_argument("-m", "--meta", type=str, default=None,
                        help="Path to meta.pkl (auto-detected from bin_file dir)")
    parser.add_argument("--compare", type=str, default=None,
                        help="Path to original input.txt to compare '!' counts")
    args = parser.parse_args()

    # Auto-detect meta.pkl
    if args.meta is None:
        args.meta = os.path.join(os.path.dirname(args.bin_file), "meta.pkl")
    if not os.path.exists(args.meta):
        sys.exit(f"meta.pkl not found at {args.meta}")

    with open(args.meta, "rb") as f:
        meta = pickle.load(f)

    vocab_size = meta.get("vocab_size", 0)
    tokenizer_type = meta.get("tokenizer", "unknown")
    dtype = np.uint32 if vocab_size > 65535 else np.uint16

    print(f"Tokenizer : {tokenizer_type}")
    print(f"Vocab size: {vocab_size}")

    # --- Find which token ID(s) decode to '!' ---
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sample import get_tokenizer_functions
    encode, decode = get_tokenizer_functions(meta)

    bang_ids = set()
    stoi = meta.get("stoi", {})
    if stoi:
        for tok_str, tok_id in stoi.items():
            if isinstance(tok_str, str) and tok_str.strip() == "!":
                bang_ids.add(tok_id)
    # Also try encoding '!' directly
    try:
        for tid in encode("!"):
            bang_ids.add(tid)
    except Exception:
        pass
    print(f"Token IDs that encode '!': {sorted(bang_ids)}")

    # --- Load the .bin ---
    data = np.fromfile(args.bin_file, dtype=dtype)
    total = len(data)
    n = min(args.num_tokens, total)
    ids = data[:n]

    # Count '!' token IDs in full file and in first n tokens
    bang_count_full = sum(1 for tid in data if int(tid) in bang_ids)
    bang_count_n = sum(1 for tid in ids if int(tid) in bang_ids)
    print(f"\n'!' tokens in first {n} of {total} tokens: {bang_count_n}")
    print(f"'!' tokens in entire file ({total} tokens): {bang_count_full}")
    print(f"'!' density: {bang_count_full / total * 100:.2f}%")

    # --- Decode first n tokens and show text ---
    decoded = decode(ids.tolist())
    bang_in_text = decoded.count("!")
    print(f"\n--- Decoded first {n} tokens ---")
    print(decoded[:500])
    if len(decoded) > 500:
        print(f"  ... ({len(decoded)} chars total)")
    print(f"\n'!' characters in decoded text: {bang_in_text}")

    # --- Compare with original text ---
    if args.compare and os.path.exists(args.compare):
        with open(args.compare, "r", encoding="utf-8", errors="replace") as f:
            original = f.read()
        # Take a comparable-length slice of the original
        orig_slice = original[:len(decoded)]
        orig_bang = orig_slice.count("!")
        print(f"\n--- Comparison with {args.compare} ---")
        print(f"'!' in original text (first {len(orig_slice)} chars): {orig_bang}")
        print(f"'!' in decoded .bin  (first {len(decoded)} chars): {bang_in_text}")
        if bang_in_text > orig_bang:
            print(f"  >>> {bang_in_text - orig_bang} EXTRA '!' found — likely artifacts from tokenization!")
        else:
            print(f"  Counts match — '!' in .bin came from the original text.")

    # --- Show per-token breakdown around '!' tokens ---
    print(f"\n--- Positions of '!' tokens in first {n} ---")
    found = False
    for i, tid in enumerate(ids.tolist()):
        if int(tid) in bang_ids:
            found = True
            # Show context: 2 tokens before and after
            start = max(0, i - 2)
            end = min(n, i + 3)
            context_ids = ids[start:end].tolist()
            context_decoded = [decode([cid]) for cid in context_ids]
            pointer = i - start
            display = " | ".join(
                f"[{context_ids[j]}]{repr(context_decoded[j])}"
                + (" <<<!" if j == pointer else "")
                for j in range(len(context_ids))
            )
            print(f"  pos {i:5d}: {display}")
    if not found:
        print("  (none found)")


if __name__ == "__main__":
    main()
