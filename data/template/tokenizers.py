# tokenizers.py
import os
import pickle
import tempfile
import sentencepiece as spm
import tiktoken
from tqdm import tqdm
from collections import defaultdict, Counter
import json
import math
import numpy as np
import importlib.util
from pathlib import Path
try:
    import torch
    import torchaudio
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    torchaudio = None


class Tokenizer:
    def __init__(self, args):
        self.args = args
        self.token_counts = defaultdict(int) if getattr(args, "track_token_counts", False) else None

    def tokenize(self, data):
        raise NotImplementedError("Tokenize method must be implemented by subclasses.")

    def detokenize(self, ids):
        raise NotImplementedError("Detokenize method must be implemented by subclasses.")

    def save_meta(self, meta):
        meta_path = getattr(self.args, "meta_output_path", "meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

    def record_token(self, token_id):
        if self.token_counts is not None:
            self.token_counts[token_id] += 1

    def finalize_meta(self, meta):
        if self.token_counts is not None:
            meta["token_counts"] = dict(self.token_counts)
        self.save_meta(meta)

    @staticmethod
    def get_key_from_meta(keyname, meta_path="meta.pkl"):
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                return meta.get(keyname)
        return None

class SentencePieceTokenizer(Tokenizer):
    def __init__(self, args, input_files=None):
        super().__init__(args)
        self.vocab_size = args.vocab_size
        self.spm_model_file = args.spm_model_file
        self.spm_vocab_file = args.spm_vocab_file
        self.skip_tokenization = args.skip_tokenization
        self.input_files = input_files
        self.output_dir = os.path.dirname(getattr(args, "meta_output_path", ""))
        self.sp = None

        if self.spm_model_file:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(self.spm_model_file)
        elif input_files:
            self.sp = self.train_sentencepiece_model()

    def train_sentencepiece_model(self):
        spm_model_prefix = "trained_spm_model"
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            spm_model_prefix = os.path.join(self.output_dir, spm_model_prefix)
        num_threads = os.cpu_count()
        input_arg = ""
        if isinstance(self.input_files, list):
            with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmpfile:
                for input_file in self.input_files:
                    with open(input_file, "r") as infile:
                        tmpfile.write(infile.read())
                input_arg = tmpfile.name
        else:
            input_arg = self.input_files

        spm.SentencePieceTrainer.train(
            num_threads=num_threads,
            user_defined_symbols="\n, ",
            input=input_arg,
            model_prefix=spm_model_prefix,
            split_digits=True,
            vocab_size=self.vocab_size,
            model_type="bpe",
        )
        print("SentencePiece model training complete.")

        if isinstance(self.input_files, list):
            os.remove(input_arg)

        sp = spm.SentencePieceProcessor()
        sp.load(f"{spm_model_prefix}.model")
        return sp

    def tokenize(self, data):
        if not self.sp:
            raise ValueError("SentencePiece model is not loaded.")
        ids = self.sp.encode_as_ids(data)

        # Record token counts
        for token_id in ids:
            self.record_token(token_id)

        stoi = {self.sp.id_to_piece(i): i for i in range(self.sp.GetPieceSize())}
        itos = {i: self.sp.id_to_piece(i) for i in range(self.sp.GetPieceSize())}

        meta = {
            "vocab_size": self.sp.GetPieceSize(),
            "tokenizer": "sentencepiece",
            "stoi": stoi,
            "itos": itos,
        }
        self.finalize_meta(meta)
        return ids

    def detokenize(self, ids):
        if not self.sp:
            raise ValueError("SentencePiece model is not loaded.")
        return self.sp.decode_ids(ids)

class TiktokenTokenizer(Tokenizer):
    def __init__(self, args):
        super().__init__(args)
        self.tiktoken_encoding = args.tiktoken_encoding
        self.last_token_count = 0

        # Load additional tokens if provided
        self.additional_tokens = {}
        if hasattr(args, 'additional_tokens_file') and args.additional_tokens_file:
            with open(args.additional_tokens_file, 'r') as f:
                self.additional_tokens = json.load(f)

        # Get base encoding
        base_enc = tiktoken.get_encoding(self.tiktoken_encoding)

        if self.additional_tokens:
            # Create custom encoding with additional tokens
            self.enc = tiktoken.Encoding(
                name=f"{self.tiktoken_encoding}_custom",
                pat_str=base_enc._pat_str,
                mergeable_ranks=base_enc._mergeable_ranks,
                special_tokens={**base_enc._special_tokens,
                                **self.additional_tokens},
                disallowed_special=(),
            )
            self.special_tokens = self.additional_tokens
        else:
            self.enc = base_enc
            self.special_tokens = {}

    def tokenize(self, data):
        """Tokenize the input data using tiktoken with support for special tokens."""
        token_ids = []
        current_pos = 0
        data_len = len(data)

        while current_pos < data_len:
            # Try to match special tokens first
            matched_special = False
            for token, token_id in self.special_tokens.items():
                if data.startswith(token, current_pos):
                    token_ids.append(token_id)
                    self.record_token(token_id)
                    current_pos += len(token)
                    matched_special = True
                    break

            if not matched_special:
                # Find the next special token or end of text
                next_special = data_len
                for token in self.special_tokens:
                    pos = data.find(token, current_pos)
                    if pos != -1 and pos < next_special:
                        next_special = pos

                # Take the chunk up to the next special token and let tiktoken handle it
                chunk = data[current_pos:next_special]
                if chunk:
                    # Use encode() for proper subword tokenization
                    chunk_ids = self.enc.encode(
                            chunk,
                            allowed_special=set(),
                            disallowed_special=(),
                            )
                    token_ids.extend(chunk_ids)
                    for token_id in chunk_ids:
                        self.record_token(token_id)
                current_pos = next_special

        # Save metadata
        meta = {
            "vocab_size": self.enc.n_vocab,
            "tokenizer": "tiktoken",
            "tiktoken_encoding": self.tiktoken_encoding,
            "has_additional_tokens": bool(self.additional_tokens),
            "special_tokens": self.special_tokens,
            "itos": {i: self.enc.decode([i]) for i in set(token_ids)}
        }
        self.finalize_meta(meta)

        self.last_token_count = len(token_ids)
        return token_ids

    def detokenize(self, token_ids):
        """Detokenize the token IDs back to text."""
        result = []
        for token_id in token_ids:
            # Check if it's a special token
            found = False
            for token, special_id in self.special_tokens.items():
                if token_id == special_id:
                    result.append(token)
                    found = True
                    break

            if not found:
                # Regular token
                result.append(self.enc.decode([token_id]))

        return ''.join(result)


class CustomTokenizer(Tokenizer):
    def __init__(self, args):
        super().__init__(args)
        if args.tokens_file is None:
            raise ValueError("Tokens file must be provided for custom tokenization method.")
        with open(args.tokens_file, "r") as f:
            self.tokens = [line.strip() for line in f.readlines() if line.strip()]
            self.tokens = [token.replace("\\n", "\n").replace("\\t", "\t") for token in self.tokens]
        self.stoi = {token: i for i, token in enumerate(self.tokens)}
        self.itos = {i: token for i, token in enumerate(self.tokens)}

    def tokenize(self, data):
        encoded_data = []
        i = 0
        covered_chars = 0
        data_len = len(data)
        pbar = tqdm(total=data_len, desc="Tokenizing Custom Tokens")
        while i < data_len:
            matched = False
            for token in self.tokens:
                token_len = len(token)
                if data.startswith(token, i):
                    encoded_data.append(self.stoi[token])
                    self.record_token(self.stoi[token])
                    i += token_len
                    covered_chars += token_len
                    pbar.update(token_len)
                    matched = True
                    break
            if not matched:
                i += 1  # Skip character if no token matches
                pbar.update(1)
        pbar.close()
        coverage = covered_chars / data_len
        print(f"Data coverage by tokens: {coverage*100:.2f}%")
        meta = {"vocab_size": len(self.tokens), "stoi": self.stoi, "itos": self.itos}
        self.finalize_meta(meta)
        return encoded_data

    def detokenize(self, ids):
        return ''.join([self.itos[id] for id in ids])

class ByteTokenizer(Tokenizer):
    def __init__(self, args):
        super().__init__(args)

    def tokenize(self, data):
        data_bytes = data.encode('utf-8')
        ids = list(data_bytes)
        for token_id in ids:
            self.record_token(token_id)
        meta = {
            "vocab_size": 256,
            "tokenizer": "byte",
            "itos": {i: bytes([i]) for i in range(256)},
        }
        self.finalize_meta(meta)
        return ids

    def detokenize(self, ids):
        return bytes(ids).decode('utf-8', errors='replace')


class CharTokenizer(Tokenizer):
    def __init__(self, args, train_data, val_data):
        super().__init__(args)
        self.reuse_chars = args.reuse_chars
        if self.reuse_chars:
            self.chars = self.get_key_from_meta('chars', getattr(args, "meta_output_path", "meta.pkl"))
            if self.chars is None:
                raise ValueError("No chars found in meta.pkl. Cannot reuse chars.")
        else:
            self.chars = sorted(list(set(train_data + (val_data if val_data else ""))))
            print(f"All unique characters: {''.join(self.chars)}")
            print(f"Vocab size: {len(self.chars)}")
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def tokenize(self, data):
        data_len = len(data)
        ids = []
        pbar = tqdm(total=data_len, desc="Tokenizing Characters")
        for ch in data:
            token_id = self.stoi[ch]
            self.record_token(token_id)
            ids.append(token_id)
            pbar.update(1)

        pbar.close()
        meta = {"vocab_size": len(self.chars), "itos": self.itos, "stoi": self.stoi, "chars": self.chars}
        self.finalize_meta(meta)
        return ids

    def detokenize(self, ids):
        return ''.join([self.itos[id] for id in ids])


class CharBPETokenizerWithByteFallback(Tokenizer):
    def __init__(self, args, train_data, val_data=None):
        super().__init__(args)
        self.reuse_meta_path = getattr(args, "char_bpe_vocab_path", None)
        if self.reuse_meta_path:
            meta = self._load_char_bpe_meta(self.reuse_meta_path)
            self.desired_vocab_size = meta["vocab_size"]
            self.char_tokens = meta["char_tokens"]
            self.sorted_char_tokens = meta.get(
                "char_tokens_sorted",
                sorted(self.char_tokens, key=lambda t: len(t), reverse=True),
            )
            self._build_vocab()
            return

        if getattr(args, "vocab_size", None) is None:
            raise ValueError("vocab_size must be provided for char_bpe method.")
        if args.vocab_size <= 256:
            raise ValueError("vocab_size must be greater than 256 to allow space for byte fallback tokens.")

        self.desired_vocab_size = args.vocab_size
        corpus_text = train_data or ""
        if val_data:
            corpus_text += val_data

        self.unique_chars = sorted(set(corpus_text))
        if not self.unique_chars:
            raise ValueError("Training data must contain at least one character for char_bpe tokenization.")

        self.char_tokens = list(self.unique_chars)
        self._train_merges(corpus_text)
        self._build_vocab()

    @staticmethod
    def _load_char_bpe_meta(meta_path):
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Char-BPE meta file not found: {meta_path}")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if meta.get("tokenizer") != "char_bpe":
            raise ValueError("Provided meta file is not from a char_bpe tokenizer.")
        if "char_tokens" not in meta or "vocab_size" not in meta:
            raise ValueError("Meta file missing required char_bpe vocabulary fields.")
        return meta

    def _train_merges(self, text):
        tokens = list(text)
        # Nothing to merge if text empty or target vocab already satisfied
        if len(tokens) < 2:
            return

        current_vocab_size = 256 + len(self.char_tokens)
        merges_needed = self.desired_vocab_size - current_vocab_size

        while merges_needed > 0:
            pair_counts = Counter()
            prev = None
            for token in tokens:
                if prev is not None:
                    pair_counts[(prev, token)] += 1
                prev = token

            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break

            new_token = ''.join(best_pair)
            if new_token in self.char_tokens:
                # Already present, skip to avoid duplicates
                tokens = self._apply_merge(tokens, best_pair, new_token)
            else:
                self.char_tokens.append(new_token)
                tokens = self._apply_merge(tokens, best_pair, new_token)
                merges_needed -= 1

            current_vocab_size = 256 + len(self.char_tokens)
            merges_needed = self.desired_vocab_size - current_vocab_size
            if merges_needed <= 0:
                break

        self.sorted_char_tokens = sorted(self.char_tokens, key=lambda t: len(t), reverse=True)

    @staticmethod
    def _apply_merge(tokens, pair, new_token):
        merged = []
        i = 0
        max_index = len(tokens) - 1
        while i <= max_index:
            if i < max_index and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(new_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def _build_vocab(self):
        self.stoi = {}
        self.itos = {}

        for b in range(256):
            key = bytes([b])
            self.stoi[key] = b
            self.itos[b] = key

        offset = 256
        for idx, token in enumerate(self.char_tokens):
            token_id = offset + idx
            self.stoi[token] = token_id
            self.itos[token_id] = token

        self.vocab_size = len(self.itos)
        self.sorted_char_tokens = sorted(self.char_tokens, key=lambda t: len(t), reverse=True)

    def tokenize(self, data):
        if not data:
            return []

        ids = []
        i = 0
        data_len = len(data)
        pbar = tqdm(total=data_len, desc="Tokenizing Char BPE")

        while i < data_len:
            matched = False
            for token in self.sorted_char_tokens:
                if data.startswith(token, i):
                    token_id = self.stoi[token]
                    ids.append(token_id)
                    self.record_token(token_id)
                    i += len(token)
                    pbar.update(len(token))
                    matched = True
                    break

            if matched:
                continue

            ch = data[i]
            if ch in self.stoi:
                token_id = self.stoi[ch]
                ids.append(token_id)
                self.record_token(token_id)
                i += len(ch)
                pbar.update(len(ch))
            else:
                ch_bytes = ch.encode('utf-8')
                for b in ch_bytes:
                    token_id = self.stoi[bytes([b])]
                    ids.append(token_id)
                    self.record_token(token_id)
                    pbar.update(1)
                i += 1

        pbar.close()

        meta = {
            "vocab_size": self.vocab_size,
            "tokenizer": "char_bpe",
            "stoi": self.stoi,
            "itos": self.itos,
            "char_tokens": self.char_tokens,
            "char_tokens_sorted": self.sorted_char_tokens,
            "byte_fallback": True,
        }
        self.finalize_meta(meta)
        return ids

    def detokenize(self, ids):
        out_pieces = []
        byte_buffer = []

        for token_id in ids:
            token = self.itos.get(token_id)
            if token is None:
                continue

            if isinstance(token, bytes):
                byte_buffer.append(token)
            else:
                if byte_buffer:
                    combined = b''.join(byte_buffer)
                    out_pieces.append(combined.decode('utf-8', errors='replace'))
                    byte_buffer = []
                out_pieces.append(token)

        if byte_buffer:
            combined = b''.join(byte_buffer)
            out_pieces.append(combined.decode('utf-8', errors='replace'))

        return ''.join(out_pieces)

    def finalize_meta(self, meta):
        super().finalize_meta(meta)
        self._write_vocab_jsons(meta)

    def _write_vocab_jsons(self, meta):
        vocab_json = []
        for idx in range(self.vocab_size):
            token = self.itos[idx]
            vocab_json.append(self._format_token_for_json(token))

        with open("char_bpe_vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab_json, f, ensure_ascii=False, indent=2)

        if self.token_counts is not None:
            counts_json = []
            counts = meta.get("token_counts", {})
            for idx in range(self.vocab_size):
                token = self.itos[idx]
                counts_json.append({
                    "id": idx,
                    "token": self._format_token_for_json(token),
                    "count": counts.get(idx, 0)
                })
            with open("char_bpe_token_counts.json", "w", encoding="utf-8") as f:
                json.dump(counts_json, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _format_token_for_json(token):
        if isinstance(token, bytes):
            return f"<byte:{token[0]}>"
        return token


class CustomCharTokenizerWithByteFallback(Tokenizer):
    """
    In this version, we assign IDs 0..255 to raw bytes,
    then custom tokens get IDs from 256 upwards.

    During tokenization:
      1) Convert text to UTF-8 bytes.
      2) For each position in the byte sequence, attempt to match
         a custom token's UTF-8 pattern. If we match, produce that token ID.
         Otherwise, produce the ID for the single byte.

    Detokenization:
      - If ID < 256, it's a single raw byte.
      - If ID >= 256, it's the custom token string.
    """

    def __init__(self, args):
        super().__init__(args)
        if args.custom_chars_file is None:
            raise ValueError("Custom characters file must be provided for this tokenizer.")

        # Load custom tokens from file
        with open(args.custom_chars_file, "r", encoding="utf-8") as f:
            self.custom_tokens = [line.strip() for line in f if line.strip()]

        # Build vocab dictionaries (bytes first, then custom tokens)
        self.build_vocab()

    def build_vocab(self):
        # Assign IDs 0..255 to individual bytes
        self.stoi = {}
        self.itos = {}

        for b in range(256):
            # Store key as the actual single byte
            key = bytes([b])
            self.stoi[key] = b  # ID = b
            self.itos[b] = key

        # Now assign IDs to the custom tokens from 256 onwards
        offset = 256
        self.custom_token_bytes = {}
        for i, token_str in enumerate(self.custom_tokens):
            token_id = offset + i
            self.stoi[token_str] = token_id
            self.itos[token_id] = token_str
            self.custom_token_bytes[token_str] = token_str.encode('utf-8')

        self.custom_char_count = len(self.custom_tokens)
        self.vocab_size = 256 + self.custom_char_count

    def tokenize(self, data):
        # Convert entire string to UTF-8 bytes
        data_bytes = data.encode('utf-8')
        i = 0
        n = len(data_bytes)
        ids = []

        # We'll try to match any custom token at the current position; otherwise single byte
        pbar = tqdm(total=n, desc="Tokenizing Bytes First + Custom")
        while i < n:
            matched = False
            # Check each custom token
            for token_str, token_bytes in self.custom_token_bytes.items():
                length = len(token_bytes)
                # If next 'length' bytes match this custom token
                if data_bytes[i:i+length] == token_bytes:
                    token_id = self.stoi[token_str]  # e.g., 256+
                    self.record_token(token_id)
                    ids.append(token_id)
                    i += length
                    pbar.update(length)
                    matched = True
                    break

            if not matched:
                # No custom token matched, so we treat this as a single byte
                single_byte = data_bytes[i:i+1]
                token_id = self.stoi[single_byte]  # 0..255
                self.record_token(token_id)
                ids.append(token_id)
                i += 1
                pbar.update(1)

        pbar.close()

        # Finalize metadata with token_counts
        meta = {
            "vocab_size": self.vocab_size,
            "tokenizer": "custom_char_with_byte_fallback",
            "custom_chars": self.custom_tokens,  # i.e., custom tokens
            "stoi": self.stoi,
            "itos": self.itos,
            "custom_char_count": self.custom_char_count,
        }
        self.finalize_meta(meta)
        return ids

    def detokenize(self, ids):
        """
        If ID < 256 => single byte
        If ID >= 256 => custom token string
        We'll accumulate bytes in a buffer, and whenever we see a custom token,
        we flush the buffer as text, then append the custom token as is.
        """
        out_pieces = []
        byte_buffer = []

        for idx, token_id in enumerate(ids):
            if token_id < 256:
                # Single raw byte
                byte_buffer.append(self.itos[token_id])  # e.g. b'\x61'
            else:
                # It's a custom token
                # First flush any accumulated bytes
                if byte_buffer:
                    all_bytes = b''.join(byte_buffer)
                    out_pieces.append(all_bytes.decode('utf-8', errors='replace'))
                    byte_buffer = []
                # Append the custom token string
                custom_str = self.itos[token_id]
                out_pieces.append(custom_str)

        # Flush remaining bytes
        if byte_buffer:
            all_bytes = b''.join(byte_buffer)
            out_pieces.append(all_bytes.decode('utf-8', errors='replace'))

        return ''.join(out_pieces)

class JsonByteTokenizerWithByteFallback(Tokenizer):
    """
    Similar to CustomCharTokenizerWithByteFallback, but loads tokens from a JSON array.
    IDs 0..255 are reserved for raw bytes, then custom tokens get IDs from 256 upwards.

    During tokenization:
      1) Convert text to UTF-8 bytes.
      2) For each position in the byte sequence, attempt to match
         a custom token's UTF-8 pattern. If we match, produce that token ID.
         Otherwise, produce the ID for the single byte.

    Detokenization:
      - If ID < 256, it's a single raw byte.
      - If ID >= 256, it's the custom token string.
    """

    def __init__(self, args):
        super().__init__(args)
        if args.json_tokens_file is None:
            raise ValueError("JSON tokens file must be provided for this tokenizer.")

        # Load custom tokens from JSON file
        with open(args.json_tokens_file, "r", encoding="utf-8") as f:
            self.custom_tokens = json.load(f)
            if not isinstance(self.custom_tokens, list):
                raise ValueError("JSON file must contain an array of tokens")

        # Build vocab dictionaries (bytes first, then custom tokens)
        self.build_vocab()

    def build_vocab(self):
        # Assign IDs 0..255 to individual bytes
        self.stoi = {}
        self.itos = {}

        for b in range(256):
            # Store key as the actual single byte
            key = bytes([b])
            self.stoi[key] = b  # ID = b
            self.itos[b] = key

        # Now assign IDs to the custom tokens from 256 onwards
        offset = 256
        self.custom_token_bytes = {}
        for i, token_str in enumerate(self.custom_tokens):
            token_id = offset + i
            self.stoi[token_str] = token_id
            self.itos[token_id] = token_str
            self.custom_token_bytes[token_str] = token_str.encode('utf-8')

        self.custom_token_count = len(self.custom_tokens)
        self.vocab_size = 256 + self.custom_token_count

    def tokenize(self, data):
        # Convert entire string to UTF-8 bytes
        data_bytes = data.encode('utf-8')
        i = 0
        n = len(data_bytes)
        ids = []

        # We'll try to match any custom token at the current position; otherwise single byte
        pbar = tqdm(total=n, desc="Tokenizing Bytes First + JSON Custom")
        while i < n:
            matched = False
            # Check each custom token
            for token_str, token_bytes in self.custom_token_bytes.items():
                length = len(token_bytes)
                # If next 'length' bytes match this custom token
                if data_bytes[i:i+length] == token_bytes:
                    token_id = self.stoi[token_str]  # e.g., 256+
                    self.record_token(token_id)
                    ids.append(token_id)
                    i += length
                    pbar.update(length)
                    matched = True
                    break

            if not matched:
                # No custom token matched, so we treat this as a single byte
                single_byte = data_bytes[i:i+1]
                token_id = self.stoi[single_byte]  # 0..255
                self.record_token(token_id)
                ids.append(token_id)
                i += 1
                pbar.update(1)

        pbar.close()

        # Finalize metadata with token_counts
        meta = {
            "vocab_size": self.vocab_size,
            "tokenizer": "json_byte_fallback",
            "custom_tokens": self.custom_tokens,  # i.e., custom tokens from JSON
            "stoi": self.stoi,
            "itos": self.itos,
            "custom_token_count": self.custom_token_count,
        }
        self.finalize_meta(meta)
        return ids

    def detokenize(self, ids):
        """
        If ID < 256 => single byte
        If ID >= 256 => custom token string
        We'll accumulate bytes in a buffer, and whenever we see a custom token,
        we flush the buffer as text, then append the custom token as is.
        """
        out_pieces = []
        byte_buffer = []

        for idx, token_id in enumerate(ids):
            if token_id < 256:
                # Single raw byte
                byte_buffer.append(self.itos[token_id])  # e.g. b'\x61'
            else:
                # It's a custom token
                # First flush any accumulated bytes
                if byte_buffer:
                    all_bytes = b''.join(byte_buffer)
                    out_pieces.append(all_bytes.decode('utf-8', errors='replace'))
                    byte_buffer = []
                # Append the custom token string
                custom_str = self.itos[token_id]
                out_pieces.append(custom_str)

        # Flush remaining bytes
        if byte_buffer:
            all_bytes = b''.join(byte_buffer)
            out_pieces.append(all_bytes.decode('utf-8', errors='replace'))

        return ''.join(out_pieces)


class JsonBPETokenizerWithByteFallback(Tokenizer):
    """
    JSON-constrained BPE tokenizer with byte fallback.

    - IDs 0..255 are raw byte fallback tokens.
    - Initial text tokens are single-character entries from the JSON array.
    - Additional BPE merge tokens are learned from corpus frequency by combining
      tokens from the evolving merge vocabulary seeded by the JSON array.

    This avoids external byte-combination merges while still permitting new composed
    merge tokens beyond the original JSON entries.
    """

    def __init__(self, args, train_data, val_data=None):
        super().__init__(args)
        if args.json_tokens_file is None:
            raise ValueError("JSON tokens file must be provided for this tokenizer.")

        with open(args.json_tokens_file, "r", encoding="utf-8") as f:
            json_tokens = json.load(f)
            if not isinstance(json_tokens, list):
                raise ValueError("JSON file must contain an array of tokens")

        # Preserve order while deduplicating.
        self.allowed_tokens = list(dict.fromkeys(json_tokens))
        self.allowed_token_set = set(self.allowed_tokens)

        if not self.allowed_tokens:
            raise ValueError("JSON constrained BPE requires at least one token in the JSON array.")

        self.base_char_tokens = [tok for tok in self.allowed_tokens if len(tok) == 1]
        self.base_corpus_tokens = list(self.allowed_tokens)
        self.active_tokens = list(self.base_corpus_tokens)

        requested_vocab_size = getattr(args, "vocab_size", None)
        if requested_vocab_size is not None and requested_vocab_size <= 256:
            raise ValueError("vocab_size must be greater than 256 to allow JSON-BPE tokens.")
        if requested_vocab_size is None:
            self.desired_vocab_size = None
        else:
            self.desired_vocab_size = requested_vocab_size

        corpus_text = (train_data or "") + (val_data or "")
        if corpus_text:
            self._train_merges(corpus_text)

        self._build_vocab()

    @staticmethod
    def _apply_merge(tokens, pair, new_token):
        merged = []
        i = 0
        max_index = len(tokens) - 1
        while i <= max_index:
            if i < max_index and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(new_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def _corpus_to_symbols(self, text):
        symbols = []
        i = 0
        sorted_base = sorted(self.base_corpus_tokens, key=lambda t: len(t), reverse=True)
        while i < len(text):
            matched = False
            for tok in sorted_base:
                if text.startswith(tok, i):
                    symbols.append(tok)
                    i += len(tok)
                    matched = True
                    break
            if not matched:
                # Keep a placeholder that can never merge into JSON-token-only candidates.
                symbols.append(None)
                i += 1
        return symbols

    def _train_merges(self, text):
        symbols = self._corpus_to_symbols(text)
        if len(symbols) < 2:
            return

        while self.desired_vocab_size is None or 256 + len(self.active_tokens) < self.desired_vocab_size:
            pair_counts = Counter()
            prev = None
            for token in symbols:
                if token is None:
                    prev = None
                    continue
                if prev is not None:
                    pair_counts[(prev, token)] += 1
                prev = token

            if not pair_counts:
                break

            best_new_token = None
            best_pair = None
            for pair, _count in pair_counts.most_common():
                candidate = ''.join(pair)
                if (
                    pair[0] in self.active_tokens
                    and pair[1] in self.active_tokens
                    and candidate not in self.active_tokens
                ):
                    best_new_token = candidate
                    best_pair = pair
                    break

            if best_new_token is None:
                break

            self.active_tokens.append(best_new_token)
            symbols = self._apply_merge(symbols, best_pair, best_new_token)

    def _build_vocab(self):
        self.stoi = {}
        self.itos = {}

        for b in range(256):
            key = bytes([b])
            self.stoi[key] = b
            self.itos[b] = key

        offset = 256
        for idx, token in enumerate(self.active_tokens):
            token_id = offset + idx
            self.stoi[token] = token_id
            self.itos[token_id] = token

        self.sorted_active_tokens = sorted(self.active_tokens, key=lambda t: len(t), reverse=True)
        self.vocab_size = len(self.itos)

    def tokenize(self, data):
        if not data:
            meta = {
                "vocab_size": self.vocab_size,
                "tokenizer": "json_bpe_byte_fallback",
                "stoi": self.stoi,
                "itos": self.itos,
                "allowed_tokens": self.allowed_tokens,
                "active_tokens": self.active_tokens,
                "byte_fallback": True,
            }
            self.finalize_meta(meta)
            return []

        ids = []
        i = 0
        n = len(data)
        pbar = tqdm(total=n, desc="Tokenizing JSON-constrained BPE")

        while i < n:
            matched = False
            for token in self.sorted_active_tokens:
                if data.startswith(token, i):
                    token_id = self.stoi[token]
                    ids.append(token_id)
                    self.record_token(token_id)
                    i += len(token)
                    pbar.update(len(token))
                    matched = True
                    break

            if matched:
                continue

            ch_bytes = data[i].encode("utf-8")
            for b in ch_bytes:
                token_id = self.stoi[bytes([b])]
                ids.append(token_id)
                self.record_token(token_id)
                pbar.update(1)
            i += 1

        pbar.close()

        meta = {
            "vocab_size": self.vocab_size,
            "tokenizer": "json_bpe_byte_fallback",
            "stoi": self.stoi,
            "itos": self.itos,
            "allowed_tokens": self.allowed_tokens,
            "active_tokens": self.active_tokens,
            "byte_fallback": True,
        }
        self.finalize_meta(meta)
        return ids

    def detokenize(self, ids):
        out_pieces = []
        byte_buffer = []

        for token_id in ids:
            token = self.itos.get(token_id)
            if token is None:
                continue

            if isinstance(token, bytes):
                byte_buffer.append(token)
            else:
                if byte_buffer:
                    combined = b''.join(byte_buffer)
                    out_pieces.append(combined.decode('utf-8', errors='replace'))
                    byte_buffer = []
                out_pieces.append(token)

        if byte_buffer:
            combined = b''.join(byte_buffer)
            out_pieces.append(combined.decode('utf-8', errors='replace'))

        return ''.join(out_pieces)

    def finalize_meta(self, meta):
        super().finalize_meta(meta)
        self._write_vocab_jsons(meta)

    def _write_vocab_jsons(self, meta):
        vocab_json = []
        for idx in range(self.vocab_size):
            token = self.itos[idx]
            vocab_json.append(self._format_token_for_json(token))

        with open("json_bpe_vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab_json, f, ensure_ascii=False, indent=2)

        if self.token_counts is not None:
            counts_json = []
            counts = meta.get("token_counts", {})
            for idx in range(self.vocab_size):
                token = self.itos[idx]
                counts_json.append({
                    "id": idx,
                    "token": self._format_token_for_json(token),
                    "count": counts.get(idx, 0)
                })
            with open("json_bpe_token_counts.json", "w", encoding="utf-8") as f:
                json.dump(counts_json, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _format_token_for_json(token):
        if isinstance(token, bytes):
            return f"<byte:{token[0]}>"
        return token


def _load_python_token_processor():
    """Load the PythonTokenProcessor helper without requiring a package install."""
    tokenizer_path = Path(__file__).parent / "programming_tokenizers" / "python_tokenizer.py"
    spec = importlib.util.spec_from_file_location("python_tokenizer", tokenizer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load python_tokenizer from {tokenizer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PythonTokenProcessor


class PythonProgrammingTokenizer(JsonByteTokenizerWithByteFallback):
    """Tokenize Python code with reserved tokens and byte fallback for everything else."""

    def __init__(self, args):
        if args.json_tokens_file is None:
            args.json_tokens_file = str(
                Path(__file__).parent / "premade_vocab_sets" / "python_programming_tokens.json"
            )
        super().__init__(args)
        python_processor_cls = _load_python_token_processor()
        self.python_processor = python_processor_cls(self.custom_tokens)

    def tokenize(self, data):
        ids = []

        def encode_bytes(segment: str) -> None:
            for byte_val in segment.encode("utf-8"):
                token_id = self.stoi[bytes([byte_val])]
                self.record_token(token_id)
                ids.append(token_id)

        def emit_reserved(token_text: str) -> None:
            token_id = self.stoi[token_text]
            self.record_token(token_id)
            ids.append(token_id)

        self.python_processor.encode_with_reserved_tokens(data, encode_bytes, emit_reserved)

        meta = {
            "vocab_size": self.vocab_size,
            "tokenizer": "python_json_byte_fallback",
            "custom_tokens": self.custom_tokens,
            "stoi": self.stoi,
            "itos": self.itos,
            "custom_token_count": self.custom_token_count,
        }
        self.finalize_meta(meta)
        return ids


class SineWaveTokenizer:
    """Generate a deterministic sequence of sine wave samples."""

    def __init__(self, args):
        self.period = args.sine_period
        self.points_per_period = args.sine_points_per_period
        self.num_periods = args.sine_num_periods
        self.amplitude = args.sine_amplitude
        self.max_val = 255

    def generate_wave(self):
        total_points = self.num_periods * self.points_per_period
        values = []
        for i in range(total_points):
            x = (i * 2 * math.pi) / self.points_per_period
            y = 64 + self.amplitude * math.sin(x * self.period)
            y_clamped = int(max(0, min(self.max_val, round(y))))
            values.append(y_clamped)
        return values

    def tokenize(self, data=None):
        # `data` is unused; generation is parameter driven.
        return self.generate_wave()

    def detokenize(self, ids):
        array = np.asarray(ids, dtype=np.int64)
        return ','.join(map(str, array.tolist()))

      
class WhisperMelCsvTokenizer(Tokenizer):
    """Generate Whisper-style log-mel spectrogram frames suitable for CSV export."""

    def __init__(self, args):
        super().__init__(args)
        if torch is None or torchaudio is None:
            raise ImportError("WhisperMelCsvTokenizer requires torch and torchaudio.")
        self.sample_rate = args.mel_sample_rate
        self.n_fft = args.mel_n_fft
        self.hop_length = args.mel_hop_length
        self.win_length = args.mel_win_length
        self.n_mels = args.mel_n_mels
        self.f_min = args.mel_f_min
        self.f_max = args.mel_f_max
        self.center = args.mel_center
        self.power = args.mel_power
        self.normalize = args.mel_normalize
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            center=self.center,
            power=self.power,
            mel_scale="slaney",
            norm="slaney",
        )

    def _load_audio(self, path):
        waveform, original_sample_rate = torchaudio.load(path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if original_sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=original_sample_rate, new_freq=self.sample_rate
            )
        return waveform

    def _whisper_log_mel(self, mel_spec):
        log_mel = torch.clamp(mel_spec, min=1e-10).log10()
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        return (log_mel + 4.0) / 4.0

    def tokenize(self, data):
        waveform = self._load_audio(data)
        mel_spec = self.mel_transform(waveform).squeeze(0)
        if self.normalize:
            mel_spec = self._whisper_log_mel(mel_spec)
        mel_frames = mel_spec.transpose(0, 1).contiguous()
        meta = {
            "tokenizer": "whisper_mel_csv",
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "n_mels": self.n_mels,
            "f_min": self.f_min,
            "f_max": self.f_max,
            "center": self.center,
            "power": self.power,
            "normalize": self.normalize,
        }
        self.finalize_meta(meta)
        return mel_frames.cpu().numpy()

    def detokenize(self, ids):
        array = np.asarray(ids, dtype=np.float32)
        lines = [",".join(map(str, row)) for row in array.tolist()]
        return "\n".join(lines)
      
