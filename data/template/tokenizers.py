# tokenizers.py
import os
import pickle
import tempfile
import sentencepiece as spm
import tiktoken
from tqdm import tqdm
from collections import defaultdict
import json
import math
import numpy as np
import importlib.util
from pathlib import Path


class Tokenizer:
    def __init__(self, args):
        self.args = args
        self.token_counts = defaultdict(int) if getattr(args, "track_token_counts", False) else None

    def tokenize(self, data):
        raise NotImplementedError("Tokenize method must be implemented by subclasses.")

    def detokenize(self, ids):
        raise NotImplementedError("Detokenize method must be implemented by subclasses.")

    def save_meta(self, meta):
        with open("meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    def record_token(self, token_id):
        if self.token_counts is not None:
            self.token_counts[token_id] += 1

    def finalize_meta(self, meta):
        if self.token_counts is not None:
            meta["token_counts"] = dict(self.token_counts)
        self.save_meta(meta)

    def get_vocabulary(self):
        """Return the list of string representations that make up the tokenizer's vocabulary."""
        raise NotImplementedError("Vocabulary extraction is not implemented for this tokenizer.")

    @staticmethod
    def get_key_from_meta(keyname):
        meta_path = 'meta.pkl'
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
        self.sp = None

        if self.spm_model_file:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(self.spm_model_file)
        elif input_files:
            self.sp = self.train_sentencepiece_model()

    def train_sentencepiece_model(self):
        spm_model_prefix = "trained_spm_model"
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

    def get_vocabulary(self):
        if not self.sp:
            raise ValueError("SentencePiece model is not loaded.")
        return [self.sp.id_to_piece(i) for i in range(self.sp.GetPieceSize())]

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

    def get_vocabulary(self):
        vocab = []
        seen = set()
        for token_id in range(self.enc.n_vocab):
            token_bytes = self.enc.decode_single_token_bytes(token_id)
            token_str = token_bytes.decode('utf-8', errors='replace')
            if token_str not in seen:
                seen.add(token_str)
                vocab.append(token_str)

        # Include known special tokens (base and additional)
        special_tokens = {}
        if hasattr(self.enc, "_special_tokens") and isinstance(self.enc._special_tokens, dict):
            special_tokens.update(self.enc._special_tokens)
        special_tokens.update(self.special_tokens)

        for token in special_tokens.keys():
            if token not in seen:
                seen.add(token)
                vocab.append(token)

        return vocab


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

    def get_vocabulary(self):
        return list(self.tokens)

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

    def get_vocabulary(self):
        return [chr(i) for i in range(256)]


class CharTokenizer(Tokenizer):
    def __init__(self, args, train_data, val_data):
        super().__init__(args)
        self.reuse_chars = args.reuse_chars
        if self.reuse_chars:
            self.chars = self.get_key_from_meta('chars')
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

    def get_vocabulary(self):
        return list(self.chars)

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

    def get_vocabulary(self):
        vocab = []
        for token in self.itos.values():
            if isinstance(token, bytes):
                vocab.append(token.decode('utf-8', errors='replace'))
            else:
                vocab.append(token)
        return vocab

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

    def get_vocabulary(self):
        vocab = []
        for token in self.itos.values():
            if isinstance(token, bytes):
                vocab.append(token.decode('utf-8', errors='replace'))
            else:
                vocab.append(token)
        return vocab


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

