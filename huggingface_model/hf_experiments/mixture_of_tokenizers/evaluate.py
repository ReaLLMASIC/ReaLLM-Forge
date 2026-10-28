"""Compare tokenizer routes on identical text with an explicit context policy.

BPB is the coding rate of the tokenizer's canonical token sequence, not text
likelihood marginalized over all byte/subword spellings. No benchmark downloads
or prompt rewriting happen here. The optional MC input is normalized JSONL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import torch

ROUTES = ("byte", "zh", "ko", "ja", "en")
TOTAL_KEYS = (
    "documents", "text_nll", "text_tokens", "source_bytes", "source_characters",
    "fallback_tokens", "fallback_bytes", "eos_nll", "eos_count", "windows",
)


def _route_index(route: str | int) -> int:
    if isinstance(route, str):
        return ROUTES.index(route)
    if not 0 <= int(route) < len(ROUTES):
        raise ValueError(f"Unknown route: {route}")
    return int(route)


def _device(model, device=None):
    return torch.device(device) if device is not None else next(model.parameters()).device


def unicode_byte_chunks(text: str, max_bytes: int) -> list[str]:
    """Partition text at code-point boundaries, without normalization or loss."""
    if max_bytes < 4:
        raise ValueError("Byte-context evaluation needs at least four content bytes.")
    chunks, current = [], []
    used = 0
    for char in text:
        size = len(char.encode("utf-8"))
        if used + size > max_bytes:
            chunks.append("".join(current))
            current, used = [], 0
        current.append(char)
        used += size
    if current or not chunks:
        chunks.append("".join(current))
    return chunks


@torch.inference_mode()
def _score_ids(model, ids: list[int], route: int, context_length: int, device):
    """Yield (absolute target index, target ID, NLL), scoring each target once."""
    if context_length < 2:
        raise ValueError("context_length must be at least two")
    for start in range(0, len(ids) - 1, context_length - 1):
        chunk = ids[start:start + context_length]
        input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
        output = model(
            input_ids=input_ids,
            tokenizer_ids=route,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            return_logits=True,
        )
        log_probs = output.logits[0, :-1].float().log_softmax(-1)
        targets = input_ids[0, 1:]
        nll = -log_probs.gather(-1, targets[:, None]).squeeze(-1)
        for offset, (target, value) in enumerate(zip(chunk[1:], nll.cpu().tolist()), 1):
            if not math.isfinite(value):
                raise ValueError(f"Non-finite NLL at target {start + offset}, route {route}")
            yield start + offset, target, value


def summarize(totals: dict) -> dict:
    result = dict(totals)
    nbytes, ntokens = totals["source_bytes"], totals["text_tokens"]
    result.update(
        bpb=totals["text_nll"] / (nbytes * math.log(2)) if nbytes else None,
        mean_token_nll=totals["text_nll"] / ntokens if ntokens else None,
        bytes_per_token=nbytes / ntokens if ntokens else None,
        characters_per_token=totals["source_characters"] / ntokens if ntokens else None,
        fallback_token_fraction=totals["fallback_tokens"] / ntokens if ntokens else None,
        fallback_source_byte_fraction=totals["fallback_bytes"] / nbytes if nbytes else None,
        mean_eos_nll=totals["eos_nll"] / totals["eos_count"] if totals["eos_count"] else None,
    )
    return result


def score_text(model, tokenizer, text: str, route: str | int,
               context_mode: str = "token", device=None) -> dict:
    """Score all original text bytes exactly once; score genuine document EOS separately.

    token: fixed token windows with a one-token overlap. This is a reset-window
    policy, not a rolling full-context perplexity implementation.
    bytes: fixed raw-byte chunks shared by every route, split on Unicode code
    points. Each chunk starts with BOS; only the document's final chunk has EOS.
    """
    if context_mode not in ("token", "bytes"):
        raise ValueError("context_mode must be 'token' or 'bytes'")
    route = _route_index(route)
    context = int(model.config.max_position_embeddings)
    device = _device(model, device)
    chunks = [text] if context_mode == "token" else unicode_byte_chunks(text, context - 2)
    totals = {key: 0 for key in TOTAL_KEYS}
    totals.update(documents=1, source_bytes=len(text.encode("utf-8")), source_characters=len(text))
    for index, chunk in enumerate(chunks):
        tokens = tokenizer.encode_route(chunk, route)
        final = index == len(chunks) - 1
        ids = [tokenizer.bos_id(route)] + tokens
        if final:
            ids.append(tokenizer.eos_id(route))
        if context_mode == "bytes" and len(ids) > context:
            raise ValueError("Tokenizer emitted more tokens than UTF-8 bytes; cannot match byte context.")
        totals["text_tokens"] += len(tokens)
        totals["fallback_tokens"] += sum(token < 256 for token in tokens)
        # A shared fallback token represents exactly one original UTF-8 byte.
        totals["fallback_bytes"] += sum(token < 256 for token in tokens)
        totals["windows"] += math.ceil((len(ids) - 1) / (context - 1))
        for _, token, nll in _score_ids(model, ids, route, context, device):
            if token == tokenizer.eos_id(route):
                totals["eos_nll"] += nll
                totals["eos_count"] += 1
            else:
                totals["text_nll"] += nll
    return summarize(totals)


def iter_documents(path: Path, limit: int = 0) -> Iterable[str]:
    seen = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("text"), str):
                raise ValueError(f"Each record in {path} must have a string 'text'.")
            yield row["text"]
            seen += 1
            if limit and seen >= limit:
                break


def evaluate_corpus(model, tokenizer, data_dir, split="val", max_documents=100,
                    context_mode="token", device=None) -> dict:
    data_dir = Path(data_dir)
    results, hashes = {}, {}
    for language in ROUTES[1:]:
        path = data_dir / language / f"{split}.jsonl"
        # Hash the exact evaluated text sequence, not unused trailing documents.
        digest = hashlib.sha256()
        totals = {route: {key: 0 for key in TOTAL_KEYS} for route in (language, "byte")}
        for text in iter_documents(path, max_documents):
            encoded = text.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
            for route in totals:
                scored = score_text(model, tokenizer, text, route, context_mode, device)
                for key in TOTAL_KEYS:
                    totals[route][key] += scored[key]
        if not totals[language]["documents"]:
            raise ValueError(f"No documents to evaluate in {path}")
        results[language] = {route: summarize(values) for route, values in totals.items()}
        hashes[language] = digest.hexdigest()
    macro = {}
    for route_mode in ("native", "byte"):
        values = [results[lang][lang if route_mode == "native" else "byte"]["bpb"]
                  for lang in ROUTES[1:]]
        macro[route_mode + "_bpb"] = sum(values) / len(values) if all(v is not None for v in values) else None
    return {
        "metric": "canonical_tokenization_bits_per_utf8_byte",
        "context_mode": context_mode,
        "context_tokens": int(model.config.max_position_embeddings),
        "context_policy": ("T-1 stride; one token overlap; reset positions each window"
                           if context_mode == "token" else
                           "same Unicode chunks <= T-2 UTF-8 bytes for every route; BOS resets each chunk"),
        "eos_policy": "one genuine document EOS, reported separately from BPB",
        "split": split,
        "evaluated_text_sha256": hashes,
        "macro_average": macro,
        "languages": results,
    }


def score_continuation(model, tokenizer, prompt: str, continuation: str,
                       route: str | int, device=None) -> float:
    """Sum log probabilities under an explicit prompt/answer token boundary.

    Prompt and continuation are encoded separately. This avoids a BPE merge
    straddling the score boundary, but is a specified evaluation protocol rather
    than an assertion of matching any benchmark's original harness.
    """
    route = _route_index(route)
    prompt_ids = tokenizer.encode_route(prompt, route)
    continuation_ids = tokenizer.encode_route(continuation, route)
    if not continuation_ids:
        raise ValueError("A multiple-choice continuation cannot be empty.")
    ids = [tokenizer.bos_id(route)] + prompt_ids + continuation_ids
    context = int(model.config.max_position_embeddings)
    if len(ids) > context:
        raise ValueError("Prompt and continuation exceed context; no implicit truncation is applied.")
    answer_start = 1 + len(prompt_ids)
    return -sum(nll for position, _, nll in _score_ids(model, ids, route, context, _device(model, device))
                if position >= answer_start)


def evaluate_multiple_choice(model, tokenizer, path, device=None) -> dict:
    """Score native and byte routes on a shared subset of complete MC questions."""
    path = Path(path)
    output, skipped = [], []
    context = int(model.config.max_position_embeddings)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            language, prompt, choices, answer = (
                row["language"], row["prompt"], row["choices"], row["answer_index"])
            if (language not in ROUTES[1:] or not isinstance(prompt, str)
                    or not isinstance(choices, list) or len(choices) < 2
                    or any(not isinstance(c, str) or not c for c in choices)
                    or not isinstance(answer, int) or isinstance(answer, bool)
                    or not 0 <= answer < len(choices)):
                raise ValueError(f"Invalid MC record at line {line_number}")
            routes = (language, "byte")
            fits = all(
                1 + len(tokenizer.encode_route(prompt, route)) + len(tokenizer.encode_route(choice, route)) <= context
                for route in routes for choice in choices
            )
            if not fits:
                skipped.append({"line": line_number, "reason": "context_exceeded_in_at_least_one_route"})
                continue
            record = {"line": line_number, "language": language, "answer_index": answer, "routes": {}}
            for route in routes:
                scores = [score_continuation(model, tokenizer, prompt, choice, route, device) for choice in choices]
                prediction = max(range(len(scores)), key=scores.__getitem__)
                record["routes"][route] = {
                    "sum_log_probs": scores, "prediction": prediction, "correct": prediction == answer,
                }
            output.append(record)
    metrics = {}
    for language in ROUTES[1:]:
        items = [r for r in output if r["language"] == language]
        metrics[language] = {
            "count": len(items),
            "native_accuracy": sum(r["routes"][language]["correct"] for r in items) / len(items) if items else None,
            "byte_accuracy": sum(r["routes"]["byte"]["correct"] for r in items) / len(items) if items else None,
        }
    return {
        "protocol": "separately_encoded_prompt_and_continuation; summed_answer_log_probability; no_EOS",
        "official_benchmark_score": False,
        "context_policy": "complete questions only; drop from both routes if any option exceeds context",
        "source_sha256": digest,
        "scored_count": len(output), "skipped_count": len(skipped),
        "language_metrics": metrics, "records": output, "skipped": skipped,
    }


def load_model_and_tokenizer(model_dir, tokenizer_dir=None, device="cpu"):
    from modeling_mixture import MixtureForCausalLM
    from tokenization_mixture import MixtureTokenizer
    model_dir = Path(model_dir)
    tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else model_dir / "tokenizer"
    tokenizer = MixtureTokenizer.from_pretrained(str(tokenizer_dir))
    model = MixtureForCausalLM.from_pretrained(str(model_dir)).to(device).eval()
    if model.config.lexical_size != tokenizer.lexical_size:
        raise ValueError("Model and tokenizer lexical sizes differ.")
    expected = getattr(model.config, 'tokenizer_fingerprint', None)
    if expected is not None and expected != tokenizer.fingerprint:
        raise ValueError('Tokenizer vocabulary differs from the bundle used to train this model.')
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizers", help="Defaults to MODEL/tokenizer")
    parser.add_argument("--data", default="data/prepared")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--max-documents", type=int, default=100, help="Per language; 0 means all")
    parser.add_argument("--context-mode", choices=("token", "bytes"), default="bytes")
    parser.add_argument("--mc-jsonl", help="Evaluate normalized MC records instead of held-out corpus")
    parser.add_argument("--output", default="runs/eval.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.max_documents < 0:
        parser.error("--max-documents must be nonnegative")
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    model, tokenizer = load_model_and_tokenizer(args.model, args.tokenizers, args.device)
    if args.mc_jsonl:
        result = evaluate_multiple_choice(model, tokenizer, args.mc_jsonl, args.device)
    else:
        result = evaluate_corpus(model, tokenizer, args.data, args.split, args.max_documents,
                                 args.context_mode, args.device)
    result.update(model=str(Path(args.model)), lexical_size=tokenizer.lexical_size)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result.get("macro_average", result.get("language_metrics")), indent=2))
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
