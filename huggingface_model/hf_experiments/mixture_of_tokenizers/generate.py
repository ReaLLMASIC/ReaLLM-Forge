"""Generate a continuation with one manually selected tokenizer route."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluate import ROUTES, load_model_and_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizers", help="Defaults to MODEL/tokenizer")
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 selects greedy decoding")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-json", help="Optional continuation, token IDs and exact raw bytes")
    args = parser.parse_args()
    if args.temperature < 0 or args.max_new_tokens <= 0 or not 0 < args.top_p <= 1:
        parser.error("Use temperature >= 0, max-new-tokens > 0 and 0 < top-p <= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    model, tokenizer = load_model_and_tokenizer(args.model, args.tokenizers, args.device)
    route = ROUTES.index(args.route)
    prompt_ids = tokenizer.encode_route(args.prompt, route, add_bos=True)
    available = model.config.max_position_embeddings - len(prompt_ids)
    if args.max_new_tokens > available:
        parser.error(f"Prompt uses {len(prompt_ids)} tokens; at most {available} new tokens fit the configured context")
    torch.manual_seed(args.seed)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
    generation = dict(max_new_tokens=args.max_new_tokens, do_sample=args.temperature > 0)
    if args.temperature > 0:
        generation.update(temperature=args.temperature, top_p=args.top_p)
    with torch.inference_mode():
        result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                tokenizer_ids=route, **generation)
    new_ids = result[0, len(prompt_ids):].tolist()
    raw = tokenizer.decode_bytes(new_ids, skip_special_tokens=True)
    try:
        text = raw.decode("utf-8", errors="strict")
        valid_utf8 = True
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
        valid_utf8 = False
    print(text)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "route": args.route, "text": text, "token_ids": new_ids,
            "raw_bytes_hex": raw.hex(), "valid_utf8": valid_utf8,
            "stopped_on_eos": bool(new_ids and new_ids[-1] == tokenizer.eos_id(route)),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
