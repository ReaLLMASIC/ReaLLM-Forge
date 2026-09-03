#!/usr/bin/env python3
"""
Benchmark EN->ES translation checkpoints with simple generation metrics.
"""

import argparse
from statistics import mean

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def maybe_compute_sacrebleu(predictions, references):
    try:
        import sacrebleu
    except Exception:
        return None

    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    chrf = sacrebleu.corpus_chrf(predictions, [references]).score
    return {"bleu": bleu, "chrf": chrf}


def exact_match_rate(predictions, references):
    matches = [int(p.strip().lower() == r.strip().lower()) for p, r in zip(predictions, references)]
    return mean(matches) if matches else 0.0


def main(args):
    print(f"Loading dataset: {args.dataset_name} ({args.dataset_config}) split={args.dataset_split}")
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    dataset = dataset.select(range(min(len(dataset), args.num_samples)))

    print(f"Loading model/tokenizer from: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, attn_implementation="eager")
    model.eval()
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions, references = [], []
    for i, row in enumerate(dataset):
        src = row["translation"][args.source_lang]
        tgt = row["translation"][args.target_lang]
        prompt = (
            f"Translate {args.source_lang_name} to {args.target_lang_name}:\n"
            f"{args.source_lang_name}: {src}\n"
            f"{args.target_lang_name}: "
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = decoded.replace(prompt, "").strip().split("\n")[0]

        predictions.append(pred)
        references.append(tgt)

        if i < args.print_examples:
            print(f"\n[{i}]")
            print(f"SRC: {src}")
            print(f"REF: {tgt}")
            print(f"PRD: {pred}")

    metrics = {"exact_match": exact_match_rate(predictions, references)}
    sacrebleu_metrics = maybe_compute_sacrebleu(predictions, references)
    if sacrebleu_metrics:
        metrics.update(sacrebleu_metrics)
    else:
        print("sacrebleu not installed; skipping BLEU/chrF.")

    print("\n=== Benchmark results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark EN->ES translation with Gemma checkpoints.")
    parser.add_argument("--model_name", type=str, required=True, help="Checkpoint path or HF model id.")
    parser.add_argument("--dataset_name", type=str, default="Helsinki-NLP/opus-100")
    parser.add_argument("--dataset_config", type=str, default="en-es")
    parser.add_argument("--dataset_split", type=str, default="train[10%:11%]")
    parser.add_argument("--source_lang", type=str, default="en")
    parser.add_argument("--target_lang", type=str, default="es")
    parser.add_argument("--source_lang_name", type=str, default="English")
    parser.add_argument("--target_lang_name", type=str, default="Spanish")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--print_examples", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    main(parser.parse_args())
