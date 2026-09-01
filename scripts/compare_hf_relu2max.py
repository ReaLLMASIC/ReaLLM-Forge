#!/usr/bin/env python3
"""Run matched Hugging Face Trainer pre-training trials for ReLU2Max/softmax."""

import argparse
import gc
import importlib
import importlib.util
import inspect
import json
import random
from pathlib import Path
import sys

# When this file is invoked as documented (``python scripts/<file>.py``),
# Python puts ``scripts/`` rather than the repository root on sys.path. Add the
# root before importing the local ``hf_model`` and ``train_variations``
# packages. Module execution (``python -m scripts.compare_hf_relu2max``) keeps
# working as well.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from hf_model import NanoGPTConfig, NanoGPTForCausalLM
from train_variations.muon import SingleDeviceMuonWithAuxAdam


class MuonTrainer(Trainer):
    """Trainer using nanoGPT's Muon matrix path and auxiliary Adam path."""

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        excludes = ("embed", "lm_head")
        muon, adam = [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            destination = muon if parameter.ndim >= 2 and not any(x in name for x in excludes) else adam
            destination.append(parameter)
        self.optimizer = SingleDeviceMuonWithAuxAdam([
            {"params": adam, "use_muon": False, "lr": self.args.learning_rate,
             "betas": (self.args.adam_beta1, self.args.adam_beta2),
             "eps": self.args.adam_epsilon, "weight_decay": self.args.weight_decay},
            {"params": muon, "use_muon": True, "lr": self.args.learning_rate,
             "momentum": self.args.muon_momentum, "ns_steps": self.args.muon_ns_steps,
             "nesterov": True, "weight_decay": self.args.weight_decay},
        ])
        return self.optimizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--dataset-config")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--output-dir", default="hf_comparison")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--pack-text", action="store_true",
                        help="Concatenate and chunk tokens into full block-size examples (recommended for pre-training)")
    parser.add_argument("--append-eos", action=argparse.BooleanOptionalAction, default=True,
                        help="Append EOS to each document before packing")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, help="Defaults to one tenth of max steps")
    parser.add_argument("--save-steps", type=int, default=0, help="Checkpoint interval; zero disables intermediate saves")
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", action="store_true",
                        help="Resume each variation from its newest checkpoint when present")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--kv-heads", type=int, default=6)
    parser.add_argument("--rope-length", type=int)
    parser.add_argument("--relu2max-divisor", type=float, default=256.0)
    parser.add_argument("--relu2max-divide-by-sequence-length", action="store_true")
    parser.add_argument("--relu2max-accelerator", choices=["auto", "torch", "triton"], default="auto")
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--report-to", nargs="*", default=[])
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--lm-eval-tasks", nargs="*", default=[],
                        help="lm-evaluation-harness tasks to run after each variation")
    parser.add_argument("--lm-eval-batch-size", default="auto")
    parser.add_argument("--lm-eval-limit", type=float,
                        help="Optional sample count/fraction for smoke runs; omit for publishable scores")
    return parser.parse_args()


def tokenize_dataset(args, tokenizer):
    raw = load_dataset(args.dataset, args.dataset_config)
    train = raw["train"]
    if args.train_samples > 0:
        train = train.select(range(min(args.train_samples, len(train))))
    eval_split = raw.get("validation", raw.get("test"))
    if eval_split is None:
        split = train.train_test_split(test_size=min(args.eval_samples, max(1, len(train) // 10)), seed=args.seed)
        train, eval_split = split["train"], split["test"]
    else:
        if args.eval_samples > 0:
            eval_split = eval_split.select(range(min(args.eval_samples, len(eval_split))))

    def encode(batch):
        encoded = tokenizer(
            batch[args.text_column],
            truncation=not args.pack_text,
            max_length=args.block_size if not args.pack_text else None,
            add_special_tokens=False,
        )
        if args.append_eos:
            if tokenizer.eos_token_id is None:
                raise ValueError("--append-eos requires a tokenizer with an EOS token")
            for input_ids, attention_mask in zip(encoded["input_ids"], encoded["attention_mask"]):
                input_ids.append(tokenizer.eos_token_id)
                attention_mask.append(1)
        return encoded

    remove_train = train.column_names
    remove_eval = eval_split.column_names
    train = train.map(encode, batched=True, remove_columns=remove_train)
    eval_split = eval_split.map(encode, batched=True, remove_columns=remove_eval)
    if args.pack_text:
        def group_texts(batch):
            concatenated = {key: sum(batch[key], []) for key in batch}
            usable = len(concatenated["input_ids"]) // args.block_size * args.block_size
            return {
                key: [tokens[i : i + args.block_size] for i in range(0, usable, args.block_size)]
                for key, tokens in concatenated.items()
            }

        train = train.map(group_texts, batched=True)
        eval_split = eval_split.map(group_texts, batched=True)
    return train, eval_split


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluation_strategy_argument():
    """Return the evaluation keyword supported by this Transformers version.

    Transformers renamed ``evaluation_strategy`` to ``eval_strategy`` and
    eventually removed the former spelling. The repository supports both the
    pinned 4.44 release and current releases, so select the spelling from the
    installed constructor rather than relying on a version-number comparison.
    """
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        return {"eval_strategy": "steps"}
    return {"evaluation_strategy": "steps"}


def run_lm_eval(model, tokenizer, args):
    """Evaluate an in-memory custom model with lm-evaluation-harness."""
    if not args.lm_eval_tasks:
        return None
    if importlib.util.find_spec("lm_eval") is None:
        raise RuntimeError(
            "--lm-eval-tasks requires lm-evaluation-harness; install it with "
            "`pip install lm-eval`"
        )
    evaluator = importlib.import_module("lm_eval.evaluator")
    huggingface = importlib.import_module("lm_eval.models.huggingface")
    wrapped_model = huggingface.HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.lm_eval_batch_size,
        device=str(next(model.parameters()).device),
    )
    evaluation = evaluator.simple_evaluate(
        model=wrapped_model,
        tasks=args.lm_eval_tasks,
        batch_size=args.lm_eval_batch_size,
        limit=args.lm_eval_limit,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )
    # The full object contains non-JSON configuration objects. Task metrics are
    # sufficient for a stable comparison artifact.
    return evaluation["results"]


def newest_checkpoint(run_dir):
    checkpoints = []
    for path in run_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    return str(max(checkpoints)[1]) if checkpoints else None


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset, eval_dataset = tokenize_dataset(args, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    results = {}

    # Re-seeding before each construction gives identical initial tensors and
    # Trainer sampling order; only the attention normalization differs.
    for normalizer in ("relu2max", "softmax"):
        seed_everything(args.seed)
        config = NanoGPTConfig(
            vocab_size=len(tokenizer), max_position_embeddings=args.block_size,
            hidden_size=args.hidden_size, intermediate_size=args.intermediate_size,
            num_hidden_layers=args.layers, num_attention_heads=args.heads,
            num_key_value_heads=args.kv_heads, rope_length=args.rope_length,
            attention_normalizer=normalizer, relu2max_divisor=args.relu2max_divisor,
            relu2max_divide_by_sequence_length=args.relu2max_divide_by_sequence_length,
            relu2max_accelerator=args.relu2max_accelerator,
        )
        model = NanoGPTForCausalLM(config)
        run_dir = root / normalizer
        training_args = TrainingArguments(
            output_dir=str(run_dir), overwrite_output_dir=True, max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps, lr_scheduler_type="cosine", logging_steps=10,
            eval_steps=args.eval_steps or max(1, args.max_steps // 10),
            save_strategy="steps" if args.save_steps > 0 else "no",
            save_steps=max(1, args.save_steps), save_total_limit=args.save_total_limit,
            seed=args.seed, data_seed=args.seed, fp16=args.fp16,
            bf16=args.bf16, tf32=args.tf32, report_to=args.report_to,
            dataloader_num_workers=args.dataloader_workers,
            **evaluation_strategy_argument(),
        )
        # Custom attributes consumed by MuonTrainer.create_optimizer.
        training_args.muon_momentum = args.muon_momentum
        training_args.muon_ns_steps = args.muon_ns_steps
        trainer = MuonTrainer(model=model, args=training_args, train_dataset=train_dataset,
                              eval_dataset=eval_dataset, data_collator=collator)
        resume_checkpoint = newest_checkpoint(run_dir) if args.resume_from_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        eval_result = trainer.evaluate()
        trainer.save_model()
        tokenizer.save_pretrained(run_dir)
        benchmark_results = run_lm_eval(model, tokenizer, args)
        results[normalizer] = {
            "training": {**train_result.metrics, **eval_result},
            "lm_eval": benchmark_results,
        }
        with (run_dir / "results.json").open("w") as handle:
            json.dump(results[normalizer], handle, indent=2, sort_keys=True)
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (root / "comparison.json").open("w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
