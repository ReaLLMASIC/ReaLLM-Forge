# huggingface_model/gemma/270M/finetune.py
# Prevent GPU OOM on some systems
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import argparse
import torch.nn.functional as F
from typing import Callable
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)


def _relu_variant(inputs: torch.Tensor, activation: str, divisor: float) -> torch.Tensor:
    if divisor <= 0:
        raise ValueError("--activation_divisor must be > 0.")
    if activation == "relumax":
        return torch.relu(inputs) / divisor
    if activation == "relu2max":
        return torch.relu(inputs).pow(2) / divisor
    raise ValueError(f"Unsupported activation '{activation}'")


class BlendController:
    def __init__(self, alpha_start: float, alpha_end: float, anneal_steps: int, post_zero_steps: int):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.anneal_steps = max(1, anneal_steps)
        self.post_zero_steps = max(0, post_zero_steps)
        self.alpha = alpha_start

    def update(self, step: int):
        progress = min(step, self.anneal_steps) / float(self.anneal_steps)
        self.alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * progress
        self.alpha = max(self.alpha, 0.0)  # clamp; cannot go below zero


class AlphaScheduleCallback(TrainerCallback):
    def __init__(self, blend_controller: BlendController, log_every: int = 100):
        self.blend_controller = blend_controller
        self.log_every = max(1, log_every)

    def on_step_begin(self, args, state, control, **kwargs):
        self.blend_controller.update(state.global_step)
        if state.global_step % self.log_every == 0 and state.is_world_process_zero:
            print(f"[alpha schedule] step={state.global_step} alpha={self.blend_controller.alpha:.6f}")


class SampleOutputCallback(TrainerCallback):
    """A callback that generates a sample translation periodically."""
    def __init__(self, tokenizer, source_lang="English", target_lang="Spanish", test_sentence="The sun is shining today."):
        self.tokenizer = tokenizer
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.test_sentence = test_sentence
        self.prompt = (
            f"Translate {self.source_lang} to {self.target_lang}:\n"
            f"{self.source_lang}: {self.test_sentence}\n"
            f"{self.target_lang}: "
        )

    def on_log(self, args, state, control, **kwargs):
        # This callback is triggered every `logging_steps`.
        model = kwargs.get('model')
        if model and state.is_world_process_zero:
            print(f"\n--- Sample output at step {state.global_step} ---")
            
            # Generate output using the current state of the model
            inputs = self.tokenizer(self.prompt, return_tensors="pt").to(model.device)
            # Use eos_token_id for pad_token_id to avoid warnings
            outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=self.tokenizer.eos_token_id)
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean and print the generated text
            generated_text = decoded_output.replace(self.prompt, "").strip()
            print(f"{self.source_lang}: {self.test_sentence}")
            print(f"Generated {self.target_lang}: {generated_text}")
            print("---------------------------------------\n")

def _build_attention_softmax(
    args,
    blend_controller: BlendController,
    original_softmax: Callable,
):
    def _replacement(x, dim=None, _stacklevel=3, dtype=None):
        dim = -1 if dim is None else dim
        x_in = x.to(dtype) if dtype is not None else x
        softmax_scores = original_softmax(x_in, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

        if args.attention_mode == "softmax":
            return softmax_scores

        relu_scores = _relu_variant(x_in, args.attention_activation, args.activation_divisor)

        if args.attention_mode == "sum":
            return softmax_scores + relu_scores

        if args.attention_mode == "gradual_blend":
            alpha = blend_controller.alpha
            return alpha * softmax_scores + (1.0 - alpha) * relu_scores

        raise ValueError(f"Unsupported --attention_mode: {args.attention_mode}")

    return _replacement


def _patch_output_norms(model, blend_controller: BlendController, enable_norm_blend: bool):
    if not enable_norm_blend:
        return []

    patched = []
    target_fragments = ("post_attention_layernorm", "post_feedforward_layernorm")
    for name, module in model.named_modules():
        if not any(fragment in name for fragment in target_fragments):
            continue
        original_forward = module.forward

        def wrapped_forward(hidden_states, *f_args, __orig=original_forward, **f_kwargs):
            normalized = __orig(hidden_states, *f_args, **f_kwargs)
            alpha = blend_controller.alpha
            return alpha * normalized + (1.0 - alpha) * hidden_states

        module.forward = wrapped_forward
        patched.append((name, module, original_forward))

    return patched


def _restore_output_norms(patched_norms):
    for _, module, original_forward in patched_norms:
        module.forward = original_forward


def _print_multishot_examples(model, tokenizer, args):
    prompts_en = [
        "The weather is perfect for a walk in the park.",
        "Please summarize the main argument in this paragraph.",
        "I forgot my charger at home, so my laptop battery is low.",
    ]
    print("\n=== Multi-shot EN->ES examples ===")
    for idx, sentence in enumerate(prompts_en, start=1):
        prompt = (
            f"Translate {args.source_lang_name} to {args.target_lang_name}:\n"
            f"{args.source_lang_name}: {sentence}\n"
            f"{args.target_lang_name}: "
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = decoded_output.replace(prompt, "").strip().split("\n")[0]
        print(f"[{idx}] EN: {sentence}")
        print(f"    ES: {generated}")

        

def main(args):
    print("Loading the dataset...")
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # 2. Load the tokenizer and model
    model_name = args.model_name
    print(f"Loading tokenizer and model for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager") # "eager" is the recommended setting for Gemma 270M

    # FIX: Set pad_token to eos_token for decoder-only models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Preprocess the data
    def preprocess_function(examples):
        texts = [
            (
                f"Translate {args.source_lang_name} to {args.target_lang_name}:\n"
                f"{args.source_lang_name}: {ex[args.source_lang]}\n"
                f"{args.target_lang_name}: {ex[args.target_lang]}"
            )
            for ex in examples["translation"]
        ]
        return tokenizer(texts, truncation=True, max_length=128, padding="max_length")

    print("Tokenizing the dataset...")
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

    # 4. Define training arguments from command-line args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        # Control training with steps instead of epochs
        max_steps=args.total_iterations,
        # Set strategies to 'steps' to use the step-based arguments
        logging_strategy="steps",
        eval_strategy="steps",
        save_strategy="steps",
        # Use the provided frequency for logging, eval, and saving
        logging_steps=args.sample_frequency,
        eval_steps=args.sample_frequency,
        save_steps=args.sample_frequency,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
    )

    # 5. Create data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    blend_controller = BlendController(
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        anneal_steps=max(1, args.total_iterations - args.post_zero_steps),
        post_zero_steps=args.post_zero_steps,
    )

    sample_callback = SampleOutputCallback(
        tokenizer=tokenizer,
        source_lang=args.source_lang_name,
        target_lang=args.target_lang_name,
    )
    callbacks = [sample_callback]
    if args.attention_mode == "gradual_blend":
        callbacks.append(AlphaScheduleCallback(blend_controller, log_every=args.sample_frequency))

    # 6. Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks
    )

    # 7. Start training
    print("Starting the training process...")
    original_softmax = F.softmax
    replacement_softmax = _build_attention_softmax(args, blend_controller, original_softmax)
    F.softmax = replacement_softmax
    patched_norms = _patch_output_norms(
        model=model,
        blend_controller=blend_controller,
        enable_norm_blend=args.blend_output_norm and args.attention_mode == "gradual_blend",
    )
    try:
        trainer.train()
    finally:
        F.softmax = original_softmax
        _restore_output_norms(patched_norms)

    # Save the final model state
    trainer.save_model()
    print("Training complete! Final model saved.")
    if args.print_multishot_after_train:
        _print_multishot_examples(model, tokenizer, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Gemma model for translation.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Helsinki-NLP/opus-100",
        help="HF datasets repository id."
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="en-es",
        help="HF dataset config name (e.g. 'en-es')."
    )
    parser.add_argument(
        "--source_lang",
        type=str,
        default="en",
        help="Source language key inside examples['translation']."
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        default="es",
        help="Target language key inside examples['translation']."
    )
    parser.add_argument(
        "--source_lang_name",
        type=str,
        default="English",
        help="Human-friendly source language label used in prompts."
    )
    parser.add_argument(
        "--target_lang_name",
        type=str,
        default="Spanish",
        help="Human-friendly target language label used in prompts."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-270m",
        help="HF model id or local checkpoint path."
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train[:10%]",
        help="Datasets split string passed to load_dataset for OPUS-100."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gemma-3-270m-opus-100-en-es-causal",
        help="Directory to save checkpoints and final model."
    )
    parser.add_argument(
        "--total_iterations",
        type=int,
        default=5000,
        help="Total number of training steps (iterations) to perform."
    )
    parser.add_argument(
        "--sample_frequency",
        type=int,
        default=1000,
        help="How often (in steps) to save, evaluate, and generate a sample output."
    )
    parser.add_argument(
        "--attention_activation",
        type=str,
        default="relumax",
        choices=["relumax", "relu2max"],
        help="Non-softmax branch used in attention_mode=sum or gradual_blend."
    )
    parser.add_argument(
        "--activation_divisor",
        type=float,
        default=256.0,
        help="Divisor applied for relumax/relu2max activation variants."
    )
    parser.add_argument(
        "--attention_mode",
        type=str,
        default="softmax",
        choices=["softmax", "sum", "gradual_blend"],
        help="softmax baseline, sum baseline (softmax + relu*), or alpha-annealed blend."
    )
    parser.add_argument("--alpha_start", type=float, default=1.0, help="Initial alpha for gradual_blend.")
    parser.add_argument("--alpha_end", type=float, default=0.0, help="Final alpha for gradual_blend.")
    parser.add_argument(
        "--post_zero_steps",
        type=int,
        default=0,
        help="Optional additional steps after alpha reaches 0 (uses clamped alpha=0).",
    )
    parser.add_argument(
        "--blend_output_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For gradual_blend: blend post-attn/post-ffn output norms with raw residual path.",
    )
    parser.add_argument(
        "--print_multishot_after_train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print 3 fixed EN->ES translation outputs after training.",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
