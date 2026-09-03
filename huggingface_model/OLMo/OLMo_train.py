import argparse, torch
from transformers import AutoTokenizer, OlmoForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from OLMo_model import apply_custom_attention
from OLMo_train_args import parse_args

def main():
    args = parse_args()

    # Load the pretrained model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")
    model = OlmoForCausalLM.from_pretrained("allenai/OLMo-1B-0724-hf")

    # Apply the custom attention modifications based on the chosen variant.
    if args.clip:
        # You can set specific parameters for clipped variant.
        model = apply_custom_attention(model, clip=True, gate=args.gate, obo_variant=args.obo_variant, attention_variant=args.model_variant, zeta=1.0, gamma=-0.03)
    else:
        model = apply_custom_attention(model, clip=False, gate=args.gate, obo_variant=args.obo_variant, attention_variant=args.model_variant)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Add a new pad token if the tokenizer doesn't have one
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Tokenize the dataset and add labels for language modeling
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256
        )
        # Replace pad tokens with -100 so they are ignored in loss computation.
        tokenized_inputs["labels"] = [
            [token if token != tokenizer.pad_token_id else -100 for token in input_ids]
            for input_ids in tokenized_inputs["input_ids"]
        ]
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(eval_dataset))

    def filter_valid_examples(example):
        # Count valid tokens (i.e., tokens not equal to -100)
        valid_token_count = sum(1 for token in example["labels"] if token != -100)
        return valid_token_count > 0

    # Filter the evaluation dataset
    filtered_eval_dataset = eval_dataset.filter(filter_valid_examples)
    print("Filtered eval dataset size:", len(filtered_eval_dataset))

    # Create a data collator with dynamic padding (reduces wasted memory on padded tokens)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model_variant = (
        ("clipped_" if args.clip else "") +
        ("gated_" if args.gate else "") +
        (args.obo_variant + "_" if args.obo_variant != "none" else "") +
        args.model_variant
    )
    output_dir = "olmo-" + model_variant
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_dir="./logs",
        logging_steps=100,
    )

    # Initialize the Trainer with the data collator for dynamic padding
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=filtered_eval_dataset,
        data_collator=data_collator,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    save_dir = "./fine-tuned-olmo-" + model_variant
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    main()