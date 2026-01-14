# huggingface_model/gemma/270M/finetune.py
# Prevent GPU OOM on some systems
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)

# --- Custom Callback for Generating Sample Outputs ---
class SampleOutputCallback(TrainerCallback):
    """A callback that generates a sample translation periodically."""
    def __init__(self, tokenizer, test_sentence="The sun is shining today."):
        self.tokenizer = tokenizer
        self.test_sentence = test_sentence
        self.prompt = f"Translate English to Chinese:\nEnglish: {self.test_sentence}\nChinese: "

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
            print(f"English: {self.test_sentence}")
            print(f"Generated Chinese: {generated_text}")
            print("---------------------------------------\n")

# --- Main Script ---
def main(args):
    # 1. Load the dataset
    print("Loading the dataset...")
    # Using a larger portion for more meaningful training
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-zh", split="train[:10%]")

    # Split the dataset
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # 2. Load the tokenizer and model
    model_name = "google/gemma-3-270m"
    print(f"Loading tokenizer and model for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager") # "eager" is the recommended setting for Gemma 270M

    # FIX: Set pad_token to eos_token for decoder-only models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Preprocess the data
    def preprocess_function(examples):
        texts = [f"Translate English to Chinese:\nEnglish: {ex['en']}\nChinese: {ex['zh']}" for ex in examples["translation"]]
        return tokenizer(texts, truncation=True, max_length=128, padding="max_length")

    print("Tokenizing the dataset...")
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

    # 4. Define training arguments from command-line args
    training_args = TrainingArguments(
        output_dir="./gemma-3-270m-opus-100-en-zh-causal",
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

    # Instantiate the custom callback
    sample_callback = SampleOutputCallback(tokenizer=tokenizer)

    # 6. Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[sample_callback] # Add the callback here
    )

    # 7. Start training
    print("Starting the training process...")
    trainer.train()

    # Save the final model state
    trainer.save_model()
    print("Training complete! Final model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Gemma model for translation.")
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
    parsed_args = parser.parse_args()
    main(parsed_args)

