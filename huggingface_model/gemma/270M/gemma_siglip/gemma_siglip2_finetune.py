import os
import random
import argparse
import getpass
import torch
import torch.nn as nn
import torch.nn.functional as F
import climage
from datasets import load_dataset
from huggingface_hub import login, get_token
from transformers import (
    AutoModelForCausalLM,
    SiglipVisionModel,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoImageProcessor,
    TrainerCallback
)

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
OUTPUT_DIR = "./gemma-nano-siglip2"
BATCH_SIZE = 2
ACCUMULATION_STEPS = 8
LR = 5e-4
EPOCHS = 1
MODEL_WEIGHTS = f"{OUTPUT_DIR}/best_pytorch_model.bin"
NUM_TESTS = 5
OUTPUT_IMG_DIR = "./validation_images"
TEST_SIZE = 400
SEED = 42

# ==========================================
# HUGGING FACE AUTHENTICATION
# ==========================================
def authenticate_hf():
    current_token = get_token()
    if current_token is None:
        print("⚠️ Hugging Face token not found. Gemma 3 requires it.")
        hf_token = getpass.getpass("Please enter your HF token (input will be hidden): ")
        login(token=hf_token)
        print("✅ Token saved locally.")
    else:
        print("✅ Hugging Face token found! Already authenticated.")
        login(token=current_token)

authenticate_hf()

# ==========================================
# MODEL DEFINITION
# ==========================================
class Gemma3Nano(nn.Module):
    def __init__(self, train_mode="default"):
        super().__init__()
        self.train_mode = train_mode
        print(f"🧠 Initializing Model with Train Mode: {self.train_mode.upper()}")

        print("Loading Vision: SigLIP 2 Base...")
        self.vision_tower = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224")
        self.vision_tower.requires_grad_(False)
        self.vision_hidden_size = self.vision_tower.config.hidden_size

        print("Loading Text: Gemma 3 270M...")
        self.llm = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")
        self.text_hidden_size = self.llm.config.hidden_size
        self.llm.requires_grad_(False)

        # 1. Projector for Dense Patches (AR Decoder Path)
        self.projector = nn.Sequential(
            nn.Linear(self.vision_hidden_size, self.text_hidden_size),
            nn.GELU(),
            nn.Linear(self.text_hidden_size, self.text_hidden_size),
            nn.LayerNorm(self.text_hidden_size)
        )

        # 2. Projector for MAP Head (Zero-Shot Contrastive Path & Global Token Prompting)
        self.map_projector = nn.Sequential(
            nn.Linear(self.vision_hidden_size, self.text_hidden_size),
            nn.GELU(),
            nn.Linear(self.text_hidden_size, self.text_hidden_size),
            nn.LayerNorm(self.text_hidden_size)
        )

        # 3. Learnable parameters for SigLIP Sigmoid Loss
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(10.0)))
        self.logit_bias = nn.Parameter(torch.ones([]) * -10.0)

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None, caption_input_ids=None, caption_attention_mask=None):
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        vision_outputs = None
        raw_map_embed = None

        if pixel_values is not None:
            with torch.no_grad():
                vision_outputs = self.vision_tower(pixel_values=pixel_values)
                image_features = vision_outputs.last_hidden_state

            # We always calculate the MAP embedding because we need it for the contrastive loss!
            raw_map_embed = self.map_projector(vision_outputs.pooler_output)

            # ==========================================
            # ROUTING LOGIC: Dense Patches vs MAP Token
            # ==========================================
            use_map_token = False

            if self.train_mode == "map_only":
                use_map_token = True
            elif self.train_mode == "dropout" and self.training:
                # 50% chance to drop the dense patches and rely entirely on the MAP token
                use_map_token = random.random() < 0.5

            if use_map_token:
                # Unsqueeze from [Batch, Hidden] -> [Batch, 1, Hidden] to act as a single token
                image_embeds = raw_map_embed.unsqueeze(1)
            else:
                # Use standard dense patches [Batch, 196, Hidden]
                image_embeds = self.projector(image_features)

            # ==========================================
            # Sandwich Logic
            # ==========================================
            bos_embeds = inputs_embeds[:, :1, :]
            rest_embeds = inputs_embeds[:, 1:, :]
            inputs_embeds = torch.cat([bos_embeds, image_embeds, rest_embeds], dim=1)

            if attention_mask is not None:
                bos_mask = attention_mask[:, :1]
                rest_mask = attention_mask[:, 1:]
                dummy_mask = torch.ones((attention_mask.shape[0], image_embeds.shape[1]), device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([bos_mask, dummy_mask, rest_mask], dim=1)

            if labels is not None:
                bos_labels = labels[:, :1]
                rest_labels = labels[:, 1:]
                dummy_labels = torch.full((labels.shape[0], image_embeds.shape[1]), -100, device=labels.device, dtype=labels.dtype)
                labels = torch.cat([bos_labels, dummy_labels, rest_labels], dim=1)

        # Main AR Decoder Forward Pass
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, return_dict=True)

        # ==========================================
        # MULTI-TASK LOSS: Add SigLIP Contrastive Loss
        # ==========================================
        if labels is not None and vision_outputs is not None and caption_input_ids is not None:
            # 1. Image Global Embedding (Normalize the already projected MAP token)
            image_global_embed = F.normalize(raw_map_embed, dim=-1)

            # 2. Text Global Embedding (Raw Gemma token embeddings mean-pooled)
            raw_text_embeds = self.llm.get_input_embeddings()(caption_input_ids)
            mask_expanded = caption_attention_mask.unsqueeze(-1).expand(raw_text_embeds.size()).float()
            sum_embeddings = torch.sum(raw_text_embeds * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            text_global_embed = sum_embeddings / sum_mask
            text_global_embed = F.normalize(text_global_embed, dim=-1)

            # 3. Pairwise Sigmoid Loss Calculation
            logits = torch.matmul(image_global_embed, text_global_embed.T) * self.logit_scale.exp() + self.logit_bias

            batch_size = logits.size(0)
            siglip_labels = 2 * torch.eye(batch_size, device=logits.device) - 1

            contrastive_loss = -torch.sum(F.logsigmoid(siglip_labels * logits)) / batch_size

            # Combine Auto-Regressive Loss (75%) and Contrastive Loss (25%)
            outputs.loss = outputs.loss + (0.33 * contrastive_loss)

        return outputs

# ==========================================
# TRAINING COMPONENTS
# ==========================================
class SaveBestModelCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.best_loss = float('inf')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.save_path = f"{self.output_dir}/best_pytorch_model.bin"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_loss = float(metrics.get('eval_loss', float('inf')))
        if eval_loss < self.best_loss:
            print(f"\n🌟 New best validation loss: {eval_loss:.4f}. Saving!")
            self.best_loss = eval_loss
            torch.save(kwargs['model'].state_dict(), self.save_path)

def prepare_data(batch, tokenizer, processor):
    images = [x.convert("RGB") for x in batch['image']]
    pixel_values = processor(images=images, return_tensors="pt").pixel_values

    prompt_prefix = "<start_of_turn>user\nDescribe this image.<end_of_turn>\n<start_of_turn>model\n"
    input_ids_list, attention_mask_list, labels_list = [], [], []

    prompt_ids = tokenizer(prompt_prefix, add_special_tokens=False).input_ids
    prompt_mask_len = 1 + len(prompt_ids)
    tokenizer.padding_side = "right"

    for t in batch['text']:
        full_text = f"{prompt_prefix}{t}<end_of_turn>"
        tokens = tokenizer(full_text, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        ids = tokens.input_ids[0]
        mask = tokens.attention_mask[0]
        lbls = ids.clone()
        lbls[:prompt_mask_len] = -100
        lbls[mask == 0] = -100

        input_ids_list.append(ids)
        attention_mask_list.append(mask)
        labels_list.append(lbls)

    raw_captions = [str(t) for t in batch['text']]
    caption_tokens = tokenizer(raw_captions, max_length=64, truncation=True, padding="max_length", return_tensors="pt")

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "pixel_values": pixel_values,
        "labels": torch.stack(labels_list),
        "caption_input_ids": caption_tokens.input_ids,
        "caption_attention_mask": caption_tokens.attention_mask
    }

def train_nano(train_mode="default"):
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    tokenizer.pad_token = tokenizer.eos_token
    processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")

    # Pass the selected mode into our custom architecture
    model = Gemma3Nano(train_mode=train_mode)

    print("Loading 30k COCO Dataset for Real Pre-Training...")
    dataset = load_dataset("sayakpaul/coco-30-val-2014", split="train")
    split_ds = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)

    def collate_fn(batch):
        texts = [item.get('text', item.get('caption', 'A photo.')) for item in batch]
        texts = [t[0] if isinstance(t, list) else t for t in texts]
        return prepare_data({'image': [item['image'] for item in batch], 'text': texts}, tokenizer, processor)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        learning_rate=LR,
        bf16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="no",
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_ds['train'],
        eval_dataset=split_ds['test'],
        data_collator=collate_fn,
        callbacks=[SaveBestModelCallback(OUTPUT_DIR)]
    )
    trainer.train()

def run_batch_validation():
    print("\nSkipping dense validation in this snippet for brevity, run inference.py to test!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gemma3Nano")
    parser.add_argument("--mode", type=str, choices=["train", "validate", "both"], default="train")
    parser.add_argument("--train_mode", type=str, choices=["default", "dropout", "map_only"], default="default",
                        help="Select how image embeddings are fed to the LLM during training.")
    args = parser.parse_args()

    if args.mode in ["train", "both"]:
        print(f"\n>>> INITIATING TRAINING PHASE (Mode: {args.train_mode}) <<<")
        train_nano(train_mode=args.train_mode)
