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
OUTPUT_DIR = "./qwen-nano-siglip2"
BATCH_SIZE = 2
ACCUMULATION_STEPS = 8
LR = 5e-4
EPOCHS = 10
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
        print("⚠️ Hugging Face token not found. Gated Qwen models or datasets may require it.")
        hf_token = getpass.getpass("Please enter your HF token (input will be hidden): ")
        login(token=hf_token)
        print("✅ Token saved locally.")
    else:
        print("✅ Hugging Face token found! Already authenticated.")
        login(token=current_token)

# ==========================================
# MODEL DEFINITION
# ==========================================
class QwenNano(nn.Module):
    def __init__(self, model_id="Qwen/Qwen3.5-0.6B", train_mode="default"):
        super().__init__()
        self.train_mode = train_mode
        self.model_id = model_id
        print(f"🧠 Initializing Model with Train Mode: {self.train_mode.upper()}")

        print("Loading Vision: SigLIP 2 Base...")
        self.vision_tower = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224")
        self.vision_tower.requires_grad_(False)
        self.vision_hidden_size = self.vision_tower.config.hidden_size

        print(f"Loading Text: {self.model_id}...")
        self.llm = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True)
        self.text_hidden_size = self.llm.config.hidden_size
        self.llm.requires_grad_(False)

        # 1. Projector for Dense Patches (AR Decoder Path)
        self.projector = nn.Sequential(
            nn.LayerNorm(self.vision_hidden_size),
            nn.Linear(self.vision_hidden_size, self.text_hidden_size),
            nn.LayerNorm(self.text_hidden_size)
        )

        # 2. Projector for MAP Head (Zero-Shot Contrastive Path & Global Token Prompting)
        self.map_projector = nn.Sequential(
            nn.LayerNorm(self.vision_hidden_size),
            nn.Linear(self.vision_hidden_size, self.text_hidden_size),
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
            # Dynamic Injection Logic (BOS vs No-BOS)
            # ==========================================
            has_bos = False
            # Check dynamically if sequence uses a BOS token
            if hasattr(self.llm.config, 'bos_token_id') and self.llm.config.bos_token_id is not None:
                if input_ids.shape[1] > 0 and input_ids[0, 0] == self.llm.config.bos_token_id:
                    has_bos = True

            if has_bos:
                # Sandwich the embedding strictly AFTER the BOS token
                first_embeds = inputs_embeds[:, :1, :]
                rest_embeds = inputs_embeds[:, 1:, :]
                inputs_embeds = torch.cat([first_embeds, image_embeds, rest_embeds], dim=1)

                if attention_mask is not None:
                    first_mask = attention_mask[:, :1]
                    rest_mask = attention_mask[:, 1:]
                    dummy_mask = torch.ones((attention_mask.shape[0], image_embeds.shape[1]), device=attention_mask.device, dtype=attention_mask.dtype)
                    attention_mask = torch.cat([first_mask, dummy_mask, rest_mask], dim=1)

                if labels is not None:
                    first_labels = labels[:, :1]
                    rest_labels = labels[:, 1:]
                    dummy_labels = torch.full((labels.shape[0], image_embeds.shape[1]), -100, device=labels.device, dtype=labels.dtype)
                    labels = torch.cat([first_labels, dummy_labels, rest_labels], dim=1)
            else:
                # No BOS token detected -> Prepend at the absolute beginning of sequence
                inputs_embeds = torch.cat([image_embeds, inputs_embeds], dim=1)

                if attention_mask is not None:
                    dummy_mask = torch.ones((attention_mask.shape[0], image_embeds.shape[1]), device=attention_mask.device, dtype=attention_mask.dtype)
                    attention_mask = torch.cat([dummy_mask, attention_mask], dim=1)

                if labels is not None:
                    dummy_labels = torch.full((labels.shape[0], image_embeds.shape[1]), -100, device=labels.device, dtype=labels.dtype)
                    labels = torch.cat([dummy_labels, labels], dim=1)

        # Main AR Decoder Forward Pass
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, return_dict=True)

        # ==========================================
        # MULTI-TASK LOSS: Add SigLIP Contrastive Loss
        # ==========================================
        if labels is not None and vision_outputs is not None and caption_input_ids is not None:
            # 1. Image Global Embedding (Normalize the already projected MAP token)
            image_global_embed = F.normalize(raw_map_embed, dim=-1)

            # 2. Text Global Embedding (Raw Qwen token embeddings mean-pooled)
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

def prepare_data(batch, tokenizer, processor, prompt_prefix, eos_str):
    images = [x.convert("RGB") for x in batch['image']]
    pixel_values = processor(images=images, return_tensors="pt").pixel_values

    input_ids_list, attention_mask_list, labels_list = [], [], []

    # Accurate prompt length logic for dynamic masking
    prompt_ids = tokenizer(prompt_prefix, add_special_tokens=False).input_ids
    prompt_mask_len = len(prompt_ids)
    tokenizer.padding_side = "right"

    for t in batch['text']:
        full_text = f"{prompt_prefix}{t}{eos_str}"
        tokens = tokenizer(full_text, truncation=True, max_length=128, padding="max_length", return_tensors="pt", add_special_tokens=False)
        ids = tokens.input_ids[0]
        mask = tokens.attention_mask[0]
        
        # Mask out prefix logic so loss is only calculated on the response target
        lbls = ids.clone()
        lbls[:prompt_mask_len] = -100
        lbls[mask == 0] = -100

        input_ids_list.append(ids)
        attention_mask_list.append(mask)
        labels_list.append(lbls)

    raw_captions = [str(t) for t in batch['text']]
    caption_tokens = tokenizer(raw_captions, max_length=64, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=False)

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "pixel_values": pixel_values,
        "labels": torch.stack(labels_list),
        "caption_input_ids": caption_tokens.input_ids,
        "caption_attention_mask": caption_tokens.attention_mask
    }

def train_nano(train_mode="default", model_id="Qwen/Qwen3.5-0.6B"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Qwen tokenizers generally lack a dedicated pad_token mapped initially. Enforce fallback.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>"
        
    processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")

    # Pass the selected mode into our custom architecture
    model = QwenNano(model_id=model_id, train_mode=train_mode)

    print("Loading 30k COCO Dataset for Real Pre-Training...")
    dataset = load_dataset("sayakpaul/coco-30-val-2014", split="train")
    split_ds = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)

    # Autodetect format: Instruct Models use ChatML, Base Models use clean text continuation
    is_instruct = "instruct" in model_id.lower() or "chat" in model_id.lower()
    
    if is_instruct:
        prompt_prefix = "<|im_start|>user\nDescribe this image.<|im_end|>\n<|im_start|>assistant\n"
        eos_str = "<|im_end|>"
        print("💬 Using ChatML Instruct formatting.")
    else:
        prompt_prefix = "User: Describe this image.\nAssistant: "
        eos_str = tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>"
        print("📝 Using Base text formatting.")

    def collate_fn(batch):
        texts = [item.get('text', item.get('caption', 'A photo.')) for item in batch]
        texts = [t[0] if isinstance(t, list) else t for t in texts]
        return prepare_data(
            {'image': [item['image'] for item in batch], 'text': texts}, 
            tokenizer, 
            processor,
            prompt_prefix,
            eos_str
        )

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
    parser = argparse.ArgumentParser(description="Train QwenNano")
    parser.add_argument("--mode", type=str, choices=["train", "validate", "both"], default="train")
    parser.add_argument("--train_mode", type=str, choices=["default", "dropout", "map_only"], default="default",
                        help="Select how image embeddings are fed to the LLM during training.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-0.6B",
                        help="Hugging Face Model ID (e.g., 'Qwen/Qwen3.5-0.6B' or 'Qwen/Qwen3.5-0.6B-Instruct')")
    args = parser.parse_args()

    authenticate_hf()

    if args.mode in ["train", "both"]:
        print(f"\n>>> INITIATING TRAINING PHASE (Mode: {args.train_mode}, Model: {args.model_id}) <<<")
        train_nano(train_mode=args.train_mode, model_id=args.model_id)
