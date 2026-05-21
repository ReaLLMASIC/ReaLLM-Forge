import os
import torch
import argparse
from PIL import Image
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoImageProcessor
from gemma_siglip2_finetune import Gemma3Nano

MODEL_WEIGHTS = "./gemma-nano-siglip2/best_pytorch_model.bin"

# ==========================================
# MODE 1: ZERO-SHOT CATEGORY PREDICTION
# ==========================================
def run_category_mode(model, processor, tokenizer, device, image, concepts):
    print(f"\n🐾 MODE: Zero-Shot Categorization")
    print(f"🎯 Target Concepts: {concepts}")
    print("-" * 50)

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device, dtype=torch.bfloat16)
    gemma_embeddings_layer = model.llm.get_input_embeddings()

    with torch.no_grad():
        vision_outputs = model.vision_tower(pixel_values=pixel_values)
        image_concept_embed = model.map_projector(vision_outputs.pooler_output)
        image_concept_embed = F.normalize(image_concept_embed, p=2, dim=-1)

        results = {}
        for concept in concepts:
            # Leading space is critical for sentencepiece tokenizers
            tokens = tokenizer(f" {concept}", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            concept_embed = gemma_embeddings_layer(tokens).mean(dim=1)
            concept_embed = F.normalize(concept_embed, p=2, dim=-1)

            score = torch.matmul(image_concept_embed, concept_embed.T).item()
            results[concept] = score

    # Sort and display results
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for concept, score in sorted_results:
        print(f"➤ Similarity to '{concept}': {score:.4f}")

    print(f"\n🏆 Top Prediction: {sorted_results[0][0].upper()}")
    print("-" * 50)

# ==========================================
# MODE 2: DENSE CAPTIONING (AR Decoder)
# ==========================================
def run_caption_mode(model, processor, tokenizer, device, image, query):
    print(f"\n✨ MODE: Dense Auto-Regressive Captioning")
    print(f"🗣️  User Query: '{query}'")
    print("-" * 50)

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device, dtype=torch.bfloat16)
    prompt = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        vision_outputs = model.vision_tower(pixel_values=pixel_values)
        # Grab the dense sequence of patches (e.g., 196 tokens)
        image_embeds = model.projector(vision_outputs.last_hidden_state)

        text_embeds = model.llm.get_input_embeddings()(input_ids)

        bos_embeds = text_embeds[:, :1, :]
        rest_embeds = text_embeds[:, 1:, :]
        inputs_embeds = torch.cat([bos_embeds, image_embeds, rest_embeds], dim=1)

        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)

        output_ids = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=60,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )

    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    print(f"🤖 Output: {prediction}\n")

# ==========================================
# MODE 3: MAP-PROMPTING (Global Token)
# ==========================================
def run_map_query_mode(model, processor, tokenizer, device, image, query):
    print(f"\n🧠 MODE: MAP-Prompting (Single Global Token)")
    print(f"🗣️  User Query: '{query}'")
    print("-" * 50)

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device, dtype=torch.bfloat16)
    prompt = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        vision_outputs = model.vision_tower(pixel_values=pixel_values)

        # Grab the pooled MAP head output and project it
        # Shape goes from [1, 2048] to [1, 1, 2048] to act as a single token in the sequence!
        image_embeds = model.map_projector(vision_outputs.pooler_output).unsqueeze(1)

        text_embeds = model.llm.get_input_embeddings()(input_ids)

        # Sandwich that single "super-token" into the text
        bos_embeds = text_embeds[:, :1, :]
        rest_embeds = text_embeds[:, 1:, :]
        inputs_embeds = torch.cat([bos_embeds, image_embeds, rest_embeds], dim=1)

        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)

        output_ids = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=60,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )

    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    print(f"🤖 Output: {prediction}\n")

# ==========================================
# MAIN EXECUTION ROUTER
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Inference Script")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("--mode", type=str, required=True, choices=["category", "caption", "map_query"],
                        help="Choose which inference mode to run.")
    parser.add_argument("--concepts", type=str, nargs='+', default=["cat", "dog"],
                        help="List of concepts to score against (Only used in 'category' mode).")
    parser.add_argument("--query", type=str, default="Describe this image.",
                        help="Text prompt to send to the LLM (Used in 'caption' and 'map_query' modes).")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(MODEL_WEIGHTS):
        print(f"❌ Error: Could not find weights at {MODEL_WEIGHTS}. Have you trained the model yet?")
        return

    print("📦 Loading tokenizer, processor, and model...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")

    model = Gemma3Nano()
    state_dict = torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    model.to(device, dtype=torch.bfloat16)
    model.eval()

    try:
        image = Image.open(args.image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    # Route to the correct mode
    if args.mode == "category":
        run_category_mode(model, processor, tokenizer, device, image, args.concepts)
    elif args.mode == "caption":
        run_caption_mode(model, processor, tokenizer, device, image, args.query)
    elif args.mode == "map_query":
        run_map_query_mode(model, processor, tokenizer, device, image, args.query)

if __name__ == "__main__":
    main()
