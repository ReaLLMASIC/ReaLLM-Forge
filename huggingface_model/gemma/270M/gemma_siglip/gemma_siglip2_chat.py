import os
import torch
import argparse
from PIL import Image
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoImageProcessor

from gemma_siglip2_finetune import Gemma3Nano

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_WEIGHTS = "./gemma-nano-siglip2/best_pytorch_model.bin"

def chat_with_image(image_path):
    # 1. Standard Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Initializing on {device.upper()}...")

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")

    model = Gemma3Nano()

    try:
        # strict=False ensures it loads even if some projector weights are newly initialized
        state_dict = torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    except FileNotFoundError:
        print(f"❌ Error: Could not find weights at {MODEL_WEIGHTS}.")
        return

    model.to(device, dtype=torch.bfloat16)
    model.eval()

    # 2. Process the Image ONCE
    print(f"🖼️  Analyzing image: {image_path}...")
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device, dtype=torch.bfloat16)

    with torch.no_grad():
        # Pass through Vision Tower
        vision_outputs = model.vision_tower(pixel_values=pixel_values)

        # Pre-compute BOTH representations so they are ready for the chat loop
        dense_image_embeds = model.projector(vision_outputs.last_hidden_state)

        # Extract MAP embedding and unsqueeze to make it a [1, 1, 2048] sequence token
        raw_map_embed = model.map_projector(vision_outputs.pooler_output)
        map_image_embeds = raw_map_embed.unsqueeze(1)

    # 3. Initialize Conversation Memory
    history = ""
    print("\n" + "="*60)
    print("💬 MULTI-MODE CHAT SESSION STARTED!")
    print("="*60)
    print("Available Commands:")
    print("  [Standard text]         -> Chat using Dense Patches (Default)")
    print("  /map <question>         -> Chat using the single Global MAP Token")
    print("  /category <word1> <...> -> Run zero-shot categorization")
    print("  quit or exit            -> End session")
    print("-" * 60)

    # 4. The Interactive Loop
    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        # Exit condition
        if user_input.lower() in ['quit', 'exit']:
            print("Ending chat. Goodbye! 👋")
            break

        # ==========================================
        # MODE: ZERO-SHOT CATEGORY (Does not affect chat history)
        # ==========================================
        if user_input.lower().startswith('/category'):
            parts = user_input.split()[1:] # Get concepts after '/category'
            if not parts:
                print("⚠️ Please provide concepts to categorize (e.g., /category cat dog bird)")
                continue

            print("\n🔍 Running Zero-Shot Categorization...")
            with torch.no_grad():
                normalized_map = F.normalize(raw_map_embed, p=2, dim=-1)
                gemma_embeddings_layer = model.llm.get_input_embeddings()

                results = {}
                for concept in parts:
                    tokens = tokenizer(f" {concept}", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                    concept_embed = gemma_embeddings_layer(tokens).mean(dim=1)
                    concept_embed = F.normalize(concept_embed, p=2, dim=-1)
                    score = torch.matmul(normalized_map, concept_embed.T).item()
                    results[concept] = score

                sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
                for concept, score in sorted_results:
                    print(f"  ➤ Similarity to '{concept}': {score:.4f}")
                print(f"  🏆 Top Prediction: {sorted_results[0][0].upper()}")
            continue # Skip the LLM generation step

        # ==========================================
        # DETERMINE AR DECODER MODE (Dense vs MAP)
        # ==========================================
        is_map_mode = user_input.lower().startswith('/map')

        if is_map_mode:
            # Strip the '/map ' command from the query
            actual_query = user_input[4:].strip()
            current_image_embeds = map_image_embeds
            print("  [Using Global MAP Token]")
        else:
            actual_query = user_input
            current_image_embeds = dense_image_embeds
            print("  [Using Dense Patches]")

        # Append the new user turn to the conversation history
        history += f"<start_of_turn>user\n{actual_query}<end_of_turn>\n<start_of_turn>model\n"

        # Tokenize the ENTIRE history
        input_ids = tokenizer(history, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            text_embeds = model.llm.get_input_embeddings()(input_ids)

            # The Sandwich: Insert the selected image embeddings right after the BOS token
            bos_embeds = text_embeds[:, :1, :]
            rest_embeds = text_embeds[:, 1:, :]
            inputs_embeds = torch.cat([bos_embeds, current_image_embeds, rest_embeds], dim=1)

            attention_mask = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)

            output_ids = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode the newly generated text
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        print(f"🤖 Model: {prediction}")

        # Append the model's response to the history
        history += f"{prediction}<end_of_turn>\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with your custom Vision-Language Model")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()

    chat_with_image(args.image_path)
