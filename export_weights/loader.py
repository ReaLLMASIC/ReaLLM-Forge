import os
import torch
from model import GPT

def load_checkpoint(out_dir, device="cuda"):
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict

def load_model(model_name="gpt2"):
    return GPT.from_pretrained(model_name)
