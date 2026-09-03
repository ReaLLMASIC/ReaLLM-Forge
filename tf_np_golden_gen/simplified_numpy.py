import numpy as np
import pickle
import math
import argparse
import json
import os
from datetime import datetime
from collections import OrderedDict

class GPTConfig:
    def __init__(self, vocab_size=65, block_size=256, n_layer=6, n_head=6, n_embd=384, n_kv_group=6):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_kv_group = n_kv_group

def gelu(x):
    """Gaussian Error Linear Unit."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

class RMSNorm:
    def __init__(self, dim, eps=0, gain=None):
        self.eps = eps
        self.dim = dim
        self.weight = gain if gain is not None else np.ones((dim,))

    def __call__(self, x):
        mean_square = np.mean(x ** 2, axis=-1, keepdims=True)
        normed = x / np.sqrt(mean_square + self.eps)
        return normed * self.weight

class Linear:
    def __init__(self, in_features, out_features, weights):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = weights  # Shape: (out_features, in_features)

    def __call__(self, x):
        return np.dot(x, self.weight.T)

class CausalSelfAttention:
    def __init__(self, config, weights):
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.scale = 1.0 / math.sqrt(config.n_embd // config.n_head)
        self.n_kv_group = config.n_kv_group

        # Load weights for q, k, v projections and output projection
        self.c_attn_q = Linear(config.n_embd, config.n_embd, weights['attn.c_attn_q.weight'])
        self.c_attn_k = Linear(config.n_embd, config.n_kv_group * (config.n_embd // config.n_head), weights['attn.c_attn_k.weight'])
        self.c_attn_v = Linear(config.n_embd, config.n_kv_group * (config.n_embd // config.n_head), weights['attn.c_attn_v.weight'])
        self.c_proj = Linear(config.n_embd, config.n_embd, weights['attn.c_proj.weight'])

    def __call__(self, x):
        B, T, C = x.shape

        q = self.c_attn_q(x).reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        k = self.c_attn_k(x).reshape(B, T, self.n_kv_group, C // self.n_head).transpose(0, 2, 1, 3)
        v = self.c_attn_v(x).reshape(B, T, self.n_kv_group, C // self.n_head).transpose(0, 2, 1, 3)

        att = np.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        att = np.tril(np.ones((T, T)))  # Causal mask
        att = att / np.sum(att, axis=-1, keepdims=True)

        y = np.matmul(att, v).transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.c_proj(y)
        return y

class MLP:
    def __init__(self, config, weights):
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, weights['mlp.c_fc.weight'])
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, weights['mlp.c_proj.weight'])

    def __call__(self, x):
        # x = np.maximum(0, self.c_fc(x))  # ReLU activation
        x = gelu(self.c_fc(x))
        x = self.c_proj(x)
        return x

class Block:
    def __init__(self, config, weights):
        # Load ln_1 and ln_2 gains from weights
        ln_1_gain = weights.get('ln_1.gain', np.ones(config.n_embd))
        ln_2_gain = weights.get('ln_2.gain', np.ones(config.n_embd))

        self.ln_1 = RMSNorm(config.n_embd, gain=ln_1_gain)
        self.attn = CausalSelfAttention(config, weights)
        self.ln_2 = RMSNorm(config.n_embd, gain=ln_2_gain)
        self.mlp = MLP(config, weights)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT:
    def __init__(self, config, weights, stoi, itos):
        self.config = config
        self.stoi = stoi  # String to Index mapping
        self.itos = itos  # Index to String mapping

        # Initialize weights correctly based on loaded keys
        self.wte = weights['transformer.wte.weight']  # Embedding weights
        self.blocks = [Block(config, {k.split(f'transformer.h.{i}.')[-1]: weights[k] for k in weights if f'transformer.h.{i}.' in k}) for i in range(config.n_layer)]
        self.ln_f = RMSNorm(config.n_embd, gain=weights.get('transformer.ln_f.gain', np.ones(config.n_embd)))
        self.lm_head = weights['lm_head.weight']

    def __call__(self, idx, max_new_tokens, temperature=1.0, top_k=None, repetition_penalty=1.1):
        B, T = idx.shape
        x = self.wte[idx]  # Token embedding lookup
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = np.dot(x, self.lm_head.T)  # Final linear projection

        for _ in range(max_new_tokens):
            # Apply temperature
            logits[:, -1, :] /= temperature

            # For numerical stability, subtract the max logit from logits
            max_logits = np.max(logits[:, -1, :], axis=-1, keepdims=True)
            logits[:, -1, :] -= max_logits

            # Apply softmax
            exp_logits = np.exp(logits[:, -1, :])
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            # Apply a penalty to logits of previously generated tokens
            for token in np.unique(idx):
                probs[:, token] /= repetition_penalty

            # Apply top-k sampling
            if top_k is not None and top_k < logits.shape[-1]:
                top_k_indices = np.argpartition(logits[:, -1, :], -top_k)[:, -top_k:]
                top_k_logits = logits[:, -1, top_k_indices[0]]
                top_k_probs = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits), axis=-1)
                idx_next = np.random.choice(top_k_indices[0], p=top_k_probs.flatten())
            else:
                idx_next = np.random.choice(np.arange(probs.shape[-1]), p=probs.flatten())

            idx = np.concatenate([idx, np.array([[idx_next]])], axis=1)

        return idx

    def decode(self, indices):
        return ''.join([self.itos[i] for i in indices.flatten()])


def load_weights(weights_path):
    with open(weights_path, 'rb') as f:
        weights = pickle.load(f)
    return weights

def load_meta(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return meta['stoi'], meta['itos']

def save_args(args, out_dir):
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description="Numpy-based GPT Inference")
    parser.add_argument("--out_dir", type=str, default="out", help="Directory to save output")
    parser.add_argument("--weights_path", type=str, default="weights.pkl", help="Path to the weights file")
    parser.add_argument("--meta_path", type=str, default="meta.pkl", help="Path to the meta file")
    parser.add_argument("--start", type=str, default="Hello world", help="Start text for generation")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.2, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--sample_file", type=str, default=None, help="File to save generated samples")

    return parser.parse_args()

def main():
    args = parse_args()

    np.random.seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    save_args(args, out_dir)

    weights = load_weights(args.weights_path)
    stoi, itos = load_meta(args.meta_path)

    config = GPTConfig()
    model = GPT(config, weights, stoi, itos)

    # Handle start token
    start_ids = np.array([[stoi[char] for char in args.start]])

    for _ in range(args.num_samples):
        result = model(start_ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        decoded_result = model.decode(result)
        print("Generated text:", decoded_result)

        if args.sample_file:
            with open(args.sample_file, "a") as f:
                f.write(decoded_result + "\n")

if __name__ == "__main__":
    main()

