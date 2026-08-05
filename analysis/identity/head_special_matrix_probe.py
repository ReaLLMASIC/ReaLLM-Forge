#!/usr/bin/env python3
"""
head_special_matrix_probe.py

Only generates per-layer/per-head heatmaps for whether W_O^h W_V^h
resembles identity, inverse/rotation-like, projection-like, etc. It also tests
whether the equivalent head-space value/output map W_V^h W_O^h resembles
specified block-diagonal 2D rotations.

Additionally, we extract the product matrix W_Q^h (W_K^h)^T, partition it into
its symmetric and skew-symmetric components to measure their L2 mass, and
precisely tabulate the spectrum of the symmetric part (positive, negative, and
zero eigenvalues). To bypass O(d_model^3) complexity, this utilizes an exact
O(d_head^3) low-rank algebraic decomposition!

Generates per-metric PNG heatmaps, interactive Plotly HTML pages, and an overall
HTML dashboard report connecting everything.

Examples:

  python head_special_matrix_probe.py --preset qwen-0.5b --device cuda

  python head_special_matrix_probe.py --preset smollm2-135m-instruct --device cuda --dtype bfloat16
"""

import argparse
import os
import csv

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly is not installed. Plotly interactive webpages will not be generated.")

try:
    from model import GPT
    from gpt_conf import GPTConfig
except ImportError:
    pass


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument(
        "--preset",
        default=None,
        choices=[
            "qwen-0.5b",
            "gemma-3-270m",
            "gemma-3-270m-it",
            "smollm2-135m-instruct",
        ],
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--outdir", default="./head_special_matrix_out")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--ckpt", default=None, help="Optional local nanoGPT checkpoint path to probe instead of a HuggingFace model.")
    p.add_argument("--max-layers", type=int, default=None)
    p.add_argument("--show", action="store_true")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--rotation-degrees",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Explicit head-space rotation angles in degrees to test. "
            "Overrides --rotation-min-deg/--rotation-max-deg/--rotation-step-deg."
        ),
    )
    p.add_argument("--rotation-min-deg", type=float, default=10.0)
    p.add_argument("--rotation-max-deg", type=float, default=35.0)
    p.add_argument("--rotation-step-deg", type=float, default=5.0)

    return p.parse_args()


def apply_preset(args):
    if args.preset == "qwen-0.5b":
        args.model = "Qwen/Qwen2.5-0.5B"

    elif args.preset == "gemma-3-270m":
        args.model = "google/gemma-3-270m"
        args.trust_remote_code = True

    elif args.preset == "gemma-3-270m-it":
        args.model = "google/gemma-3-270m-it"
        args.trust_remote_code = True

    elif args.preset == "smollm2-135m-instruct":
        args.model = "HuggingFaceTB/SmolLM2-135M-Instruct"

    return args


def get_dtype(name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def format_degree_label(deg):
    return f"{deg:g}".replace("-", "neg_").replace(".", "p")


def get_rotation_degrees(args):
    if args.rotation_degrees is not None:
        degrees = args.rotation_degrees
    else:
        if args.rotation_step_deg <= 0:
            raise ValueError("--rotation-step-deg must be positive.")
        if args.rotation_min_deg > args.rotation_max_deg:
            raise ValueError("--rotation-min-deg must be <= --rotation-max-deg.")

        degrees = []
        current = args.rotation_min_deg
        # Include the max endpoint despite small floating-point accumulation error.
        while current <= args.rotation_max_deg + (args.rotation_step_deg * 1e-9):
            degrees.append(current)
            current += args.rotation_step_deg

    # Preserve order while removing duplicates from equivalent display labels.
    seen = set()
    unique = []
    for degree in degrees:
        label = format_degree_label(degree)
        if label not in seen:
            seen.add(label)
            unique.append(float(degree))
    return unique


def rotation_matrix_2d_blocks(dim, degrees, device):
    """Return a block-diagonal 2D rotation target for head-space matrices."""
    theta = torch.tensor(np.deg2rad(degrees), dtype=torch.float32, device=device)
    c = torch.cos(theta)
    s = torch.sin(theta)
    R = torch.eye(dim, dtype=torch.float32, device=device)
    for i in range(0, dim - 1, 2):
        R[i, i] = c
        R[i, i + 1] = -s
        R[i + 1, i] = s
        R[i + 1, i + 1] = c
    return R


def save_heatmap(arr, title, xlabel, ylabel, cbar, path, dpi=200, show=False):
    plt.figure(figsize=(14, 8))
    plt.imshow(arr, aspect="auto")
    plt.colorbar(label=cbar)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if arr.shape[1] <= 64:
        plt.xticks(range(arr.shape[1]))
    if arr.shape[0] <= 80:
        plt.yticks(range(arr.shape[0]))

    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def generate_html_report(args, metric_names, descriptions):
    """Generates a cohesive HTML dashboard linking to Plotly and PNG heatmaps."""
    report_path = os.path.join(args.outdir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("<html>\n<head>\n<title>Probe Report</title>\n")
        f.write("<style>\n")
        f.write("body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 30px; background: #f8f9fa; color: #212529; }\n")
        f.write("h1 { text-align: center; color: #343a40; margin-bottom: 5px; }\n")
        f.write("h2 { border-bottom: 2px solid #dee2e6; padding-bottom: 5px; margin-top: 40px; color: #495057; }\n")
        f.write(".grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; }\n")
        f.write(".card { border: 1px solid #e9ecef; padding: 20px; border-radius: 10px; background: #ffffff; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }\n")
        f.write(".card h3 { margin-top: 0; font-size: 1.2em; color: #0056b3; word-wrap: break-word; }\n")
        f.write("a { text-decoration: none; color: #007bff; font-weight: bold; }\n")
        f.write("a:hover { text-decoration: underline; color: #0056b3; }\n")
        f.write(".img-preview { max-width: 100%; height: auto; display: block; margin-top: 15px; border: 1px solid #dee2e6; border-radius: 6px; }\n")
        f.write(".model-subtitle { text-align: center; color: #6c757d; font-size: 1.1em; margin-bottom: 40px; }\n")
        f.write("</style>\n</head>\n<body>\n")
        f.write(f"<h1>Attention Head Matrix Probe Report</h1>\n")
        f.write(f"<p class='model-subtitle'><strong>Model:</strong> {args.model}</p>\n")

        categories = {
            "QK Matrix Inner Product Structure Metrics": [m for m in metric_names if m.startswith("QK_")],
            "OV Identity & Standard Properties": [m for m in metric_names if "identity" in m or "orthogonal" in m or "involution" in m or "projection" in m or "symmetric" in m or "skew" in m],
            "Head-space VO Metrics": [m for m in metric_names if "headspace" in m],
        }

        for cat, metrics in categories.items():
            if not metrics:
                continue
            f.write(f"<h2>{cat}</h2>\n")
            f.write("<div class='grid'>\n")
            for m in metrics:
                desc = descriptions.get(m, "")
                f.write(f"<div class='card'>\n")
                f.write(f"<h3>{m}</h3>\n")
                f.write(f"<p>{desc}</p>\n")
                if PLOTLY_AVAILABLE:
                    f.write(f"<p><a href='plotly/{m}.html' target='_blank'>&#128200; View Interactive Plotly Heatmap</a></p>\n")
                f.write(f"<a href='{m}_heatmap.png' target='_blank'><img class='img-preview' src='{m}_heatmap.png' alt='{m} heatmap'></a>\n")
                f.write(f"</div>\n")
            f.write("</div>\n")

        f.write("</body>\n</html>\n")


def cosine_to(A, B):
    return F.cosine_similarity(A.flatten(), B.flatten(), dim=0).item()


def rel_frob(A, B, eps=1e-12):
    return (torch.linalg.norm(A - B) / (torch.linalg.norm(B) + eps)).item()


def frob(A, B):
    return torch.linalg.norm(A - B).item()


def get_hf_decoder_layers(model):
    layer_roots = [
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        getattr(getattr(model, "transformer", None), "h", None),
    ]
    for root in layer_roots:
        if root is None:
            continue
        if hasattr(root, "layers"):
            return root.layers
        if hasattr(root, "h"):
            return root.h
        if isinstance(root, (list, torch.nn.ModuleList)):
            return root
    raise ValueError("Could not find decoder layers on the HuggingFace model.")


def get_hf_config_value(config, name, default=None):
    if hasattr(config, name):
        return getattr(config, name)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, name):
        return getattr(text_config, name)
    return default


def load_hf_model(args, dtype):
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    model.eval()

    cfg = model.config
    hidden = get_hf_config_value(cfg, "hidden_size")
    config_heads = get_hf_config_value(cfg, "num_attention_heads")
    config_kv_heads = get_hf_config_value(cfg, "num_key_value_heads")
    layers = get_hf_decoder_layers(model)
    return model, layers, hidden, config_heads, config_kv_heads, "hf"


def load_nanogpt_checkpoint(args):
    print(f"Loading nanoGPT checkpoint: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=args.device)
    model_args = checkpoint["model_args"]
    config = GPTConfig(**model_args)
    model = GPT(config)

    state_dict = checkpoint["model"]
    for key in list(state_dict.keys()):
        if key.startswith("_orig_mod."):
            state_dict[key[len("_orig_mod."):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    hidden = config.n_embd
    config_heads = config.n_head
    config_kv_heads = config.n_kv_group if config.n_kv_group is not None else config.n_head
    layers = model.transformer.h
    return model, layers, hidden, config_heads, config_kv_heads, "nanogpt"


def get_attention_weights(block, model_kind):
    if model_kind == "hf":
        attn = block.self_attn
        return (
            attn.q_proj.weight.detach().float(),
            attn.k_proj.weight.detach().float(),
            attn.v_proj.weight.detach().float(),
            attn.o_proj.weight.detach().float(),
        )

    attn = block.attn
    if not all(hasattr(attn, name) for name in ("c_attn_q", "c_attn_k", "c_attn_v", "c_proj")):
        raise ValueError(
            "Local checkpoint attention module must expose c_attn_q, c_attn_k, "
            "c_attn_v, and c_proj weights for this probe."
        )
    if hasattr(attn, "c_proj_list"):
        raise ValueError("c_proj_list attention variants are not supported by this probe yet.")

    return (
        attn.c_attn_q.weight.detach().float(),
        attn.c_attn_k.weight.detach().float(),
        attn.c_attn_v.weight.detach().float(),
        attn.c_proj.weight.detach().float(),
    )


def infer_head_shapes(Wq, Wk, Wv, Wo, hidden, config_heads=None):
    if config_heads is not None and Wq.shape[0] % config_heads == 0:
        head_dim = Wq.shape[0] // config_heads
    else:
        raise ValueError("Could not infer head_dim from config_heads.")

    q_heads = Wq.shape[0] // head_dim
    k_heads = Wk.shape[0] // head_dim
    v_heads = Wv.shape[0] // head_dim
    o_heads = Wo.shape[1] // head_dim

    return head_dim, q_heads, k_heads, v_heads, o_heads


def main():
    args = apply_preset(parse_args())
    os.makedirs(args.outdir, exist_ok=True)

    plotly_dir = os.path.join(args.outdir, "plotly")
    if PLOTLY_AVAILABLE:
        os.makedirs(plotly_dir, exist_ok=True)

    dtype = get_dtype(args.dtype)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        args.device = "cpu"

    rotation_degrees = get_rotation_degrees(args)
    print("rotation degree targets:", rotation_degrees)

    if args.ckpt:
        model, layers, hidden, config_heads, config_kv_heads, model_kind = load_nanogpt_checkpoint(args)
    else:
        model, layers, hidden, config_heads, config_kv_heads, model_kind = load_hf_model(args, dtype)

    print("hidden:", hidden)
    print("config num_attention_heads:", config_heads)
    print("config num_key_value_heads:", config_kv_heads)

    if args.max_layers is not None:
        layers = layers[:args.max_layers]

    metric_names = [
        "cos_identity_OV",
        "rel_frob_identity_OV",
        "frob_identity_OV",

        "cos_negative_identity_OV",
        "rel_frob_negative_identity_OV",

        "rel_frob_orthogonal_OVtOV_I",
        "cos_orthogonal_OVtOV_I",

        "rel_frob_involution_OV2_I",
        "cos_involution_OV2_I",

        "rel_frob_projection_OV2_OV",
        "cos_projection_OV2_OV",

        "rel_frob_symmetric_OV_OVt",
        "cos_symmetric_OV_OVt",

        "rel_frob_skew_OV_negOVt",
        "cos_skew_OV_negOVt",

        "rel_frob_headspace_VO_I",
        "cos_headspace_VO_I",

        "QK_norm_frob",
        "QK_norm_sym",
        "QK_norm_skew",
        "QK_mass_sym",
        "QK_mass_skew",
        "QK_pos_eigs",
        "QK_neg_eigs",
        "QK_zero_eigs",
    ]

    rotation_metric_names = []
    rotation_metric_degrees = []
    for degree in rotation_degrees:
        label = format_degree_label(degree)
        rotation_metric_names.extend([
            f"rel_frob_headspace_VO_rotation_{label}deg",
            f"cos_headspace_VO_rotation_{label}deg",
        ])
        rotation_metric_degrees.append((degree, label))

    metric_names.extend(rotation_metric_names)

    scores = {m: [] for m in metric_names}
    rows = []

    with torch.no_grad():
        for layer_idx, block in enumerate(layers):
            Wq, Wk, Wv, Wo = get_attention_weights(block, model_kind)

            head_dim, q_heads, k_heads, v_heads, o_heads = infer_head_shapes(
                Wq, Wk, Wv, Wo, hidden, config_heads
            )

            # View weights to separate multiple heads
            Wq_h = Wq.view(q_heads, head_dim, hidden)
            Wk_h = Wk.view(k_heads, head_dim, hidden)
            Wv_h = Wv.view(v_heads, head_dim, hidden)
            Wo_h = Wo.view(hidden, o_heads, head_dim).permute(1, 0, 2)

            usable_heads = min(q_heads, o_heads)
            kv_group_size = max(1, q_heads // max(1, v_heads))
            k_group_size = max(1, q_heads // max(1, k_heads))

            if layer_idx == 0:
                print("inferred head_dim:", head_dim)
                print("inferred q_heads:", q_heads)
                print("inferred k_heads:", k_heads)
                print("inferred v_heads:", v_heads)
                print("inferred o_heads:", o_heads)
                print("usable_heads:", usable_heads)
                print("kv_group_size:", kv_group_size)

            print(f"Layer {layer_idx}")

            I_hidden = torch.eye(hidden, dtype=torch.float32, device=Wv.device)
            neg_I_hidden = -I_hidden
            I_head = torch.eye(head_dim, dtype=torch.float32, device=Wv.device)
            rotation_targets = {
                label: rotation_matrix_2d_blocks(head_dim, degree, Wv.device)
                for degree, label in rotation_metric_degrees
            }

            layer_metric_scores = {m: [] for m in metric_names}

            for h in range(usable_heads):
                kv_h = min(h // kv_group_size, v_heads - 1)
                k_h = min(h // k_group_size, k_heads - 1)

                values = {}

                # =================================================================
                # QK Bilinear Space (W_Q W_K^T) Extraction & Low-Rank Optimization
                # =================================================================
                # The full d_model x d_model attention matrix is extremely slow to
                # eigendecompose. Using trace permutations, norms and eigenvalues
                # are perfectly reconstructed in O(d_head^3) time instead of O(d_model^3).
                wq_f = Wq_h[h].float()
                wk_f = Wk_h[k_h].float()

                Pq = wq_f @ wq_f.T # [head_dim, head_dim]
                Pk = wk_f @ wk_f.T # [head_dim, head_dim]
                C = wk_f @ wq_f.T  # [head_dim, head_dim]

                # ||C||_F^2 = Tr(C^T C) = Tr(P_q P_k)
                norm_QK_sq = torch.sum(Pq * Pk).item()
                tr_C2 = torch.sum(C * C.T).item()

                # S = 0.5 * (C + C.T), A = 0.5 * (C - C.T)
                norm_S_sq = 0.5 * (norm_QK_sq + tr_C2)
                norm_A_sq = 0.5 * (norm_QK_sq - tr_C2)

                values["QK_norm_frob"] = max(norm_QK_sq, 0.0) ** 0.5
                values["QK_norm_sym"] = max(norm_S_sq, 0.0) ** 0.5
                values["QK_norm_skew"] = max(norm_A_sq, 0.0) ** 0.5

                if norm_QK_sq > 0:
                    values["QK_mass_sym"] = max(norm_S_sq, 0.0) / norm_QK_sq
                    values["QK_mass_skew"] = max(norm_A_sq, 0.0) / norm_QK_sq
                else:
                    values["QK_mass_sym"] = 0.0
                    values["QK_mass_skew"] = 0.0

                # S = U V^T. The non-zero eigenvalues of S exactly match V^T U
                # which collapses down to a tiny 2*head_dim x 2*head_dim matrix.
                top_m = torch.cat([C, Pk], dim=1)
                bot_m = torch.cat([Pq, C.T], dim=1)
                M = 0.5 * torch.cat([top_m, bot_m], dim=0)

                eigs = torch.linalg.eigvals(M).real
                eigs_max = eigs.abs().max().item() if eigs.numel() > 0 else 0.0
                tol = 1e-5 * max(eigs_max, 1e-7)

                pos_eigs = (eigs > tol).sum().item()
                neg_eigs = (eigs < -tol).sum().item()
                # Missing eigenvalues (up to d_model) belong dynamically to the null space
                zero_eigs = hidden - pos_eigs - neg_eigs

                values["QK_pos_eigs"] = pos_eigs
                values["QK_neg_eigs"] = neg_eigs
                values["QK_zero_eigs"] = zero_eigs


                # =================================================================
                # OV Identity / Special Matrix Operators
                # =================================================================
                # Hidden-space operator:
                # OV: [hidden, hidden]
                OV = Wo_h[h] @ Wv_h[kv_h]

                # Head-space operator:
                # VO: [head_dim, head_dim]
                # If this is identity-like, then Wv and Wo are inverse-like
                # inside the small head subspace.
                VO = Wv_h[kv_h] @ Wo_h[h]

                OVt = OV.T
                OV2 = OV @ OV
                OVtOV = OVt @ OV

                values["cos_identity_OV"] = cosine_to(OV, I_hidden)
                values["rel_frob_identity_OV"] = rel_frob(OV, I_hidden)
                values["frob_identity_OV"] = frob(OV, I_hidden)

                values["cos_negative_identity_OV"] = cosine_to(OV, neg_I_hidden)
                values["rel_frob_negative_identity_OV"] = rel_frob(OV, neg_I_hidden)

                values["rel_frob_orthogonal_OVtOV_I"] = rel_frob(OVtOV, I_hidden)
                values["cos_orthogonal_OVtOV_I"] = cosine_to(OVtOV, I_hidden)

                values["rel_frob_involution_OV2_I"] = rel_frob(OV2, I_hidden)
                values["cos_involution_OV2_I"] = cosine_to(OV2, I_hidden)

                values["rel_frob_projection_OV2_OV"] = rel_frob(OV2, OV)
                values["cos_projection_OV2_OV"] = cosine_to(OV2, OV)

                values["rel_frob_symmetric_OV_OVt"] = rel_frob(OV, OVt)
                values["cos_symmetric_OV_OVt"] = cosine_to(OV, OVt)

                values["rel_frob_skew_OV_negOVt"] = rel_frob(OV, -OVt)
                values["cos_skew_OV_negOVt"] = cosine_to(OV, -OVt)

                values["rel_frob_headspace_VO_I"] = rel_frob(VO, I_head)
                values["cos_headspace_VO_I"] = cosine_to(VO, I_head)

                for degree, label in rotation_metric_degrees:
                    R_head = rotation_targets[label]
                    values[f"rel_frob_headspace_VO_rotation_{label}deg"] = rel_frob(VO, R_head)
                    values[f"cos_headspace_VO_rotation_{label}deg"] = cosine_to(VO, R_head)

                for m in metric_names:
                    layer_metric_scores[m].append(values[m])

                row = {
                    "layer": layer_idx,
                    "head": h,
                    "kv_head": kv_h,
                    "head_dim": head_dim,
                    "q_heads": q_heads,
                    "k_heads": k_heads,
                    "v_heads": v_heads,
                    "o_heads": o_heads,
                    "usable_heads": usable_heads,
                }
                row.update(values)
                rows.append(row)

            for m in metric_names:
                scores[m].append(layer_metric_scores[m])

    csv_path = os.path.join(args.outdir, "head_special_matrix_scores.csv")
    fieldnames = [
        "layer",
        "head",
        "kv_head",
        "head_dim",
        "q_heads",
        "k_heads",
        "v_heads",
        "o_heads",
        "usable_heads",
    ] + metric_names

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Saved CSV:", csv_path)

    descriptions = {
        "cos_identity_OV": "OV cosine to identity, higher = more identity-like",
        "rel_frob_identity_OV": "OV relative Frobenius distance to identity, lower = more identity-like",
        "frob_identity_OV": "OV Frobenius distance to identity, lower = more identity-like",

        "cos_negative_identity_OV": "OV cosine to negative identity, higher = more -I-like",
        "rel_frob_negative_identity_OV": "OV relative Frobenius distance to -I, lower = more -I-like",

        "rel_frob_orthogonal_OVtOV_I": "OVᵀOV relative distance to I, lower = more orthogonal/rotation-like",
        "cos_orthogonal_OVtOV_I": "OVᵀOV cosine to I, higher = more orthogonal/rotation-like",

        "rel_frob_involution_OV2_I": "OV² relative distance to I, lower = more inverse/involution-like",
        "cos_involution_OV2_I": "OV² cosine to I, higher = more inverse/involution-like",

        "rel_frob_projection_OV2_OV": "OV² relative distance to OV, lower = more projection-like",
        "cos_projection_OV2_OV": "OV² cosine to OV, higher = more projection-like",

        "rel_frob_symmetric_OV_OVt": "OV relative distance to OVᵀ, lower = more symmetric",
        "cos_symmetric_OV_OVt": "OV cosine to OVᵀ, higher = more symmetric",

        "rel_frob_skew_OV_negOVt": "OV relative distance to -OVᵀ, lower = more skew-symmetric",
        "cos_skew_OV_negOVt": "OV cosine to -OVᵀ, higher = more skew-symmetric",

        "rel_frob_headspace_VO_I": "V_hO_h relative distance to head identity, lower = more inverse-like in head space",
        "cos_headspace_VO_I": "V_hO_h cosine to head identity, higher = more inverse-like in head space",

        "QK_norm_frob": "L2 (Frobenius) norm of the QK operator W_Q^h (W_K^h)^T",
        "QK_norm_sym": "L2 norm of the purely symmetric part of the QK operator",
        "QK_norm_skew": "L2 norm of the purely skew-symmetric part of the QK operator",
        "QK_mass_sym": "Fraction of the QK matrix's mass strictly in the symmetric part (squared L2)",
        "QK_mass_skew": "Fraction of the QK matrix's mass strictly in the skew-symmetric part (squared L2)",
        "QK_pos_eigs": "Number of positive eigenvalues of the symmetric QK part",
        "QK_neg_eigs": "Number of negative eigenvalues of the symmetric QK part",
        "QK_zero_eigs": "Number of zero eigenvalues of the symmetric QK part",
    }

    for degree, label in rotation_metric_degrees:
        descriptions[f"rel_frob_headspace_VO_rotation_{label}deg"] = (
            f"V_hO_h relative distance to a {degree:g}° block-diagonal head-space rotation, "
            "lower = more like that equivalent Wv/Wo rotation"
        )
        descriptions[f"cos_headspace_VO_rotation_{label}deg"] = (
            f"V_hO_h cosine to a {degree:g}° block-diagonal head-space rotation, "
            "higher = more like that equivalent Wv/Wo rotation"
        )

    for m in metric_names:
        arr = np.array(scores[m], dtype=float)

        path = os.path.join(args.outdir, f"{m}_heatmap.png")

        save_heatmap(
            arr,
            m,
            "Head",
            "Layer",
            descriptions[m],
            path,
            dpi=args.dpi,
            show=args.show,
        )
        print("Saved static heatmap:", path)

        if PLOTLY_AVAILABLE:
            path_html = os.path.join(plotly_dir, f"{m}.html")
            fig = px.imshow(
                arr,
                labels=dict(x="Head", y="Layer", color=m),
                title=f"{m}<br><sup>{descriptions[m]}</sup>",
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(title_x=0.5)
            # Ensure correct and aesthetic integer tick orientations
            fig.update_xaxes(side="bottom")
            if arr.shape[1] <= 64:
                fig.update_xaxes(tickmode='linear', tick0=0, dtick=1)
            if arr.shape[0] <= 80:
                fig.update_yaxes(tickmode='linear', tick0=0, dtick=1)

            fig.write_html(path_html, include_plotlyjs="cdn")

    generate_html_report(args, metric_names, descriptions)
    print(f"Generated comprehensive HTML dashboard: {os.path.join(args.outdir, 'report.html')}")

    print("Done.")


if __name__ == "__main__":
    main()

