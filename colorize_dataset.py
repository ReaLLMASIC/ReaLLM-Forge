# colorize_dataset.py  (rolling-window mode added)
"""Colourise a dataset split using a trained GPT model.

Modes
-----
* **minmax**  – colour on chosen-token *logits* (after min-max normalisation).
* **softmax** – colour on chosen-token *probabilities*.

Window strategies
-----------------
* **block**   – default.  Chunk the dataset into **non-overlapping** blocks of
  `block_size` tokens.  Fast, but context resets each block.
* **rolling** – slide a *rolling* window: shift by **one token at a time** so
  every prediction gets the longest possible context.  Slower but preserves
  continuity.

Example
-------
```bash
# rolling window, colour 3 000 tokens starting  at offset 50 000
python colorize_dataset.py \
  --out_dir out/my_run \
  --dataset tiny_shakespeare \
  --split train \
  --offset 50000 \
  --num_tokens 3000 \
  --mode softmax \
  --window rolling
```
"""
from __future__ import annotations

import argparse, io, pickle, math
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
import torch, torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from rich.text import Text
import tiktoken  # type: ignore

from model import GPT, GPTConfig

################################################################################
# helpers
################################################################################

def _ansi(renderable) -> str:
    buf = io.StringIO()
    Console(file=buf, force_terminal=True, color_system="truecolor").print(renderable)
    return buf.getvalue()


def _escape_ws(text: str) -> str:
    return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

def _colour(ids: List[int], scalars: List[float], decode: Callable[[Sequence[int]], str], escape_ws: bool = True) -> Text:
    vals = torch.tensor(scalars, dtype=torch.float32)
    norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-6)
    out = Text()
    for tid, v in zip(ids, norm):
        r = int((1 - v.item()) * 255); g = int(v.item() * 255)
        token = decode([tid])
        if escape_ws:
            token = _escape_ws(token)
        out.append(token, style=f"bold #{r:02x}{g:02x}00")
    return out

# byte-fallback helpers -------------------------------------------------------

def _ccwb_encode(text: str, stoi):
    lst: List[int] = []
    for ch in text:
        if ch in stoi:
            lst.append(stoi[ch])
        else:
            lst.extend(stoi[bytes([b])] for b in ch.encode())
    return lst


def _ccwb_decode(ids: List[int], itos):
    out, buf = [], []
    def flush():
        if buf:
            out.append(b"".join(buf).decode("utf-8", "replace")); buf.clear()
    for tok in ids:
        if tok < 256:
            buf.append(itos[tok])
        else:
            flush(); out.append(itos[tok])
    flush(); return "".join(out)

################################################################################
# CLI / loading
################################################################################

def parse_args():
    p = argparse.ArgumentParser("Colourise a dataset split with a trained GPT model")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--ckpt_name", default="ckpt.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--num_tokens", type=int, default=1024)
    p.add_argument("--block_size", type=int)
    p.add_argument("--mode", choices=["minmax", "softmax"], default="minmax")
    p.add_argument("--window", choices=["block", "rolling"], default="block", help="Context window strategy")
    p.add_argument("--offset", type=int, default=0, help="Starting token index within the binary dataset file")
    p.add_argument("--output_file", default="dataset_color.txt")
    p.add_argument("--display", choices=["token", "topk"], default="token", help="Output format")
    p.add_argument("--topk", type=int, default=10, help="Number of top predictions to display when using topk display")
    p.add_argument("--max_token_chars", type=int, default=20, help="Maximum characters for top-k token columns (-1 to disable clipping)")
    p.add_argument("--rank_red", type=int, default=100, help="Rank value treated as fully red in heatmap")
    p.add_argument("--target_style", default="cyan",
                   help="Color to highlight the target token or 'underline' to underline it")
    p.add_argument("--bold_target", action=argparse.BooleanOptionalAction, default=True,
                   help="Bold the target token when highlighting")
    p.add_argument("--escape_whitespace", action=argparse.BooleanOptionalAction, default=True,
                   help="Show newline and tab characters as escape sequences")
    p.add_argument("--plot_metrics", action="store_true", help="Generate Plotly graphs for prediction metrics")
    p.add_argument("--plot_file", default="metrics.html", help="Output HTML file for Plotly metrics")
    p.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma for metrics plotting")
    p.add_argument(
        "--activation_view",
        choices=["target", "rank", "rank_word", "none"],
        default="target",
        help=(
            "Per-layer activation columns: 'target' shows dot-product with target token, "
            "'rank' shows rank of token best aligned with activation, 'rank_word' also "
            "shows that token alongside its rank, and 'none' hides these columns"
        ),
    )
    p.add_argument(
        "--components",
        choices=["wte", "attn", "mlp", "resid"],
        nargs="+",
        default=["wte", "attn", "mlp"],
        help="Activation components to display in activation_view modes",
    )
    return p.parse_args()


def load_tok(meta: Path):
    meta_obj = pickle.load(meta.open("rb"))
    tk = meta_obj.get("tokenizer"); stoi, itos = meta_obj.get("stoi"), meta_obj.get("itos")
    if tk == "tiktoken":
        enc = tiktoken.get_encoding(meta_obj["tiktoken_encoding"])
        return lambda s: enc.encode(s, allowed_special={""}), lambda l: enc.decode(l)
    if tk == "sentencepiece":
        return lambda s: [stoi[c] for c in s], lambda l: "".join(itos[i] for i in l)
    if tk == "custom_char_with_byte_fallback":
        return lambda s: _ccwb_encode(s, stoi), lambda l: _ccwb_decode(l, itos)
    return lambda s: [stoi[c] for c in s], lambda l: "".join(itos[i] for i in l)

################################################################################
# main
################################################################################

def main():
    args = parse_args(); console = Console()

    # --- load model -----------------------------------------------------------------
    ckpt = torch.load(Path(args.out_dir) / args.ckpt_name, map_location=args.device)
    gptconf = GPTConfig(**ckpt["model_args"])
    model = GPT(gptconf)
    sd = ckpt["model"]
    for k in list(sd):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd, strict=False)
    model.to(args.device).eval(); torch.set_grad_enabled(False)
    if args.block_size:
        model.update_block_size(args.block_size)

    # --- tokenizer ------------------------------------------------------------------
    encode, decode = load_tok(Path("data") / args.dataset / "meta.pkl")

    # --- data mm --------------------------------------------------------------------
    dtype = np.uint32 if model.config.vocab_size == 100277 else np.uint16
    data = np.memmap(Path("data") / args.dataset / f"{args.split}.bin", dtype=dtype, mode="r")
    if args.offset >= len(data) - 1:
        raise ValueError("offset beyond dataset length")

    ptd = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=ptd)
        if "cuda" in args.device else torch.no_grad()
    )

    block = args.block_size or model.config.block_size
    pos = args.offset
    tokens_left = min(args.num_tokens, len(data) - 1 - pos)

    lines: List[str] = []

    if args.display == "token":
        ids: List[int] = []
        scalars: List[float] = []
    else:
        table = Table(show_header=True, box=None, pad_edge=False)
        table.add_column("target", no_wrap=True)
        table.add_column("xent", justify="right", no_wrap=True)
        table.add_column("rank", justify="right", no_wrap=True)
        table.add_column("p_tgt", justify="right", no_wrap=True)
        table.add_column("p_left", justify="right", no_wrap=True)
        if args.activation_view != "none":
            if "wte" in args.components:
                table.add_column("t0", justify="right", no_wrap=True)
            for i in range(model.config.n_layer):
                if "attn" in args.components:
                    table.add_column(f"a{i+1}", justify="right", no_wrap=True)
                if "resid" in args.components:
                    table.add_column(f"ar{i+1}", justify="right", no_wrap=True)
                if "mlp" in args.components:
                    table.add_column(f"m{i+1}", justify="right", no_wrap=True)
                if "resid" in args.components:
                    table.add_column(f"mr{i+1}", justify="right", no_wrap=True)
        for i in range(args.topk):
            table.add_column(f"top{i+1}", justify="center", no_wrap=True)

    if args.plot_metrics:
        metrics_rank: List[float] = []
        metrics_p_left: List[float] = []
        metrics_p_tgt: List[float] = []
        metrics_p_top1: List[float] = []
        metrics_ce: List[float] = []
        metrics_focal: List[float] = []

    while tokens_left > 0:
        # Build window
        seq = data[pos : pos + block + 1]
        if len(seq) < 2:
            break  # not enough tokens to predict next

        ctx_tok = torch.from_numpy(seq[:-1].astype(np.int64))[None].to(args.device)

        activations = {"t0": None, "attn": [], "mlp": [], "ar": [], "mr": []}
        handles = []
        if args.display != "token" and args.activation_view != "none":
            def t0_hook(module, inp, out):
                activations["t0"] = out[0, -1, :].detach()

            handles.append(model.transformer.drop.register_forward_hook(t0_hook))

            for blk in model.transformer.h:
                def make_attn_hook(blk):
                    def hook(module, inp, out):
                        outp = out
                        if getattr(blk, "use_peri_ln_attn", False):
                            outp = blk.peri_ln_attn(outp)
                        if getattr(blk, "attn_resid_scaler", None):
                            outp = blk.attn_resid_scaler(outp)
                        activations["attn"].append(outp[0, -1, :].detach())
                    return hook

                def make_mlp_hook(blk):
                    def hook(module, inp, out):
                        outp = out
                        if getattr(blk, "use_peri_ln_mlp", False):
                            outp = blk.peri_ln_mlp(outp)
                        if getattr(blk, "mlp_resid_scaler", None):
                            outp = blk.mlp_resid_scaler(outp)
                        activations["mlp"].append(outp[0, -1, :].detach())
                    return hook

                handles.append(blk.attn.register_forward_hook(make_attn_hook(blk)))
                handles.append(blk.mlp.register_forward_hook(make_mlp_hook(blk)))

        with autocast_ctx:
            logits, _ = model(ctx_tok)

        for h in handles:
            h.remove()

        if args.activation_view != "none" and "resid" in args.components and activations["t0"] is not None:
            resid = activations["t0"].float().clone()
            for a, m in zip(activations["attn"], activations["mlp"]):
                resid = resid + a.float()
                activations["ar"].append(resid.clone())
                resid = resid + m.float()
                activations["mr"].append(resid.clone())

        logits = logits.squeeze(0)  # (ctx_len, vocab)
        ctx_len = logits.size(0)
        tgt_token = int(seq[-1])  # ground-truth next token

        layer_texts: List[Text] = []
        if args.activation_view != "none" and activations["t0"] is not None:
            def iter_vecs():
                if "wte" in args.components:
                    yield activations["t0"]
                for i in range(model.config.n_layer):
                    if "attn" in args.components:
                        yield activations["attn"][i]
                    if "resid" in args.components:
                        yield activations["ar"][i]
                    if "mlp" in args.components:
                        yield activations["mlp"][i]
                    if "resid" in args.components:
                        yield activations["mr"][i]

            if args.activation_view == "target":
                correct_vec = model.lm_head.weight[tgt_token].detach()
                layer_vals = [torch.dot(v.float(), correct_vec.float()).item() for v in iter_vecs()]
                lv = torch.tensor(layer_vals)
                lv_norm = (lv - lv.min()) / (lv.max() - lv.min() + 1e-6)
                for v, n in zip(lv.tolist(), lv_norm.tolist()):
                    r = int((1 - n) * 255); g = int(n * 255)
                    layer_texts.append(Text(f"{v:.2f}", style=f"bold #{r:02x}{g:02x}00"))
            elif args.activation_view == "rank":
                emb = model.lm_head.weight.detach()
                ranks = []
                for vec in iter_vecs():
                    tok = torch.argmax(emb @ vec.float()).item()
                    rnk = int((logits[-1] > logits[-1, tok]).sum().item()) + 1
                    ranks.append(rnk)
                for rnk in ranks:
                    rank_norm = 1 - (min(rnk, args.rank_red) - 1) / max(args.rank_red - 1, 1)
                    r = int((1 - rank_norm) * 255); g = int(rank_norm * 255)
                    layer_texts.append(Text(str(rnk), style=f"bold #{r:02x}{g:02x}00"))
            else:  # args.activation_view == "rank_word"
                emb = model.lm_head.weight.detach()
                toks: List[int] = []
                ranks: List[int] = []
                for vec in iter_vecs():
                    tok = torch.argmax(emb @ vec.float()).item()
                    toks.append(tok)
                    ranks.append(int((logits[-1] > logits[-1, tok]).sum().item()) + 1)
                for tok_id, rnk in zip(toks, ranks):
                    token = decode([tok_id])
                    if args.max_token_chars >= 0:
                        token = token[: args.max_token_chars]
                    if args.escape_whitespace:
                        token = _escape_ws(token)
                    rank_norm = 1 - (min(rnk, args.rank_red) - 1) / max(args.rank_red - 1, 1)
                    r = int((1 - rank_norm) * 255); g = int(rank_norm * 255)
                    layer_texts.append(Text(f"{token}:{rnk}", style=f"bold #{r:02x}{g:02x}00"))

        probs = F.softmax(logits[-1], dim=-1)
        tgt_prob = probs[tgt_token].item()
        tgt_logit = logits[-1, tgt_token].item()
        rank = int((logits[-1] > logits[-1, tgt_token]).sum().item()) + 1
        prob_left = probs[logits[-1] > logits[-1, tgt_token]].sum().item()
        p_top1 = probs.max().item()
        ce = -math.log(tgt_prob + 1e-12)
        focal = ((1 - tgt_prob) ** args.focal_gamma) * ce

        if args.plot_metrics:
            metrics_rank.append(rank)
            metrics_p_left.append(prob_left)
            metrics_p_tgt.append(tgt_prob)
            metrics_p_top1.append(p_top1)
            metrics_ce.append(ce)
            metrics_focal.append(focal)

        if args.display == "token":
            scalar_val = tgt_prob if args.mode == "softmax" else tgt_logit
            ids.append(tgt_token)
            scalars.append(scalar_val)
        else:
            topv, topi = logits[-1].topk(args.topk)
            norm = (topv - topv.min()) / (topv.max() - topv.min() + 1e-6)
            words: List[Text] = []
            for idx, v in zip(topi.tolist(), norm.tolist()):
                r = int((1 - v) * 255); g = int(v * 255)
                style = f"#{r:02x}{g:02x}00"
                if idx == tgt_token:
                    if args.target_style == "underline":
                        if args.bold_target:
                            style += " bold"
                        style += " underline"
                    else:
                        style = args.target_style
                        if args.bold_target:
                            style = f"bold {style}"
                token = decode([idx])
                if args.max_token_chars >= 0:
                    token = token[: args.max_token_chars]
                if args.escape_whitespace:
                    token = _escape_ws(token)
                words.append(Text(token, style=style))

            rank_norm = 1 - (min(rank, args.rank_red) - 1) / max(args.rank_red - 1, 1)
            r = int((1 - rank_norm) * 255); g = int(rank_norm * 255)
            rank_text = Text(str(rank), style=f"bold #{r:02x}{g:02x}00")

            v = tgt_prob
            r = int((1 - v) * 255); g = int(v * 255)
            p_tgt_text = Text(f"{tgt_prob:.4f}", style=f"bold #{r:02x}{g:02x}00")

            v = 1 - prob_left
            r = int((1 - v) * 255); g = int(v * 255)
            p_left_text = Text(f"{prob_left:.4f}", style=f"bold #{r:02x}{g:02x}00")

            target_word = decode([tgt_token])
            if args.max_token_chars >= 0:
                target_word = target_word[: args.max_token_chars]
            if args.escape_whitespace:
                target_word = _escape_ws(target_word)
            if args.target_style == "underline":
                target_style = "underline"
            else:
                target_style = args.target_style
            if args.bold_target:
                target_style = f"bold {target_style}" if target_style else "bold"

            row = [Text(target_word, style=target_style), f"{ce:.4f}", rank_text, p_tgt_text, p_left_text] + layer_texts + words
            table.add_row(*row)

        # advance
        step = 1 if args.window == "rolling" else ctx_len
        pos += step
        tokens_left -= 1 if args.window == "rolling" else min(ctx_len, tokens_left)

    if args.display == "token":
        coloured = _colour(ids, scalars, decode, escape_ws=args.escape_whitespace)
        console.print(coloured)
        lines.append(_ansi(coloured))
    else:
        console.print(table)
        lines.append(_ansi(table))

    if args.output_file:
        Path(args.output_file).write_text("".join(lines), "utf-8", errors="replace")
        console.print(f"[cyan]Saved → {args.output_file}[/cyan]")

    if args.plot_metrics:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        x = list(range(len(metrics_rank)))
        fig = make_subplots(rows=6, cols=1, shared_xaxes=True,
                            subplot_titles=["target rank", "prob left", "p(target)",
                                            "p(top1)", "cross entropy", "focal loss"])
        fig.add_trace(go.Scatter(x=x, y=metrics_rank, name="rank"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=metrics_p_left, name="p_left"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=metrics_p_tgt, name="p_tgt"), row=3, col=1)
        fig.add_trace(go.Scatter(x=x, y=metrics_p_top1, name="p_top1"), row=4, col=1)
        fig.add_trace(go.Scatter(x=x, y=metrics_ce, name="cross_entropy"), row=5, col=1)
        fig.add_trace(go.Scatter(x=x, y=metrics_focal, name="focal"), row=6, col=1)
        fig.update_layout(height=300 * 6, showlegend=False)
        fig.write_html(args.plot_file)
        console.print(f"[cyan]Saved plot → {args.plot_file}[/cyan]")


if __name__ == "__main__":
    main()
