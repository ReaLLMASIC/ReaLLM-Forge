"""Streamlit webapp for exploring Gemma 3 270M LM-head token angles.

Features:
- Pairwise angle + magnitudes between any two vocab tokens.
- Case-insensitive token search with infix matching suggestions.
- Neighborhood mode: choose one token and list all vocab tokens sorted by angle.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class TokenInfo:
    token_id: int
    raw: str
    display: str
    normalized: str


def _norm(text: str) -> str:
    return text.casefold().strip()


def _build_token_infos(tokenizer: AutoTokenizer) -> list[TokenInfo]:
    vocab = tokenizer.get_vocab()
    infos: list[TokenInfo] = []
    for tok, idx in vocab.items():
        cleaned = tok.replace("▁", " ")
        display = cleaned.encode("utf-8", "replace").decode("utf-8")
        infos.append(
            TokenInfo(
                token_id=idx,
                raw=tok,
                display=display,
                normalized=_norm(tok) + " " + _norm(display),
            )
        )
    infos.sort(key=lambda x: x.token_id)
    return infos


@lru_cache(maxsize=4)
def load_model_assets(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    model.to(device)
    model.eval()

    weight = model.lm_head.weight.detach().to(device=device, dtype=torch.float32)
    norms = torch.linalg.norm(weight, dim=1)
    token_infos = _build_token_infos(tokenizer)

    return tokenizer, token_infos, weight, norms


def search_tokens(infos: list[TokenInfo], query: str, max_results: int) -> list[TokenInfo]:
    q = _norm(query)
    if not q:
        return infos[:max_results]
    matches = [info for info in infos if q in info.normalized]
    return matches[:max_results]


def label(info: TokenInfo) -> str:
    return f"{info.token_id:6d} | {repr(info.raw)}"


def angle_degrees(weight: torch.Tensor, norms: torch.Tensor, id_a: int, id_b: int) -> float:
    cos_val = torch.dot(weight[id_a], weight[id_b]) / (norms[id_a] * norms[id_b]).clamp_min(1e-12)
    cos_val = torch.clamp(cos_val, -1.0, 1.0)
    return float(torch.rad2deg(torch.arccos(cos_val)).item())


def neighborhood(weight: torch.Tensor, norms: torch.Tensor, anchor_id: int) -> pd.DataFrame:
    anchor = weight[anchor_id]
    anchor_norm = norms[anchor_id].clamp_min(1e-12)
    cos = (weight @ anchor) / (norms * anchor_norm).clamp_min(1e-12)
    cos = torch.clamp(cos, -1.0, 1.0)
    angles = torch.rad2deg(torch.arccos(cos))

    ids = torch.arange(weight.shape[0], device=weight.device)
    df = pd.DataFrame(
        {
            "token_id": ids.cpu().numpy(),
            "angle_deg": angles.cpu().numpy(),
            "magnitude": norms.cpu().numpy(),
        }
    )
    df = df.sort_values(["angle_deg", "token_id"], ascending=[True, True]).reset_index(drop=True)
    return df


def token_picker(prefix: str, infos: list[TokenInfo], max_results: int = 200) -> tuple[str, int | None]:
    method = st.radio(
        f"{prefix} selection method",
        options=["Search text", "Enter token id"],
        horizontal=True,
        key=f"{prefix}_method",
    )

    if method == "Enter token id":
        token_id = st.number_input(
            f"{prefix} token id",
            min_value=0,
            max_value=len(infos) - 1,
            value=0,
            step=1,
            key=f"{prefix}_token_id",
        )
        selected_id = int(token_id)
        st.caption(f"Selected: {label(infos[selected_id])}")
        return "", selected_id

    query = st.text_input(
        f"{prefix} query",
        placeholder="Type token text or substring (case-insensitive)",
        key=f"{prefix}_query",
    )
    candidates = search_tokens(infos, query, max_results=max_results)
    if not candidates:
        st.warning("No matching tokens found.")
        return query, None

    selected_label = st.selectbox(
        f"{prefix} matches",
        [label(x) for x in candidates],
        index=0,
        key=f"{prefix}_matches",
    )
    selected_id = int(selected_label.split("|", 1)[0])
    return query, selected_id


def main() -> None:
    st.set_page_config(page_title="Gemma LM-Head Angle Explorer", layout="wide")
    st.title("Gemma 3 270M LM-Head Angle Explorer")

    with st.sidebar:
        model_name = st.text_input("Model", value="google/gemma-3-270m")
        device = st.selectbox("Device", ["cpu", "cuda"], index=0)
        if device == "cuda" and not torch.cuda.is_available():
            st.warning("CUDA not available in this runtime; using CPU.")
            device = "cpu"

    tokenizer, infos, weight, norms = load_model_assets(model_name, device)
    st.caption(f"Vocab size: {weight.shape[0]:,} | Hidden dim: {weight.shape[1]:,}")

    tab_pair, tab_single = st.tabs(["Pairwise angle", "Single-token neighborhood"])

    with tab_pair:
        st.subheader("Angle between two LM-head vocabulary vectors")
        _, token_a = token_picker("Token A", infos)
        _, token_b = token_picker("Token B", infos)

        if token_a is not None and token_b is not None:
            angle = angle_degrees(weight, norms, token_a, token_b)
            mag_a = float(norms[token_a].item())
            mag_b = float(norms[token_b].item())
            st.metric("Angle (degrees)", f"{angle:.6f}")
            st.write(
                {
                    "token_a_id": token_a,
                    "token_a_raw": infos[token_a].raw,
                    "token_a_magnitude": mag_a,
                    "token_b_id": token_b,
                    "token_b_raw": infos[token_b].raw,
                    "token_b_magnitude": mag_b,
                }
            )

    with tab_single:
        st.subheader("All-token list from closest to furthest angle")
        _, anchor_id = token_picker("Anchor token", infos)
        top_n = st.number_input("Rows to show", min_value=10, max_value=len(infos), value=500, step=10)

        if anchor_id is not None:
            anchor_mag = float(norms[anchor_id].item())
            st.write(
                {
                    "anchor_id": anchor_id,
                    "anchor_raw": infos[anchor_id].raw,
                    "anchor_magnitude": anchor_mag,
                }
            )
            df = neighborhood(weight, norms, anchor_id)
            df["token_raw"] = [infos[i].raw for i in df["token_id"].tolist()]
            df["token_display"] = [infos[i].display for i in df["token_id"].tolist()]
            st.dataframe(df.head(int(top_n)), use_container_width=True)
            st.download_button(
                "Download full sorted neighborhood CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"token_{anchor_id}_neighborhood.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
