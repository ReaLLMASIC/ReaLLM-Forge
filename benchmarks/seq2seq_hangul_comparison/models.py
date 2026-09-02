#!/usr/bin/env python3
"""Seq2Seq Transformer architectures for English-to-Korean translation.

Implements:
1. Seq2SeqThreeHotConditional: Transformer Seq2Seq with Cognetta et al. 3-step conditional RNN decoder.
2. Seq2SeqThreeHotIndependent: Transformer Seq2Seq with Song et al. 3 independent output heads.
3. Seq2SeqHangulFactorizer: Transformer Seq2Seq with 23-lane articulatory factorizer embeddings and multi-heads.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class BaseSeq2SeqTransformer(nn.Module):
    """Base Transformer Encoder-Decoder matching EACL 2023 paper specifications.

    Encoder: 6 layers, d_model=512, d_ff=512, 8 heads.
    Decoder: 6 layers, d_model=512, d_ff=512, 8 heads.
    """

    def __init__(
        self,
        src_vocab_size: int,
        d_model: int = 512,
        d_ff: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def encode(
        self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def decode_step(
        self,
        tgt_emb: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tgt_emb = self.pos_decoder(tgt_emb * math.sqrt(self.d_model))
        return self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )


class Seq2SeqThreeHotConditional(BaseSeq2SeqTransformer):
    """Transformer Seq2Seq with Conditional 3-Step RNN Decoder (Cognetta et al., EACL 2023)."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_sizes: List[int],  # [len(lane0), len(lane1), len(lane2)]
        d_model: int = 512,
        d_ff: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__(
            src_vocab_size=src_vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
        )
        self.tgt_vocab_sizes = tgt_vocab_sizes

        # 3 target embeddings
        self.emb_i = nn.Embedding(tgt_vocab_sizes[0], d_model, padding_idx=0)
        self.emb_v = nn.Embedding(tgt_vocab_sizes[1], d_model, padding_idx=0)
        self.emb_f = nn.Embedding(tgt_vocab_sizes[2], d_model, padding_idx=0)

        # Conditional 3-step RNN transition matrices (Section 2.3)
        self.W_e = nn.Linear(d_model, d_model, bias=False)
        self.W_h = nn.Linear(d_model, d_model, bias=True)
        self.h0 = nn.Parameter(torch.zeros(1, 1, d_model))

        # Output projection heads
        self.head_i = nn.Linear(d_model, tgt_vocab_sizes[0], bias=False)
        self.head_v = nn.Linear(d_model, tgt_vocab_sizes[1], bias=False)
        self.head_f = nn.Linear(d_model, tgt_vocab_sizes[2], bias=False)

    def embed_target(self, tgt: torch.Tensor) -> torch.Tensor:
        # tgt: (B, T, 3)
        return (
            self.emb_i(tgt[:, :, 0])
            + self.emb_v(tgt[:, :, 1])
            + self.emb_f(tgt[:, :, 2])
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with teacher forcing."""
        memory = self.encode(src, src_key_padding_mask=src_padding_mask)
        tgt_emb = self.embed_target(tgt)
        seq_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=tgt.device)

        # Decoder states h_t: (B, T, d_model)
        h_t = self.decode_step(
            tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask
        )

        b, t, d = h_t.size()
        h0_expanded = self.h0.expand(b, t, d)

        # Step 1: Initial consonant / non-Korean token
        h_i = torch.tanh(self.W_e(h_t) + self.W_h(h0_expanded))
        logits_i = self.head_i(h_i)

        # Step 2: Medial vowel (conditioned on ground truth chosen i)
        embi_true = self.emb_i(tgt[:, :, 0])
        h_v = torch.tanh(self.W_e(embi_true) + self.W_h(h_i))
        logits_v = self.head_v(h_v)

        # Step 3: Final consonant (conditioned on ground truth chosen v)
        embv_true = self.emb_v(tgt[:, :, 1])
        h_f = torch.tanh(self.W_e(embv_true) + self.W_h(h_v))
        logits_f = self.head_f(h_f)

        return logits_i, logits_v, logits_f

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 120,
        sos_id: int = 1,
        eos_id: int = 2,
    ) -> torch.Tensor:
        """Autoregressive greedy generation."""
        self.eval()
        device = src.device
        b = src.size(0)
        memory = self.encode(src)

        # Start with SOS token (1, 0, 0)
        generated = torch.zeros((b, 1, 3), dtype=torch.long, device=device)
        generated[:, 0, 0] = sos_id

        finished = torch.zeros(b, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_emb = self.embed_target(generated)
            seq_len = generated.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
            h_t = self.decode_step(tgt_emb, memory, tgt_mask=tgt_mask)

            last_h = h_t[:, -1:, :]  # (B, 1, d)
            h0_expanded = self.h0.expand(b, 1, self.d_model)

            # 1. Predict i
            h_i = torch.tanh(self.W_e(last_h) + self.W_h(h0_expanded))
            pi_i = self.head_i(h_i).argmax(dim=-1)  # (B, 1)

            # 2. Predict v conditioned on predicted i
            embi_pred = self.emb_i(pi_i)
            h_v = torch.tanh(self.W_e(embi_pred) + self.W_h(h_i))
            pi_v = self.head_v(h_v).argmax(dim=-1)  # (B, 1)

            # 3. Predict f conditioned on predicted v
            embv_pred = self.emb_v(pi_v)
            h_f = torch.tanh(self.W_e(embv_pred) + self.W_h(h_v))
            pi_f = self.head_f(h_f).argmax(dim=-1)  # (B, 1)

            next_step = torch.cat([pi_i, pi_v, pi_f], dim=-1).unsqueeze(1)  # (B, 1, 3)
            generated = torch.cat([generated, next_step], dim=1)

            # Check EOS
            is_eos = pi_i.squeeze(-1) == eos_id
            finished = finished | is_eos
            if finished.all():
                break

        return generated


class Seq2SeqThreeHotIndependent(BaseSeq2SeqTransformer):
    """Transformer Seq2Seq with Independent 3 Output Heads (Song et al., 2018)."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_sizes: List[int],
        d_model: int = 512,
        d_ff: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__(
            src_vocab_size=src_vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
        )
        self.tgt_vocab_sizes = tgt_vocab_sizes

        self.emb_i = nn.Embedding(tgt_vocab_sizes[0], d_model, padding_idx=0)
        self.emb_v = nn.Embedding(tgt_vocab_sizes[1], d_model, padding_idx=0)
        self.emb_f = nn.Embedding(tgt_vocab_sizes[2], d_model, padding_idx=0)

        # 3 Independent output heads directly on h_t
        self.head_i = nn.Linear(d_model, tgt_vocab_sizes[0], bias=False)
        self.head_v = nn.Linear(d_model, tgt_vocab_sizes[1], bias=False)
        self.head_f = nn.Linear(d_model, tgt_vocab_sizes[2], bias=False)

    def embed_target(self, tgt: torch.Tensor) -> torch.Tensor:
        return (
            self.emb_i(tgt[:, :, 0])
            + self.emb_v(tgt[:, :, 1])
            + self.emb_f(tgt[:, :, 2])
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory = self.encode(src, src_key_padding_mask=src_padding_mask)
        tgt_emb = self.embed_target(tgt)
        seq_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=tgt.device)
        h_t = self.decode_step(
            tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask
        )

        logits_i = self.head_i(h_t)
        logits_v = self.head_v(h_t)
        logits_f = self.head_f(h_t)
        return logits_i, logits_v, logits_f

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 120,
        sos_id: int = 1,
        eos_id: int = 2,
    ) -> torch.Tensor:
        self.eval()
        device = src.device
        b = src.size(0)
        memory = self.encode(src)

        generated = torch.zeros((b, 1, 3), dtype=torch.long, device=device)
        generated[:, 0, 0] = sos_id
        finished = torch.zeros(b, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_emb = self.embed_target(generated)
            seq_len = generated.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
            h_t = self.decode_step(tgt_emb, memory, tgt_mask=tgt_mask)

            last_h = h_t[:, -1:, :]
            pi_i = self.head_i(last_h).argmax(dim=-1)
            pi_v = self.head_v(last_h).argmax(dim=-1)
            pi_f = self.head_f(last_h).argmax(dim=-1)

            next_step = torch.cat([pi_i, pi_v, pi_f], dim=-1).unsqueeze(1)
            generated = torch.cat([generated, next_step], dim=1)

            is_eos = pi_i.squeeze(-1) == eos_id
            finished = finished | is_eos
            if finished.all():
                break

        return generated


class Seq2SeqHangulFactorizer(BaseSeq2SeqTransformer):
    """Transformer Seq2Seq with 23-Lane Hangul Factorizer."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_sizes: List[int],  # 23 lane sizes
        d_model: int = 512,
        d_ff: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__(
            src_vocab_size=src_vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
        )
        self.tgt_vocab_sizes = tgt_vocab_sizes
        self.num_lanes = len(tgt_vocab_sizes)

        # 23 Embedding layers (summed)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab, d_model, padding_idx=0) for vocab in tgt_vocab_sizes]
        )

        # 23 Independent linear prediction heads
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, vocab, bias=False) for vocab in tgt_vocab_sizes]
        )

    def embed_target(self, tgt: torch.Tensor) -> torch.Tensor:
        # tgt: (B, T, 23)
        emb = None
        for k in range(self.num_lanes):
            lane_emb = self.embeddings[k](tgt[:, :, k])
            emb = lane_emb if emb is None else emb + lane_emb
        return emb

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        memory = self.encode(src, src_key_padding_mask=src_padding_mask)
        tgt_emb = self.embed_target(tgt)
        seq_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=tgt.device)
        h_t = self.decode_step(
            tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask
        )

        logits = [self.heads[k](h_t) for k in range(self.num_lanes)]
        return logits

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 120,
        sos_id: int = 1,
        eos_id: int = 2,
    ) -> torch.Tensor:
        self.eval()
        device = src.device
        b = src.size(0)
        memory = self.encode(src)

        generated = torch.zeros((b, 1, self.num_lanes), dtype=torch.long, device=device)
        generated[:, 0, 0] = sos_id  # SOS in Lane 0, PAD in rest
        finished = torch.zeros(b, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_emb = self.embed_target(generated)
            seq_len = generated.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
            h_t = self.decode_step(tgt_emb, memory, tgt_mask=tgt_mask)

            last_h = h_t[:, -1:, :]
            lane_preds = [self.heads[k](last_h).argmax(dim=-1) for k in range(self.num_lanes)]
            next_step = torch.cat(lane_preds, dim=-1).unsqueeze(1)  # (B, 1, 23)
            generated = torch.cat([generated, next_step], dim=1)

            is_eos = lane_preds[0].squeeze(-1) == eos_id
            finished = finished | is_eos
            if finished.all():
                break

        return generated
