"""Query-conditioned GRU baseline for V9 pipeline.

Same as models/lstm_v9.py but with nn.GRU replacing nn.LSTM.
Drop-in for scripts/train.py via --model gru.
"""

import torch
import torch.nn as nn

from .encodings import TemporalEncoding, GeographicEncoding
from .temporal_diff import TemporalDiffEncoder, GatedDiffEncoder


class GRUPredictorV9(nn.Module):
    def __init__(self, emb_dim=1152, n_heads=8, hidden_dim=512, n_layers=2,
                 ff_dim=2048, dropout=0.2, max_seq_len=53,
                 use_diffs=False, use_gated_diffs=False,
                 use_learned_input_pos=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len
        self.use_diffs = use_diffs
        self.use_learned_input_pos = use_learned_input_pos

        self.before_proj = nn.Linear(emb_dim, emb_dim)
        self.input_norm = nn.LayerNorm(emb_dim)

        if use_diffs:
            if use_gated_diffs:
                self.diff_encoder = GatedDiffEncoder(emb_dim)
            else:
                self.diff_encoder = TemporalDiffEncoder(emb_dim)

        if use_learned_input_pos:
            self.pos_embedding = nn.Embedding(max_seq_len + 1, emb_dim)
        else:
            self.temporal_enc = TemporalEncoding(emb_dim)

        self.geo_enc = GeographicEncoding(emb_dim)

        self.rnn = nn.GRU(
            input_size=emb_dim, hidden_size=hidden_dim, num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True, bidirectional=True,
        )
        self.rnn_out_proj = nn.Linear(2 * hidden_dim, emb_dim)

        self.query_enc = TemporalEncoding(emb_dim)
        self.query_cross_attn = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.query_norm = nn.LayerNorm(emb_dim)

        self.output_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_dim),
        )

        self.time_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, sequence, mask, before_emb, temporal_offsets=None,
                query_offset=None, latitude=None, longitude=None, **kwargs):
        B, T, D = sequence.shape
        device = sequence.device

        if self.use_diffs:
            sequence = self.diff_encoder(sequence, mask)

        before_tok = self.before_proj(before_emb).unsqueeze(1)
        x = torch.cat([before_tok, self.input_norm(sequence)], dim=1)  # (B, T+1, D)

        if self.use_learned_input_pos:
            positions = torch.arange(T + 1, device=device).unsqueeze(0).expand(B, -1)
            x = x + self.pos_embedding(positions)
        elif temporal_offsets is not None:
            zero_before = torch.zeros(B, 1, D, device=device)
            t_enc = self.temporal_enc(temporal_offsets)
            x = x + torch.cat([zero_before, t_enc], dim=1)

        if latitude is not None and longitude is not None:
            x = x + self.geo_enc(latitude, longitude).unsqueeze(1)

        pad_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=device),
            ~mask,
        ], dim=1)

        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        rnn_out, _ = self.rnn(x)  # (B, T+1, 2*hidden)
        h = self.rnn_out_proj(rnn_out)  # (B, T+1, D)

        query_tok = self.query_enc(query_offset.unsqueeze(-1)).squeeze(1).unsqueeze(1)  # (B, 1, D)
        query_out, _ = self.query_cross_attn(
            self.query_norm(query_tok), h, h,
            key_padding_mask=pad_mask,
        )
        query_out = query_tok + query_out

        hidden = query_out.squeeze(1)
        pred = self.output_head(hidden)

        if kwargs.get("predict_time", False):
            time_pred = self.time_head(hidden).squeeze(-1)
            return pred, time_pred

        return pred
