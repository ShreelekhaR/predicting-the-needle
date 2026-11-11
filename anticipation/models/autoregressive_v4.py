"""V4 Autoregressive models for variable-horizon prediction.

Arch A (MultiHorizonAR): 4 output heads, one per fixed horizon.
Arch B (TemporalConditionedAR): single output conditioned on queried future offset.
Both support use_diffs: fuse change velocity [x_i; x_i - x_{i-1}] into input tokens.
"""

import torch
import torch.nn as nn

from .encodings import TemporalEncoding, GeographicEncoding, fourier_encode
from .temporal_diff import TemporalDiffEncoder, GatedDiffEncoder


class MultiHorizonAR(nn.Module):
    """Arch A: Causal transformer with 4 output heads (1wk/1mo/3mo/6mo).

    Input: [BEFORE, w1, ..., wN] with temporal offsets.
    Output: 4 predicted embeddings, one per horizon.
    """

    def __init__(self, emb_dim=1152, n_heads=8, n_layers=4, ff_dim=2048,
                 dropout=0.2, max_seq_len=53, n_horizons=4,
                 use_sinusoidal_temporal=False, use_diffs=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len
        self.n_horizons = n_horizons
        self.use_sinusoidal_temporal = use_sinusoidal_temporal
        self.use_diffs = use_diffs

        self.before_proj = nn.Linear(emb_dim, emb_dim)
        self.input_norm = nn.LayerNorm(emb_dim)

        # Change velocity encoder
        if use_diffs:
            self.diff_encoder = TemporalDiffEncoder(emb_dim)

        # Temporal encoding
        if use_sinusoidal_temporal:
            self.temporal_enc = TemporalEncoding(emb_dim)
        else:
            # Learned positional embedding
            self.pos_embedding = nn.Embedding(max_seq_len + 1, emb_dim)

        self.geo_enc = GeographicEncoding(emb_dim)

        # Causal transformer
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)

        # Per-horizon output heads
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, emb_dim),
            )
            for _ in range(n_horizons)
        ])

    def _causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, sequence, mask, before_emb, temporal_offsets=None,
                latitude=None, longitude=None, **kwargs):
        B, T, D = sequence.shape
        device = sequence.device

        # Apply change velocity encoding
        if self.use_diffs:
            sequence = self.diff_encoder(sequence, mask)

        # Prepend BEFORE token
        before_tok = self.before_proj(before_emb).unsqueeze(1)
        x = torch.cat([before_tok, self.input_norm(sequence)], dim=1)  # (B, T+1, D)

        # Temporal/positional encoding
        if self.use_sinusoidal_temporal and temporal_offsets is not None:
            zero_before = torch.zeros(B, 1, D, device=device)
            t_enc = self.temporal_enc(temporal_offsets)
            x = x + torch.cat([zero_before, t_enc], dim=1)
        else:
            positions = torch.arange(T + 1, device=device).unsqueeze(0).expand(B, -1)
            x = x + self.pos_embedding(positions)

        # Geographic encoding
        if latitude is not None and longitude is not None:
            x = x + self.geo_enc(latitude, longitude).unsqueeze(1)

        # Causal attention
        causal_mask = self._causal_mask(T + 1, device)
        pad_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=device),
            ~mask
        ], dim=1)

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)

        # Get last valid hidden state
        lengths = mask.sum(dim=1).long()
        batch_idx = torch.arange(B, device=device)
        last_hidden = x[batch_idx, lengths]  # (B, D)

        # Predict at each horizon
        preds = torch.stack([head(last_hidden) for head in self.horizon_heads], dim=1)  # (B, 4, D)
        return preds


class TemporalConditionedAR(nn.Module):
    """Arch B: Causal transformer conditioned on a queried future time offset.

    Input: [BEFORE, w1, ..., wN] with sinusoidal temporal offsets.
    Query: future time offset (days) encoded via sinusoidal features.
    Output: single predicted embedding at the queried future time.
    """

    def __init__(self, emb_dim=1152, n_heads=8, n_layers=4, ff_dim=2048,
                 dropout=0.2, max_seq_len=53, use_diffs=False,
                 use_learned_input_pos=False, use_gated_diffs=False,
                 time_from_sequence=False, residual_last=False):
        super().__init__()
        # residual_last: pred = last valid input obs + output_head(...). "No change" is the
        # zero solution, so the model starts level with the static baseline and only has to
        # learn the change. Same parameters; default False keeps old behaviour.
        self.residual_last = residual_last
        # time_from_sequence: the time head reads the encoded sequence (last valid token)
        # instead of the query-conditioned state, so "when will this finish" does not
        # depend on the queried horizon. Same parameters; default False keeps old behaviour.
        self.time_from_sequence = time_from_sequence
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len
        self.use_diffs = use_diffs or use_gated_diffs
        self.use_learned_input_pos = use_learned_input_pos

        self.before_proj = nn.Linear(emb_dim, emb_dim)
        self.input_norm = nn.LayerNorm(emb_dim)

        # Change velocity encoder
        if use_gated_diffs:
            self.diff_encoder = GatedDiffEncoder(emb_dim)
        elif use_diffs:
            self.diff_encoder = TemporalDiffEncoder(emb_dim)

        # Input temporal encoding: learned positional or sinusoidal
        if use_learned_input_pos:
            self.pos_embedding = nn.Embedding(max_seq_len + 1, emb_dim)
        else:
            self.temporal_enc = TemporalEncoding(emb_dim)

        # Query time encoding (always sinusoidal — continuous future offset)
        self.query_enc = TemporalEncoding(emb_dim)

        self.geo_enc = GeographicEncoding(emb_dim)

        # Causal transformer
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)

        # Cross-attention: query attends to encoder output
        self.query_cross_attn = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.query_norm = nn.LayerNorm(emb_dim)

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_dim),
        )

        # Time-to-completion head (optional, enabled by predict_time=True)
        self.time_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def _causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, sequence, mask, before_emb, temporal_offsets=None,
                query_offset=None, latitude=None, longitude=None, **kwargs):
        """
        Args:
            query_offset: (B,) days into the future to predict
        """
        B, T, D = sequence.shape
        device = sequence.device

        if self.residual_last or self.time_from_sequence:
            # index of the last valid obs (robust to masked-out holes from frame dropout)
            last_i = (mask.long() * torch.arange(T, device=device)).argmax(dim=1)
            raw_last = sequence[torch.arange(B, device=device), last_i]

        # Apply change velocity encoding
        if self.use_diffs:
            sequence = self.diff_encoder(sequence, mask)

        # Prepend BEFORE token
        before_tok = self.before_proj(before_emb).unsqueeze(1)
        x = torch.cat([before_tok, self.input_norm(sequence)], dim=1)  # (B, T+1, D)

        # Input temporal encoding
        if self.use_learned_input_pos:
            positions = torch.arange(T + 1, device=device).unsqueeze(0).expand(B, -1)
            x = x + self.pos_embedding(positions)
        elif temporal_offsets is not None:
            zero_before = torch.zeros(B, 1, D, device=device)
            t_enc = self.temporal_enc(temporal_offsets)
            x = x + torch.cat([zero_before, t_enc], dim=1)

        # Geographic encoding
        if latitude is not None and longitude is not None:
            x = x + self.geo_enc(latitude, longitude).unsqueeze(1)

        # Causal attention over input sequence
        causal_mask = self._causal_mask(T + 1, device)
        pad_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=device),
            ~mask
        ], dim=1)

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)

        # Encode query time offset
        query_tok = self.query_enc(query_offset.unsqueeze(-1)).squeeze(1)  # (B, D)
        query_tok = query_tok.unsqueeze(1)  # (B, 1, D)

        # Cross-attention: query attends to encoded sequence
        query_out, _ = self.query_cross_attn(
            self.query_norm(query_tok), x, x,
            key_padding_mask=pad_mask,
        )
        query_out = query_tok + query_out  # residual

        hidden = query_out.squeeze(1)  # (B, D)
        pred = self.output_head(hidden)  # (B, D)
        if self.residual_last:
            pred = raw_last + pred

        if kwargs.get("predict_time", False):
            if self.time_from_sequence:
                # encoded state of the last valid obs; x = [first_obs, w1..wN] so +1
                seq_state = x[torch.arange(B, device=device), last_i + 1]  # (B, D)
                time_pred = self.time_head(seq_state).squeeze(-1)
            else:
                time_pred = self.time_head(hidden).squeeze(-1)  # (B,)
            return pred, time_pred

        return pred


class JEPAConditionedAR(nn.Module):
    """JEPA-style encoder-predictor conditioned on a queried future time offset.

    Encoder: Transformer over [BEFORE, w1, ..., wN].
    Predictor: Learnable target queries cross-attend to encoded context through
    multiple predictor blocks (self-attn + cross-attn + FFN). The predictor
    operates in a lower-dimensional bottleneck (pred_dim < emb_dim).

    The query time offset is added to the target queries so the predictor
    knows which future date to predict.
    """

    def __init__(self, emb_dim=1152, pred_dim=768, n_heads=8,
                 n_encoder_layers=4, n_predictor_layers=4,
                 ff_dim=2048, pred_ff_dim=1536,
                 dropout=0.2, max_seq_len=53, n_target_queries=4,
                 use_diffs=False, use_learned_input_pos=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.pred_dim = pred_dim
        self.use_diffs = use_diffs
        self.use_learned_input_pos = use_learned_input_pos

        self.before_proj = nn.Linear(emb_dim, emb_dim)
        self.input_norm = nn.LayerNorm(emb_dim)

        if use_diffs:
            self.diff_encoder = TemporalDiffEncoder(emb_dim)

        # Input temporal encoding
        if use_learned_input_pos:
            self.pos_embedding = nn.Embedding(max_seq_len + 1, emb_dim)
        else:
            self.temporal_enc = TemporalEncoding(emb_dim)

        # Query time encoding (always sinusoidal)
        self.query_enc = TemporalEncoding(pred_dim)

        self.geo_enc = GeographicEncoding(emb_dim)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # Predictor
        self.target_queries = nn.Parameter(torch.randn(1, n_target_queries, pred_dim) * 0.02)
        self.ctx_proj = nn.Linear(emb_dim, pred_dim)

        self.predictor_blocks = nn.ModuleList()
        for _ in range(n_predictor_layers):
            self.predictor_blocks.append(_PredictorBlock(
                pred_dim, n_heads, pred_ff_dim, dropout))

        # Output
        self.output_norm = nn.LayerNorm(pred_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(pred_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_dim),
        )

    def forward(self, sequence, mask, before_emb, temporal_offsets=None,
                query_offset=None, latitude=None, longitude=None, **kwargs):
        B, T, D = sequence.shape
        device = sequence.device

        if self.use_diffs:
            sequence = self.diff_encoder(sequence, mask)

        # Prepend BEFORE token
        before_tok = self.before_proj(before_emb).unsqueeze(1)
        x = torch.cat([before_tok, self.input_norm(sequence)], dim=1)

        # Input temporal encoding
        if self.use_learned_input_pos:
            positions = torch.arange(T + 1, device=device).unsqueeze(0).expand(B, -1)
            x = x + self.pos_embedding(positions)
        elif temporal_offsets is not None:
            zero_before = torch.zeros(B, 1, D, device=device)
            t_enc = self.temporal_enc(temporal_offsets)
            x = x + torch.cat([zero_before, t_enc], dim=1)

        if latitude is not None and longitude is not None:
            x = x + self.geo_enc(latitude, longitude).unsqueeze(1)

        # Encode
        pad_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=device), ~mask
        ], dim=1)
        context = self.encoder(x, src_key_padding_mask=pad_mask)

        # Project context to predictor dimension
        ctx = self.ctx_proj(context)

        # Condition target queries on query time
        queries = self.target_queries.expand(B, -1, -1)
        if query_offset is not None:
            time_enc = self.query_enc(query_offset.unsqueeze(-1)).squeeze(1)  # (B, pred_dim)
            queries = queries + time_enc.unsqueeze(1)

        # Predictor blocks
        for block in self.predictor_blocks:
            queries = block(queries, ctx, pad_mask)

        # Pool and project
        pooled = self.output_norm(queries.mean(dim=1))
        return self.output_proj(pooled)


class _PredictorBlock(nn.Module):
    """Self-attention + cross-attention + FFN block for JEPA predictor."""

    def __init__(self, dim, n_heads, ff_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_ctx = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x, context, context_mask=None):
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h)
        x = x + h

        h = self.norm2(x)
        ctx = self.norm_ctx(context)
        h, _ = self.cross_attn(h, ctx, ctx, key_padding_mask=context_mask)
        x = x + h

        h = self.norm3(x)
        x = x + self.ffn(h)
        return x
