"""Temporal difference encoder: fuse consecutive embedding diffs with raw tokens."""

import torch
import torch.nn as nn


class TemporalDiffEncoder(nn.Module):
    """Fuse consecutive embedding differences with raw tokens.

    Computes delta_i = x_{i+1} - x_i for consecutive valid tokens,
    concatenates [x_i; delta_i], and projects back to emb_dim.
    First valid token and padding positions get zero diffs.

    Computed inside the model (not dataset) so it works correctly
    with frame-drop augmentation.
    """

    def __init__(self, emb_dim=1152):
        super().__init__()
        self.fusion_proj = nn.Linear(emb_dim * 2, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, sequence, mask):
        B, T, D = sequence.shape

        diffs = torch.zeros_like(sequence)
        diffs[:, :-1] = sequence[:, 1:] - sequence[:, :-1]

        valid_pair = mask[:, :-1] & mask[:, 1:]
        diff_mask = torch.zeros(B, T, dtype=torch.bool, device=sequence.device)
        diff_mask[:, :-1] = valid_pair
        diffs = diffs * diff_mask.unsqueeze(-1).float()

        fused = torch.cat([sequence, diffs], dim=-1)
        fused = self.fusion_proj(fused)
        return self.norm(fused)


class GatedDiffEncoder(nn.Module):
    """Gated fusion of consecutive embedding differences with raw tokens.

    gate = sigmoid(W [x_i; delta_i])
    output = x_i + gate * MLP(delta_i)

    Preserves raw token via residual. Gate learns when to use diffs
    (real change) vs ignore them (noise, clouds, seasonal).
    """

    def __init__(self, emb_dim=1152):
        super().__init__()
        self.diff_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, sequence, mask):
        B, T, D = sequence.shape

        diffs = torch.zeros_like(sequence)
        diffs[:, :-1] = sequence[:, 1:] - sequence[:, :-1]

        valid_pair = mask[:, :-1] & mask[:, 1:]
        diff_mask = torch.zeros(B, T, dtype=torch.bool, device=sequence.device)
        diff_mask[:, :-1] = valid_pair
        diffs = diffs * diff_mask.unsqueeze(-1).float()

        g = self.gate(torch.cat([sequence, diffs], dim=-1))
        diff_signal = self.diff_mlp(diffs)
        fused = sequence + g * diff_signal

        return self.norm(fused)
