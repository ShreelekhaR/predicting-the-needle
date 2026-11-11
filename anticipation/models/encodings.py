"""Shared Fourier feature encodings for temporal and geographic inputs."""

import math

import torch
import torch.nn as nn


def fourier_encode(x, num_frequencies=32):
    """Fourier positional encoding.

    Maps scalar inputs to a higher-dimensional sinusoidal representation.

    Args:
        x: input tensor of any shape (...)
        num_frequencies: number of frequency bands

    Returns:
        Fourier features of shape (..., 2 * num_frequencies)
    """
    freqs = 2.0 ** torch.arange(num_frequencies, device=x.device, dtype=x.dtype) * math.pi
    x = x.unsqueeze(-1) * freqs  # (..., num_frequencies)
    return torch.cat([x.sin(), x.cos()], dim=-1)  # (..., 2 * num_frequencies)


class TemporalEncoding(nn.Module):
    """Encode per-token temporal offsets (days before current_date)."""

    def __init__(self, emb_dim, num_frequencies=32):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.proj = nn.Linear(2 * num_frequencies, emb_dim)

    def forward(self, temporal_offsets):
        """
        Args:
            temporal_offsets: (B, T) days before current_date, normalized by /365

        Returns:
            (B, T, emb_dim) temporal encoding to add to token embeddings
        """
        enc = fourier_encode(temporal_offsets / 365.0, self.num_frequencies)
        return self.proj(enc)


class GeographicEncoding(nn.Module):
    """Encode per-sample lat/lng coordinates."""

    def __init__(self, emb_dim, num_frequencies=32):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.proj = nn.Linear(4 * num_frequencies, emb_dim)

    def forward(self, latitude, longitude):
        """
        Args:
            latitude: (B,) in degrees
            longitude: (B,) in degrees

        Returns:
            (B, emb_dim) geographic encoding to add to embeddings
        """
        lat_enc = fourier_encode(latitude / 90.0, self.num_frequencies)   # normalize to [-1, 1]
        lng_enc = fourier_encode(longitude / 180.0, self.num_frequencies)  # normalize to [-1, 1]
        geo = torch.cat([lat_enc, lng_enc], dim=-1)  # (B, 4 * num_freq)
        return self.proj(geo)
