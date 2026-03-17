from __future__ import annotations

import torch
from torch import Tensor, nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ComputeNodeCell(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        delay_bins: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cache_q_ln = nn.LayerNorm(d_model)
        self.cache_kv_ln = nn.LayerNorm(d_model)
        self.cache_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cache_ff_ln = nn.LayerNorm(d_model)
        self.cache_ff = FeedForward(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

        self.packet_ln = nn.LayerNorm(d_model)
        self.packet_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.packet_ff_ln = nn.LayerNorm(d_model)
        self.packet_ff = FeedForward(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

        self.out_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        packets: Tensor,
        packet_mask: Tensor,
        cache: Tensor,
        cache_mask: Tensor,
    ) -> dict[str, Tensor]:
        x = packets

        rows_with_cache = cache_mask.any(dim=1)
        if rows_with_cache.any():
            x = x.clone()
            q = self.cache_q_ln(x[rows_with_cache])
            kv = self.cache_kv_ln(cache[rows_with_cache])
            attn_out, _ = self.cache_attn(
                q,
                kv,
                kv,
                key_padding_mask=~cache_mask[rows_with_cache],
                need_weights=False,
            )
            x[rows_with_cache] = x[rows_with_cache] + attn_out
        x = x + self.cache_ff(self.cache_ff_ln(x))
        x = x.masked_fill(~packet_mask.unsqueeze(-1), 0.0)

        attn_in = self.packet_ln(x)
        attn_out, _ = self.packet_attn(
            attn_in,
            attn_in,
            attn_in,
            key_padding_mask=~packet_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.packet_ff(self.packet_ff_ln(x))
        x = x.masked_fill(~packet_mask.unsqueeze(-1), 0.0)

        return {"hidden": self.out_ln(x)}
