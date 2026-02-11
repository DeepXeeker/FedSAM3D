from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: [B, Ccls, dim], kv: [B, N, dim]
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        out, _ = self.attn(qn, kvn, kvn, need_weights=False)
        q = q + out
        q = q + self.mlp(self.norm_mlp(q))
        return q

class PromptConditioned3DDecoder(nn.Module):
    """Prompt-conditioned decoder producing C binary organ masks (sigmoid per class).

    Design:
      - Flatten volumetric tokens (D*H*W) as key/value memory
      - Class queries attend to tokens via a few cross-attention layers
      - Produce per-class logits by dot(query_proj, token_feat) and reshape back to volume
    """
    def __init__(self, feat_dim: int, q_dim: int, num_classes: int, num_layers: int = 3, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.q_proj = nn.Linear(q_dim, feat_dim)
        self.kv_proj = nn.Linear(feat_dim, feat_dim)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(feat_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
        ])
        self.mask_proj = nn.Linear(feat_dim, feat_dim)

    def forward(self, feats: torch.Tensor, queries: torch.Tensor, out_shape) -> torch.Tensor:
        # feats: [B, C, D, H, W]  (C=feat_dim)
        # queries: [B, num_classes, q_dim]
        B, C, D, H, W = feats.shape
        N = D * H * W
        tokens = feats.view(B, C, N).permute(0, 2, 1).contiguous()  # [B, N, C]
        tokens = self.kv_proj(tokens)

        q = self.q_proj(queries)  # [B, num_classes, C]
        for blk in self.blocks:
            q = blk(q, tokens)

        # Mask logits: (B, num_classes, N)
        q_m = self.mask_proj(q)  # [B, num_classes, C]
        logits = torch.einsum("bkc,bnc->bkn", q_m, tokens)
        logits = logits.view(B, self.num_classes, D, H, W)

        # Upsample logits to requested output spatial shape
        logits = F.interpolate(logits, size=out_shape, mode="trilinear", align_corners=False)
        return logits
