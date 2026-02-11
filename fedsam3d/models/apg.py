from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoPromptGenerator(nn.Module):
    """Auto Prompt Generator (APG).

    A compact 3D conv network that produces one query embedding per class from volumetric features.
    Matches the paper's goal: fully automatic prompting (no manual points/boxes).  
    """
    def __init__(self, in_ch: int, num_classes: int, q_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.q_dim = q_dim
        mid = max(64, in_ch // 2)

        self.net = nn.Sequential(
            nn.Conv3d(in_ch, mid, 3, padding=1),
            nn.InstanceNorm3d(mid, affine=True),
            nn.GELU(),
            nn.Conv3d(mid, mid, 3, padding=1),
            nn.InstanceNorm3d(mid, affine=True),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(mid, num_classes * q_dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, C, D, H, W]
        x = self.net(feats)
        x = self.pool(x).flatten(1)     # [B, mid]
        q = self.fc(x)                  # [B, num_classes*q_dim]
        q = q.view(-1, self.num_classes, self.q_dim)
        return q
