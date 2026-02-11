from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthMixer(nn.Module):
    """Depth-wise 3D conv along depth axis (k x 1 x 1), applied on feature maps.
    Matches paper description of lightweight depth mixing.
    """
    def __init__(self, channels: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv3d(
            channels, channels,
            kernel_size=(kernel, 1, 1),
            padding=(pad, 0, 0),
            groups=channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W]
        return self.conv(x)

class VolumetricAdapter(nn.Module):
    """Low-rank bottleneck adapter with depth-wise 3D conv in reduced space.

    Implements Eq.(volumetric_adapter_3dconv) in the paper:
      U + A_up * DWConv3D( phi(U * A_down) )
    where conv operates in reduced rank r space.
    """
    def __init__(self, channels: int, rank: int = 16, kernel: int = 3):
        super().__init__()
        self.down = nn.Linear(channels, rank, bias=False)
        self.up = nn.Linear(rank, channels, bias=False)
        self.act = nn.GELU()
        pad = kernel // 2
        self.dwconv = nn.Conv3d(rank, rank, kernel_size=kernel, padding=pad, groups=rank, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        u = x.permute(0, 2, 3, 4, 1).contiguous()  # [B, D, H, W, C]
        u = self.down(u)                            # [B, D, H, W, r]
        u = self.act(u)
        u = u.permute(0, 4, 1, 2, 3).contiguous()   # [B, r, D, H, W]
        u = self.dwconv(u)
        u = u.permute(0, 2, 3, 4, 1).contiguous()   # [B, D, H, W, r]
        u = self.up(u)                              # [B, D, H, W, C]
        u = u.permute(0, 4, 1, 2, 3).contiguous()   # [B, C, D, H, W]
        return x + u
