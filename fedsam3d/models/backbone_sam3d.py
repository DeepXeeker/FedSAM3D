from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import DepthMixer, VolumetricAdapter

class SAM2DWrapper(nn.Module):
    """Thin wrapper around Meta's Segment Anything image encoder (2D ViT).

    We use segment-anything if installed; otherwise raise a clear error.
    The encoder is applied slice-wise (D treated as batch dimension), then enhanced with
    depth-aware modules (depth positional embedding, depth mixer, volumetric adapters).
    """
    def __init__(
        self,
        sam_variant: str,
        sam_checkpoint: str,
        adapter_rank: int,
        adapter_layers: int,
        depth_mixer_kernel: int,
        enable_depth_pos: bool = True,
        enable_depth_mixer: bool = True,
        freeze_sam: bool = True,
    ):
        super().__init__()
        try:
            from segment_anything import build_sam_vit_b, build_sam_vit_l, build_sam_vit_h
        except Exception as e:
            raise ImportError(
                "segment-anything is required for SAM backbone. Install with: pip install segment-anything"
            ) from e

        if sam_variant == "vit_b":
            sam = build_sam_vit_b(checkpoint=sam_checkpoint)
        elif sam_variant == "vit_l":
            sam = build_sam_vit_l(checkpoint=sam_checkpoint)
        elif sam_variant == "vit_h":
            sam = build_sam_vit_h(checkpoint=sam_checkpoint)
        else:
            raise ValueError(f"Unknown sam_variant: {sam_variant}")

        self.image_encoder = sam.image_encoder  # ImageEncoderViT
        self.embed_dim = self.image_encoder.neck[0].out_channels  # typically 256 for SAM neck

        if freeze_sam:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        self.enable_depth_pos = enable_depth_pos
        self.enable_depth_mixer = enable_depth_mixer

        # Depth positional embedding table (trainable, initialized to zeros)
        self.max_depth = 512  # safe upper bound; supports interpolation if needed
        if enable_depth_pos:
            self.depth_pos = nn.Embedding(self.max_depth, self.embed_dim)
            nn.init.zeros_(self.depth_pos.weight)

        if enable_depth_mixer:
            self.depth_mixer = DepthMixer(self.embed_dim, kernel=depth_mixer_kernel)

        self.adapters = nn.ModuleList([
            VolumetricAdapter(self.embed_dim, rank=adapter_rank, kernel=3)
            for _ in range(adapter_layers)
        ])

        # Make adapters trainable
        for p in self.adapters.parameters():
            p.requires_grad = True
        if enable_depth_pos:
            for p in self.depth_pos.parameters():
                p.requires_grad = True
        if enable_depth_mixer:
            for p in self.depth_mixer.parameters():
                p.requires_grad = True

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        """Encode a volume.

        Args:
          vol: [B, 1, D, H, W] (CT intensity normalized)

        Returns:
          feats: [B, C, D', H', W'] where C=self.embed_dim and spatial dims are SAM neck resolution.
        """
        assert vol.ndim == 5 and vol.shape[1] == 1, "Expected [B,1,D,H,W]"
        B, _, D, H, W = vol.shape

        # SAM expects 3-channel input; repeat the CT channel.
        x = vol.repeat(1, 3, 1, 1, 1)  # [B,3,D,H,W]
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * D, 3, H, W)  # [B*D,3,H,W]

        # Image encoder produces [B*D, C, h', w']
        feats2d = self.image_encoder(x)
        _, C, h2, w2 = feats2d.shape

        feats = feats2d.view(B, D, C, h2, w2).permute(0, 2, 1, 3, 4).contiguous()  # [B,C,D,h2,w2]

        if self.enable_depth_pos:
            if D > self.max_depth:
                # interpolate positional embeddings if needed
                # (simple linear interpolation on depth index)
                pos = self.depth_pos.weight  # [max_depth, C]
                pos = pos.transpose(0, 1).unsqueeze(0)  # [1,C,max_depth]
                pos = F.interpolate(pos, size=D, mode="linear", align_corners=False)
                pos = pos.squeeze(0).transpose(0, 1)  # [D,C]
            else:
                idx = torch.arange(D, device=feats.device)
                pos = self.depth_pos(idx)  # [D,C]
            feats = feats + pos.transpose(0, 1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [B,C,D,1,1]

        if self.enable_depth_mixer:
            feats = feats + self.depth_mixer(feats)

        for ad in self.adapters:
            feats = ad(feats)

        return feats
