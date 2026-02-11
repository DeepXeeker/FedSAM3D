from __future__ import annotations
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_sam3d import SAM2DWrapper
from .apg import AutoPromptGenerator
from .decoder import PromptConditioned3DDecoder

class FedSAM3D(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        mcfg = cfg["model"]
        self.num_classes = int(mcfg["num_classes"])
        self.use_apg = bool(mcfg.get("use_apg", True))

        self.backbone = SAM2DWrapper(
            sam_variant=mcfg["sam_variant"],
            sam_checkpoint=cfg["paths"]["sam_checkpoint"] or cfg["paths"].get("sam_checkpoint", ""),
            adapter_rank=int(mcfg["adapter_rank"]),
            adapter_layers=int(mcfg["adapter_layers"]),
            depth_mixer_kernel=int(mcfg["depth_mixer_kernel"]),
            enable_depth_pos=bool(mcfg.get("enable_depth_pos", True)),
            enable_depth_mixer=bool(mcfg.get("enable_depth_mixer", True)),
            freeze_sam=bool(mcfg["freeze_sam"]),
        )
        feat_dim = self.backbone.embed_dim

        q_dim = int(mcfg["apg_dim"])
        if self.use_apg:
            self.apg = AutoPromptGenerator(in_ch=feat_dim, num_classes=self.num_classes, q_dim=q_dim)
        else:
            # Fixed learnable class tokens (ablation: no APG)
            self.class_tokens = nn.Parameter(torch.zeros(self.num_classes, q_dim))
            nn.init.normal_(self.class_tokens, std=0.02)

        dcfg = mcfg["decoder"]
        self.decoder = PromptConditioned3DDecoder(
            feat_dim=feat_dim,
            q_dim=q_dim,
            num_classes=self.num_classes,
            num_layers=int(dcfg["num_layers"]),
            num_heads=int(dcfg["num_heads"]),
            mlp_ratio=float(dcfg["mlp_ratio"]),
            dropout=float(dcfg["dropout"]),
        )

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
          vol: [B,1,D,H,W]

        Returns:
          probs: [B,C,D,H,W] with sigmoid probabilities per organ.
        """
        B, _, D, H, W = vol.shape
        feats = self.backbone(vol)  # [B, Cfeat, D', H', W']
        if self.use_apg:
            queries = self.apg(feats)  # [B,C,q_dim]
        else:
            queries = self.class_tokens.unsqueeze(0).expand(B, -1, -1)

        logits = self.decoder(feats, queries, out_shape=(D, H, W))  # [B,C,D,H,W]
        probs = torch.sigmoid(logits)
        return probs

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return a state_dict containing only trainable parameters (parameter-efficient FL payload)."""
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if self._is_trainable_param(k)}

    def load_trainable_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        current = self.state_dict()
        for k, v in sd.items():
            if k in current:
                current[k].copy_(v.to(current[k].device))
        self.load_state_dict(current, strict=False)

    def _is_trainable_param(self, name: str) -> bool:
        p = dict(self.named_parameters()).get(name, None)
        return (p is not None) and p.requires_grad
