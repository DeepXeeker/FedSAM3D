from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import random
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from ..engine.utils import to_device
from ..losses import masked_dice_bce_loss, global_kd_loss, peer_kd_loss, prompt_diversity_loss
from ..models.experts import ExpertPackage, ExpertBank

@dataclass
class ClientResult:
    client_id: str
    trainable_state: Dict[str, torch.Tensor]
    supervision_mass: float
    experts: List[ExpertPackage]
    val_metrics: Dict[str, float]

class FederatedClient:
    def __init__(self, client_id: str, model: nn.Module, train_loader, val_loader, meta: Dict[str, Any], cfg: Dict[str, Any]):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.meta = meta
        self.cfg = cfg
        self.avail = meta["availability_mask"]

    def local_train(
        self,
        global_trainable_state: Dict[str, torch.Tensor],
        teacher_model: nn.Module,
        peer_bank: ExpertBank,
        device: torch.device,
    ) -> ClientResult:
        # Load global weights
        self.model.load_trainable_state_dict(global_trainable_state)
        self.model.to(device)
        teacher_model.to(device)
        teacher_model.eval()

        tcfg = self.cfg["train"]
        dcfg = self.cfg["distill"]
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
        scaler = GradScaler(enabled=bool(tcfg["amp"]))
        self.model.train()

        local_steps = int(tcfg["local_steps"])
        beta = float(dcfg["beta"])
        gamma = float(dcfg["gamma"])
        tau_g = float(dcfg["tau_gkd"])
        tau_p = float(dcfg["tau_pkd"])
        mk = int(dcfg["mk"])

        supervision_mass = 0.0
        it = iter(self.train_loader)

        for step in range(local_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.train_loader)
                batch = next(it)
            batch = to_device(batch, device)
            vol = batch["image"]
            gt = batch["label"]

            with torch.no_grad():
                teacher_probs = teacher_model(vol)

            with autocast(enabled=bool(tcfg["amp"])):
                student_probs = self.model(vol)
                sup = masked_dice_bce_loss(student_probs, gt, self.avail)

                # GKD on missing classes only
                gkd = global_kd_loss(student_probs, teacher_probs, self.avail, tau=tau_g)

                # PKD: sample m_k missing classes
                missing = [i for i in range(student_probs.shape[1]) if self.avail[i] == 0]
                selected = random.sample(missing, k=min(mk, len(missing))) if (gamma > 0 and len(missing) > 0) else []
                peer_targets = torch.zeros_like(student_probs)
                if len(selected) > 0:
                    from .peer_experts import build_peer_targets
                    peer_targets = build_peer_targets(self.model, vol, peer_bank, selected, device)

                pkd = peer_kd_loss(student_probs, peer_targets, selected, tau_p=tau_p)

                # APG diversity regularization (if APG enabled)
                div = torch.zeros([], device=device)
                if getattr(self.model, "use_apg", False):
                    # recompute queries cheaply by tapping APG
                    feats = self.model.backbone(vol)
                    q = self.model.apg(feats)
                    div = 0.01 * prompt_diversity_loss(q)

                loss = sup + beta * gkd + gamma * pkd + div

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if float(tcfg["grad_clip"]) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(tcfg["grad_clip"]))
            scaler.step(opt)
            scaler.update()

            # Supervision mass proxy: |O_k| * |Omega| per patch (dense supervision for available classes)
            Ck = sum(self.avail)
            vox = gt.shape[-1] * gt.shape[-2] * gt.shape[-3]
            supervision_mass += float(Ck * vox)

        # Extract per-class experts for classes supervised at this client.
        experts = self._extract_experts()

        # Validation
        from ..engine.evaluator import evaluate_model
        val_metrics = evaluate_model(self.model, self.val_loader, self.avail, device, self.cfg["data"]["patch_size"])

        # Return trainable state
        state = self.model.trainable_state_dict()
        return ClientResult(self.client_id, state, supervision_mass, experts, val_metrics)

    def _extract_experts(self) -> List[ExpertPackage]:
        """Extract lightweight per-class expert params from the local model.

        Implementation: save decoder projection rows for each supervised class.
        This keeps experts small while being applicable for peer distillation.
        """
        experts: List[ExpertPackage] = []
        sd = self.model.state_dict()
        # We store full decoder params as expert (simple, robust). If you want smaller experts,
        # slice per-class rows in decoder.q_proj/mask_proj; kept simple here.
        for c in range(1, self.model.num_classes + 1):
            if self.avail[c - 1] == 0:
                continue
            keys = [k for k in sd.keys() if k.startswith("decoder.") or k.startswith("apg.")]
            expert_sd = {k: sd[k].detach().cpu().clone() for k in keys}
            experts.append(ExpertPackage(class_id=c, state_dict=expert_sd, provider=self.client_id))
        return experts
