from __future__ import annotations
from typing import Dict, Any, List, Tuple
import random
import torch
import torch.nn as nn

from .aggregation import normalize_weights, aggregate_state_dicts
from ..models.experts import ExpertBank, ExpertPackage
from ..engine.utils import seed_everything

class FederatedServer:
    def __init__(self, global_model: nn.Module, clients: Dict[str, Any], cfg: Dict[str, Any]):
        self.global_model = global_model
        self.clients = clients  # client_id -> FederatedClient
        self.cfg = cfg
        self.peer_bank = ExpertBank()

    def run(self, device: torch.device, out_dir: str) -> None:
        seed_everything(int(self.cfg["seed"]))
        rounds = int(self.cfg["train"]["max_rounds"])
        participation = float(self.cfg["federated"]["participation"])
        agg_mode = self.cfg["federated"]["aggregation"]

        # Initialize global trainable state
        global_state = self.global_model.trainable_state_dict()

        for r in range(rounds):
            client_ids = list(self.clients.keys())
            m = max(1, int(round(participation * len(client_ids))))
            selected = random.sample(client_ids, k=m)

            # Teacher is current global model (frozen)
            teacher = self._clone_global_model(device)
            teacher.load_trainable_state_dict(global_state)

            updates: Dict[str, Dict[str, torch.Tensor]] = {}
            masses: Dict[str, float] = {}
            new_experts: List[ExpertPackage] = []

            for cid in selected:
                res = self.clients[cid].local_train(
                    global_trainable_state=global_state,
                    teacher_model=teacher,
                    peer_bank=self.peer_bank,
                    device=device,
                )
                updates[cid] = res.trainable_state
                masses[cid] = res.supervision_mass
                new_experts.extend(res.experts)

            # Update peer bank (relay to others next round)
            self.peer_bank.clear()
            for pkg in new_experts:
                self.peer_bank.add(pkg)

            # Aggregation weights
            if agg_mode == "uniform":
                w = normalize_weights({cid: 1.0 for cid in updates})
            elif agg_mode == "fedavg":
                # fall back to supervision mass if dataset size isn't provided
                w = normalize_weights(masses)
            elif agg_mode == "label_coverage":
                # weight by |O_k| * mass proxy
                w = normalize_weights({cid: masses[cid] for cid in updates})
            else:  # supervision_mass (paper)
                w = normalize_weights(masses)

            global_state = aggregate_state_dicts(updates, w)

            # Optionally save checkpoints
            if (r + 1) % 10 == 0 or r == rounds - 1:
                self._save_checkpoint(global_state, out_dir, r + 1)

        # Finalize global model weights
        self.global_model.load_trainable_state_dict(global_state)
        self._save_checkpoint(global_state, out_dir, rounds, final=True)

    def _clone_global_model(self, device: torch.device) -> nn.Module:
        import copy
        m = copy.deepcopy(self.global_model)
        m.to(device)
        for p in m.parameters():
            p.requires_grad = False
        m.eval()
        return m

    def _save_checkpoint(self, trainable_state: Dict[str, torch.Tensor], out_dir: str, round_idx: int, final: bool = False) -> None:
        import os
        os.makedirs(out_dir, exist_ok=True)
        name = "global_final.pt" if final else f"global_round_{round_idx:04d}.pt"
        path = os.path.join(out_dir, name)
        torch.save({"trainable_state": trainable_state, "cfg": self.cfg, "round": round_idx}, path)
