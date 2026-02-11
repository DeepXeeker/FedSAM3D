from __future__ import annotations
import torch
import torch.nn.functional as F

def prompt_diversity_loss(queries: torch.Tensor) -> torch.Tensor:
    """Encourage APG queries to be diverse (avoid collapse).

    We normalize queries and penalize off-diagonal cosine similarity.
    queries: [B, C, q_dim]
    """
    q = F.normalize(queries, dim=-1)
    # cosine similarity: [B,C,C]
    sim = torch.matmul(q, q.transpose(1, 2))
    C = sim.shape[-1]
    off_diag = sim - torch.eye(C, device=sim.device).unsqueeze(0)
    return (off_diag ** 2).mean()
