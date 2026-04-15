"""
Loss functions for CoSegNet.

L_total = L_NTXent + 0.5·L_rep + 0.01·L_spatial + 0.0001·L_ent + 0.1·L_EMA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss


# ── NTXent (co-contrastive) ──────────────────────────────────────────────────

_ntxent_obj = NTXentLoss(temperature=0.07)

def contrastive_loss(fg_obj: torch.Tensor, bg_obj: torch.Tensor) -> torch.Tensor:
    """
    Object-level NTXent: FG object vectors attract each other cross-batch,
    repel BG object vectors.
    fg_obj, bg_obj : (B, D)
    """
    B   = fg_obj.shape[0]
    emb = torch.cat([fg_obj, bg_obj], dim=0)                  # (2B, D)
    lbl = torch.cat([torch.zeros(B), torch.arange(1, B+1)]).long().to(fg_obj.device)
    return _ntxent_obj(emb, lbl)


# ── Repulsion ────────────────────────────────────────────────────────────────

def repulsion_loss(fg_obj: torch.Tensor, bg_obj: torch.Tensor) -> torch.Tensor:
    """
    Minimise cosine similarity between FG and BG object vectors.
    Prevents samplers from selecting the same region.
    """
    return F.cosine_similarity(fg_obj, bg_obj, dim=-1).mean()


# ── Spatial consistency ──────────────────────────────────────────────────────

def spatial_consistency_loss(pf: torch.Tensor, xyz: torch.Tensor,
                              k: int = 10) -> torch.Tensor:
    """
    Penalise L2 distance between features of geometrically close points.
    Neighbourhood is computed in XYZ space (not feature space).

    pf  : (B, D, N)  per-point features
    xyz : (B, N, 3)  point coordinates
    """
    B, D, N = pf.shape
    dist    = torch.cdist(xyz, xyz)                       # (B, N, N)
    idx     = dist.topk(k + 1, dim=-1, largest=False).indices[:, :, 1:]  # (B, N, k)
    feats   = pf.permute(0, 2, 1)                         # (B, N, D)
    flat    = idx.reshape(B, -1)
    flat_e  = flat.unsqueeze(-1).expand(B, N * k, D)
    nbr_f   = torch.gather(feats, 1, flat_e).view(B, N, k, D)
    fi      = feats.unsqueeze(2).expand(B, N, k, D)
    return ((fi - nbr_f) ** 2).sum(-1).mean()


# ── Entropy regularisation ────────────────────────────────────────────────────

def entropy_loss(part_probs: torch.Tensor) -> torch.Tensor:
    """
    Minimise per-point entropy → push each point to confident 0/1 assignment.
    part_probs : (B, N, num_parts)
    """
    p = part_probs.clamp(min=1e-8)
    return -(p * torch.log(p)).sum(dim=-1).mean()


# ── EMA consistency (KL) ──────────────────────────────────────────────────────

def ema_consistency_loss(p_teacher: torch.Tensor, p_student: torch.Tensor,
                         threshold: float = None) -> torch.Tensor:
    """
    KL(p_teacher || p_student).
    teacher is always detached — no gradients flow through it.

    p_teacher, p_student : (B, N, num_parts)
    threshold            : if set, only compute loss where max(p_teacher) > threshold
                           (confidence-guided variant)
    """
    p_t = p_teacher.detach().clamp(min=1e-8)
    p_s = p_student.clamp(min=1e-8)

    if threshold is not None:
        mask = (p_t.max(dim=-1)[0] > threshold).float()   # (B, N)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=p_t.device)
        kl  = (p_t * (p_t.log() - p_s.log())).sum(dim=-1)  # (B, N)
        return (kl * mask).sum() / mask.sum()

    kl = (p_t * (p_t.log() - p_s.log())).sum(dim=-1)
    return kl.mean()


# ── Combined loss helper ─────────────────────────────────────────────────────

def total_loss(fg_feats, bg_feats, fg_obj, bg_obj, per_pt, xyz,
               p_teacher=None, p_student=None,
               lambda_rep=0.5, lambda_sp=0.01, k_sp=10,
               lambda_ent=0.0001, lambda_ema=0.1,
               ema_active=True, conf_threshold=None):
    """
    Full combined loss.
    Returns scalar loss + dict of individual components for logging.
    """
    l_con = contrastive_loss(fg_obj, bg_obj)
    l_rep = repulsion_loss(fg_obj, bg_obj)
    l_sp  = spatial_consistency_loss(per_pt, xyz, k=k_sp)

    # Part probabilities from per_pt via entropy head (passed as p_student)
    l_ent = entropy_loss(p_student) if p_student is not None else torch.tensor(0.0)

    l_ema = torch.tensor(0.0, device=fg_obj.device)
    if ema_active and p_teacher is not None and p_student is not None:
        l_ema = ema_consistency_loss(p_teacher, p_student, threshold=conf_threshold)

    loss = l_con + lambda_rep * l_rep + lambda_sp * l_sp + lambda_ent * l_ent + lambda_ema * l_ema

    return loss, {
        'total': loss.item(), 'contrastive': l_con.item(),
        'repulsion': l_rep.item(), 'spatial': l_sp.item(),
        'entropy': l_ent.item(), 'ema': l_ema.item(),
    }
