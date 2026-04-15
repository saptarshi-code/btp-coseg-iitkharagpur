"""
CoSegNet — Pure PyTorch Reimplementation
Unsupervised 3D Point Cloud Co-Segmentation

Based on: Yang et al., ICCV 2021
"Unsupervised Point Cloud Object Co-Segmentation by
 Co-Contrastive Learning and Mutual Attention Sampling"

Reimplemented without CUDA extensions (knn_cuda, pointnet2, SoftProjection).
Tested on Google Colab with PyTorch 2.x + CUDA 12.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# KNN helper  (replaces knn_cuda)
# ─────────────────────────────────────────────────────────────────────────────

def knn_graph(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute k-nearest neighbour indices for each point.
    Args:
        x : (B, D, N)  point features
        k : number of neighbours
    Returns:
        idx : (B, N, k) indices of k nearest neighbours
    """
    xt   = x.permute(0, 2, 1)           # (B, N, D)
    dist = torch.cdist(xt, xt)          # (B, N, N)  all-pairs L2
    # exclude self (distance 0) → take k+1 and drop column 0
    idx  = dist.topk(k + 1, dim=-1, largest=False).indices[:, :, 1:]
    return idx                          # (B, N, k)


# ─────────────────────────────────────────────────────────────────────────────
# EdgeConv
# ─────────────────────────────────────────────────────────────────────────────

def get_graph_feature(x: torch.Tensor, k: int = 20) -> torch.Tensor:
    """
    Compute edge features for each point's k-neighbourhood.
    Edge feature = [neighbour - centre | centre]  (2D channels)
    """
    B, D, N = x.shape
    idx      = knn_graph(x, k)                              # (B, N, k)
    base     = torch.arange(B, device=x.device).view(-1, 1, 1) * N
    flat     = (idx + base).view(-1)
    xt       = x.permute(0, 2, 1).contiguous()             # (B, N, D)
    nbr      = xt.view(B * N, -1)[flat].view(B, N, k, D)
    ctr      = xt.view(B, N, 1, D).expand(B, N, k, D)
    edge     = torch.cat([nbr - ctr, ctr], dim=3)          # (B, N, k, 2D)
    return edge.permute(0, 3, 1, 2).contiguous()           # (B, 2D, N, k)


# ─────────────────────────────────────────────────────────────────────────────
# DGCNN Encoder
# ─────────────────────────────────────────────────────────────────────────────

class DGCNN(nn.Module):
    """
    Dynamic Graph CNN encoder.
    4 × EdgeConv layers, outputs concatenated → 512-dim per-point features.

    Input  : (B, 3, N)
    Output : (B, emb_dim, N)
    """

    def __init__(self, k: int = 20, emb_dim: int = 512, num_parts: int = 2):
        super().__init__()
        self.k = k

        self.ec1 = nn.Sequential(
            nn.Conv2d(6,   64,  1, bias=False), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2))
        self.ec2 = nn.Sequential(
            nn.Conv2d(128, 64,  1, bias=False), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2))
        self.ec3 = nn.Sequential(
            nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.ec4 = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))

        self.proj = nn.Sequential(
            nn.Conv1d(512, emb_dim, 1, bias=False),
            nn.BatchNorm1d(emb_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
        )

        # Global descriptor (max + avg pooling → 2 × emb_dim → 512)
        self.global_head = nn.Sequential(
            nn.Linear(emb_dim * 2, 512, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(0.4),
            nn.Linear(512,         256, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.4),
        )

        # Per-point entropy head (used for L_ent)
        self.entropy_head = nn.Conv1d(emb_dim, num_parts, 1, bias=True)

    def forward(self, x: torch.Tensor):
        """
        x : (B, 3, N)
        Returns:
            pf        : (B, emb_dim, N)  per-point features
            global_f  : (B, 256)          global descriptor
            part_prob : (B, N, num_parts) per-point part probabilities
        """
        x1 = self.ec1(get_graph_feature(x,  self.k)).max(dim=-1)[0]   # (B,  64, N)
        x2 = self.ec2(get_graph_feature(x1, self.k)).max(dim=-1)[0]   # (B,  64, N)
        x3 = self.ec3(get_graph_feature(x2, self.k)).max(dim=-1)[0]   # (B, 128, N)
        x4 = self.ec4(get_graph_feature(x3, self.k)).max(dim=-1)[0]   # (B, 256, N)

        pf = self.proj(torch.cat([x1, x2, x3, x4], dim=1))            # (B, emb_dim, N)

        g_max = F.adaptive_max_pool1d(pf, 1).squeeze(-1)
        g_avg = F.adaptive_avg_pool1d(pf, 1).squeeze(-1)
        global_f  = self.global_head(torch.cat([g_max, g_avg], dim=1))
        part_prob = torch.softmax(self.entropy_head(pf), dim=1)         # (B, num_parts, N)
        part_prob = part_prob.permute(0, 2, 1)                          # (B, N, num_parts)

        return pf, global_f, part_prob


# ─────────────────────────────────────────────────────────────────────────────
# Mutual Attention Module  (MAS)
# ─────────────────────────────────────────────────────────────────────────────

class MutualAttention(nn.Module):
    """
    Cross-shape attention: each shape attends to a batch-shifted shape.
    Batch shift: shape[b] attends to shape[(b-1) mod B].

    Replaces NONLocalBlock1D_mutual from the original paper.
    """

    def __init__(self, in_dim: int = 128):
        super().__init__()
        d = max(in_dim // 2, 32)
        self.theta = nn.Conv1d(in_dim, d, 1, bias=False)   # query
        self.phi   = nn.Conv1d(in_dim, d, 1, bias=False)   # key
        self.g     = nn.Conv1d(in_dim, d, 1, bias=False)   # value
        self.W     = nn.Conv1d(d * 2,  in_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h : (B, C, N)
        Returns: y : (B, C, N)  attended features + residual
        """
        B, C, N = h.shape
        # Batch-shift: shape i uses shape (i-1 mod B) as cross reference
        shift = torch.cat([h[-1:], h[:-1]], dim=0)    # (B, C, N)

        Q = self.theta(h)      # (B, d, N)
        K = self.phi(shift)    # (B, d, N)
        V = self.g(shift)      # (B, d, N)

        # Attention map: (B, N, N)
        A = torch.softmax(Q.permute(0, 2, 1).bmm(K), dim=-1)   # (B, N, N)

        # Per-point attention weight scalar c_i = mean_j(A_ij)
        c = A.mean(dim=-1, keepdim=True)              # (B, N, 1) → (B, 1, N)
        c = c.permute(0, 2, 1)

        fg_feat = c       * V                          # (B, d, N)
        bg_feat = (1 - c) * V                          # (B, d, N)

        y = self.W(torch.cat([fg_feat, bg_feat], dim=1))  # (B, C, N)
        return y + h                                       # residual


# ─────────────────────────────────────────────────────────────────────────────
# Point Sampler  (standard)
# ─────────────────────────────────────────────────────────────────────────────

class PointSampler(nn.Module):
    """
    Learns to score each point and selects the top-k via differentiable softmax.
    Replaces SampleNet (which requires ChamferDistance / SoftProjection CUDA ext).

    Input  : (B, emb_dim, N)
    Output : (B, emb_dim, num_out)  — sampled feature subset
    """

    def __init__(self, in_dim: int = 512, num_out: int = 256):
        super().__init__()
        self.num_out = num_out
        self.score = nn.Sequential(
            nn.Conv1d(in_dim, 256, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256,    128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128,      1, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        x : (B, C, N)
        Returns sampled : (B, C, num_out)
        """
        scores  = self.score(x).squeeze(1)              # (B, N)
        weights = torch.softmax(scores, dim=-1)          # differentiable
        idx     = weights.topk(self.num_out, dim=-1).indices   # (B, num_out)
        idx_e   = idx.unsqueeze(1).expand(-1, x.shape[1], -1)  # (B, C, num_out)
        return torch.gather(x, 2, idx_e)                # (B, C, num_out)


# ─────────────────────────────────────────────────────────────────────────────
# Point Sampler with MAS
# ─────────────────────────────────────────────────────────────────────────────

class PointSamplerMAS(nn.Module):
    """
    Point sampler with Mutual Attention Sampling embedded after the 128-dim bottleneck.
    """

    def __init__(self, in_dim: int = 512, num_out: int = 256):
        super().__init__()
        self.num_out = num_out
        self.enc = nn.Sequential(
            nn.Conv1d(in_dim, 256, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256,    128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.mas   = MutualAttention(in_dim=128)
        self.score = nn.Conv1d(128, 1, 1)

    def forward(self, x: torch.Tensor):
        h       = self.enc(x)                            # (B, 128, N)
        h       = self.mas(h)                            # (B, 128, N)
        scores  = self.score(h).squeeze(1)               # (B, N)
        weights = torch.softmax(scores, dim=-1)
        idx     = weights.topk(self.num_out, dim=-1).indices
        idx_e   = idx.unsqueeze(1).expand(-1, x.shape[1], -1)
        return torch.gather(x, 2, idx_e)                 # (B, C, num_out)


# ─────────────────────────────────────────────────────────────────────────────
# CoSegNet  (without MAS)
# ─────────────────────────────────────────────────────────────────────────────

class CoSegNet(nn.Module):
    """
    Full co-segmentation network:
        DGCNN encoder → FG sampler + BG sampler → losses

    Evaluation:
        Prototype distance — assign FG if closer to mean(fg_feats) than mean(bg_feats).
        Fully unsupervised: no labels used at any stage.
    """

    def __init__(self, n_fg: int = 256, n_bg: int = 256, emb_dim: int = 512,
                 dgcnn_k: int = 20, num_parts: int = 2):
        super().__init__()
        self.encoder    = DGCNN(k=dgcnn_k, emb_dim=emb_dim, num_parts=num_parts)
        self.fg_sampler = PointSampler(in_dim=emb_dim, num_out=n_fg)
        self.bg_sampler = PointSampler(in_dim=emb_dim, num_out=n_bg)

    def forward(self, xyz: torch.Tensor):
        """
        xyz : (B, 3, N)
        Returns:
            fg_feats   : (B, emb_dim, n_fg)
            bg_feats   : (B, emb_dim, n_bg)
            fg_obj     : (B, 256)    global FG descriptor
            bg_obj     : (B, 256)    global BG descriptor
            part_prob  : (B, N, 2)
            per_pt     : (B, emb_dim, N)
        """
        pf, g, part_prob = self.encoder(xyz)
        fg_feats = self.fg_sampler(pf)
        bg_feats = self.bg_sampler(pf)
        fg_obj   = fg_feats.mean(dim=-1)   # (B, emb_dim)
        bg_obj   = bg_feats.mean(dim=-1)
        return fg_feats, bg_feats, fg_obj, bg_obj, part_prob, pf

    @torch.no_grad()
    def predict(self, xyz: torch.Tensor):
        """
        Prototype-distance prediction (unsupervised, evaluation only).
        Returns binary labels (B, N): 1 = foreground.
        """
        pf, _, _ = self.encoder(xyz)
        fg_feats = self.fg_sampler(pf)   # (B, C, n_fg)
        bg_feats = self.bg_sampler(pf)   # (B, C, n_bg)
        mu_fg    = fg_feats.mean(dim=-1, keepdim=True)  # (B, C, 1)
        mu_bg    = bg_feats.mean(dim=-1, keepdim=True)
        pts      = pf.permute(0, 2, 1)   # (B, N, C)
        d_fg     = ((pts - mu_fg.permute(0, 2, 1)) ** 2).sum(-1)  # (B, N)
        d_bg     = ((pts - mu_bg.permute(0, 2, 1)) ** 2).sum(-1)
        return (d_fg < d_bg).long()


# ─────────────────────────────────────────────────────────────────────────────
# CoSegNetMAS  (with Mutual Attention Sampling)
# ─────────────────────────────────────────────────────────────────────────────

class CoSegNetMAS(CoSegNet):
    """Same as CoSegNet but with MAS-equipped samplers."""

    def __init__(self, n_fg: int = 256, n_bg: int = 256, emb_dim: int = 512,
                 dgcnn_k: int = 20, num_parts: int = 2):
        super().__init__(n_fg, n_bg, emb_dim, dgcnn_k, num_parts)
        self.fg_sampler = PointSamplerMAS(in_dim=emb_dim, num_out=n_fg)
        self.bg_sampler = PointSamplerMAS(in_dim=emb_dim, num_out=n_bg)
