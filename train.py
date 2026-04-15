"""
train.py — Training script for CoSegNet / CoSegNetMAS
with optional EMA Teacher-Student framework.

Run:
    python train.py --config configs/chair_ema.yaml
    python train.py --config configs/chair_mas_ema.yaml
"""

import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score, f1_score

from datasets.shapenet_part import ShapeNetPart
from models.coseg_net import CoSegNet, CoSegNetMAS
from models.losses import contrastive_loss, repulsion_loss, \
    spatial_consistency_loss, entropy_loss, ema_consistency_loss
from utils.misc import set_seed, save_checkpoint, AverageMeter
from utils.config import load_config


# ── Argument parsing ──────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/chair_ema.yaml',
                   help='Path to YAML config file')
    p.add_argument('--resume', default=None,
                   help='Path to checkpoint to resume from')
    return p.parse_args()


# ── EMA update ───────────────────────────────────────────────────────────────

def ema_update(teacher: torch.nn.Module, student: torch.nn.Module, alpha: float):
    """θ_teacher = α·θ_teacher + (1-α)·θ_student"""
    with torch.no_grad():
        for pt, ps in zip(teacher.parameters(), student.parameters()):
            pt.data.mul_(alpha).add_(ps.data * (1.0 - alpha))


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device,
             n_fg: int = 256):
    """Prototype-distance unsupervised evaluation. Returns IoU, F1."""
    model.eval()
    all_pred, all_true = [], []

    for pc, label in loader:
        pc = pc.to(device).permute(0, 2, 1).float()   # (B, 3, N)

        fg_feats, bg_feats, *_ = model(pc)
        mu_fg = fg_feats.mean(dim=-1, keepdim=True)   # (B, C, 1)
        mu_bg = bg_feats.mean(dim=-1, keepdim=True)

        pf = model.encoder(pc)[0]                      # (B, C, N)
        pts = pf.permute(0, 2, 1)                      # (B, N, C)
        d_fg = ((pts - mu_fg.permute(0, 2, 1)) ** 2).sum(-1)
        d_bg = ((pts - mu_bg.permute(0, 2, 1)) ** 2).sum(-1)
        pred = (d_fg < d_bg).long().cpu().numpy()

        all_pred.append(pred.reshape(-1))
        all_true.append(label.numpy().reshape(-1))

    yp = np.concatenate(all_pred)
    yt = np.concatenate(all_true)
    iou = jaccard_score(yt, yp, zero_division=0)
    f1  = f1_score(yt, yp, zero_division=0)
    return iou, f1


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(cfg.seed)
    print(f'Device: {device} | Config: {cfg}')

    # ── Datasets ──
    train_ds = ShapeNetPart(cfg.data_root, cfg.obj_class, 'train',
                            cfg.num_points, cfg.train_ratio)
    test_ds  = ShapeNetPart(cfg.data_root, cfg.obj_class, 'test',
                            cfg.num_points, cfg.train_ratio)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, drop_last=True, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=1,
                              shuffle=False, num_workers=2)

    # ── Model ──
    ModelClass = CoSegNetMAS if cfg.use_mas else CoSegNet
    student = ModelClass(n_fg=cfg.n_fg, n_bg=cfg.n_bg, emb_dim=cfg.emb_dim,
                         dgcnn_k=cfg.dgcnn_k).to(device)

    teacher = None
    if cfg.use_ema:
        teacher = copy.deepcopy(student)
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f'EMA Teacher-Student enabled (α={cfg.ema_alpha})')

    opt   = optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)

    best_iou, best_f1 = 0.0, 0.0
    os.makedirs(cfg.save_dir, exist_ok=True)

    for epoch in range(cfg.n_epochs):
        student.train()
        if teacher is not None:
            teacher.eval()

        meter = AverageMeter()
        t0 = time.time()
        ema_active = cfg.use_ema and (epoch >= cfg.ema_warmup)

        for pc, _ in train_loader:
            pc = pc.to(device).permute(0, 2, 1).float()   # (B, 3, N)

            # Student forward
            fg_f, bg_f, fg_obj, bg_obj, p_s, per_pt = student(pc)
            xyz = pc.permute(0, 2, 1)                       # (B, N, 3)

            # Teacher forward (no grad)
            p_t = None
            if teacher is not None:
                with torch.no_grad():
                    *_, p_t, _ = teacher(pc)

            # Losses
            l_con = contrastive_loss(fg_obj, bg_obj)
            l_rep = repulsion_loss(fg_obj, bg_obj)
            l_sp  = spatial_consistency_loss(per_pt, xyz, k=cfg.k_spatial)
            l_ent = entropy_loss(p_s)
            l_ema = torch.tensor(0.0, device=device)
            if ema_active and p_t is not None:
                l_ema = ema_consistency_loss(p_t, p_s,
                                             threshold=cfg.get('conf_threshold', None))

            loss = (l_con
                    + cfg.lambda_rep     * l_rep
                    + cfg.lambda_spatial * l_sp
                    + cfg.lambda_entropy * l_ent
                    + cfg.lambda_ema     * l_ema)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            opt.step()

            if teacher is not None:
                ema_update(teacher, student, cfg.ema_alpha)

            meter.update(loss.item())

        sched.step()

        # Evaluate on teacher if EMA active, else student
        eval_model = teacher if (ema_active and teacher is not None) else student
        iou, f1 = evaluate(eval_model, test_loader, device, cfg.n_fg)

        mark = ''
        if f1 > best_f1:
            best_iou, best_f1 = iou, f1
            save_checkpoint(student, teacher, opt, epoch, iou, f1,
                            os.path.join(cfg.save_dir, 'best.pt'))
            mark = ' ← Best'

        print(f'Epoch {epoch:03d} | loss {meter.avg:.4f} | '
              f'IoU {iou:.4f} | F1 {f1:.4f} | '
              f'{time.time()-t0:.1f}s{mark}')

    print(f'\nTraining complete. Best IoU={best_iou:.4f} | Best F1={best_f1:.4f}')
    return best_iou, best_f1


if __name__ == '__main__':
    args = get_args()
    cfg  = load_config(args.config)
    train(cfg)
