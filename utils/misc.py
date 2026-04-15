"""utils/misc.py — Shared utility functions."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(student, teacher, optimizer, epoch, iou, f1, path):
    ckpt = {
        'epoch'          : epoch,
        'student_state'  : student.state_dict(),
        'teacher_state'  : teacher.state_dict() if teacher is not None else None,
        'optimizer_state': optimizer.state_dict(),
        'iou'            : iou,
        'f1'             : f1,
    }
    torch.save(ckpt, path)
    print(f'  Checkpoint saved → {path}  (IoU={iou:.4f}, F1={f1:.4f})')


def load_checkpoint(path, student, teacher=None, optimizer=None):
    ckpt = torch.load(path, map_location='cpu')
    student.load_state_dict(ckpt['student_state'])
    if teacher is not None and ckpt['teacher_state'] is not None:
        teacher.load_state_dict(ckpt['teacher_state'])
    if optimizer is not None and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    print(f'Loaded checkpoint from {path} (epoch={ckpt["epoch"]}, IoU={ckpt["iou"]:.4f})')
    return ckpt['epoch'], ckpt['iou'], ckpt['f1']


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count
