"""
ShapeNet Part Dataset — single-category loader for co-segmentation.

Usage:
    ds = ShapeNetPart(data_root='/path/to/shapenetpart', obj_class=4)
    # obj_class: 0=airplane, 4=chair, 6=guitar, 8=lamp, 15=table

Tested with the Kaggle mirror:
    kaggle datasets download -d majdouline20/shapenetpart-dataset
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


# ── Synset lookup ────────────────────────────────────────────────────────────

SYNSET_TO_CLASS = {
    '02691156': 0,   # airplane
    '02773838': 1,   # bag
    '02954340': 2,   # cap
    '02958343': 3,   # car
    '03001627': 4,   # chair      ← used in BTP
    '03261776': 5,   # earphone
    '03467517': 6,   # guitar     ← multi-category extension
    '03624134': 7,   # knife
    '03636649': 8,   # lamp
    '03642806': 9,   # laptop
    '03790512': 10,  # motorbike
    '03797390': 11,  # mug
    '03948459': 12,  # pistol
    '04099429': 13,  # rocket
    '04225987': 14,  # skateboard
    '04379243': 15,  # table
}
CLASS_TO_SYNSET  = {v: k for k, v in SYNSET_TO_CLASS.items()}
CLASS_TO_NAME    = {
    0:'airplane', 1:'bag', 2:'cap', 3:'car', 4:'chair',
    5:'earphone', 6:'guitar', 7:'knife', 8:'lamp', 9:'laptop',
    10:'motorbike', 11:'mug', 12:'pistol', 13:'rocket',
    14:'skateboard', 15:'table',
}


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _load_pts(path: str) -> np.ndarray:
    pts = []
    with open(path) as f:
        for line in f:
            v = line.strip().split()
            if len(v) >= 3:
                pts.append([float(v[0]), float(v[1]), float(v[2])])
    return np.array(pts, dtype='float32')

def _load_seg(path: str) -> np.ndarray:
    with open(path) as f:
        return np.array([int(l.strip()) for l in f if l.strip()], dtype='int64')


# ── Dataset ──────────────────────────────────────────────────────────────────

class ShapeNetPart(Dataset):
    """
    Single-category ShapeNet Part loader.

    Args:
        data_root   : path to the ShapeNet Part root directory
                      (should contain per-synset sub-folders)
        obj_class   : integer class index (see SYNSET_TO_CLASS above)
        partition   : 'train' or 'test'
        num_points  : number of points to sample per shape (default 1024)
        train_ratio : fraction of shapes used for training (default 0.8)
        seed        : random seed for train/test split (default 42)

    Returns per item:
        pc    : (num_points, 3)  float32 tensor — centred, unit sphere
        label : (num_points,)    int64 tensor   — binary 0/1 (bg/fg)
    """

    def __init__(self, data_root: str, obj_class: int = 4,
                 partition: str = 'train', num_points: int = 1024,
                 train_ratio: float = 0.8, seed: int = 42):

        self.num_points = num_points
        synset  = CLASS_TO_SYNSET[obj_class]
        syn_dir = os.path.join(data_root, synset)

        pts_dir = os.path.join(syn_dir, 'points')
        seg_dir = os.path.join(syn_dir, 'points_label')

        # build seg map (handle flat or nested directories)
        seg_map: dict[str, str] = {}
        if os.path.isdir(seg_dir):
            for entry in os.listdir(seg_dir):
                ep = os.path.join(seg_dir, entry)
                if entry.endswith('.seg'):
                    seg_map[entry[:-4]] = ep
                elif os.path.isdir(ep):
                    for fn in os.listdir(ep):
                        if fn.endswith('.seg') and fn[:-4] not in seg_map:
                            seg_map[fn[:-4]] = os.path.join(ep, fn)

        all_ids = sorted([f[:-4] for f in os.listdir(pts_dir) if f.endswith('.pts')])
        valid   = [i for i in all_ids if i in seg_map]

        rng  = np.random.default_rng(seed)
        perm = rng.permutation(len(valid))
        cut  = int(len(valid) * train_ratio)
        sel  = perm[:cut] if partition == 'train' else perm[cut:]

        self.samples = [(os.path.join(pts_dir, valid[i] + '.pts'), seg_map[valid[i]])
                        for i in sel]
        print(f'[ShapeNetPart | {CLASS_TO_NAME[obj_class]} | {partition}] '
              f'{len(self.samples)} shapes loaded.')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        pts_path, seg_path = self.samples[idx]
        pc  = _load_pts(pts_path)
        seg = _load_seg(seg_path)

        # Align lengths (annotation mismatch guard)
        n = min(len(pc), len(seg))
        pc, seg = pc[:n], seg[:n]

        # Random sampling to fixed size
        if n >= self.num_points:
            chosen = np.random.choice(n, self.num_points, replace=False)
        else:
            chosen = np.random.choice(n, self.num_points, replace=True)
        pc, seg = pc[chosen], seg[chosen]

        # Unit sphere normalisation
        pc -= pc.mean(axis=0)
        s   = np.max(np.linalg.norm(pc, axis=1))
        if s > 0:
            pc /= s

        # Binary label: 0 = background (min label), 1 = foreground (all other)
        binary = (seg > seg.min()).astype('int64')

        return torch.from_numpy(pc), torch.from_numpy(binary)
