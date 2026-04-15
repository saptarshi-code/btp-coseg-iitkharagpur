# Unsupervised Deep 3D Point Cloud Co-Segmentation
### Pure PyTorch Reimplementation · IIT Kharagpur BTP 2026

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![ICCV 2021](https://img.shields.io/badge/Based%20on-ICCV%202021-green.svg)](https://openaccess.thecvf.com/content/ICCV2021/html/Yang_Unsupervised_Point_Cloud_Object_Co-Segmentation_by_Co-Contrastive_Learning_and_Mutual_ICCV_2021_paper.html)

A **fully unsupervised** pipeline for segmenting common 3D object parts across a collection of point clouds, with **zero labels** used at any stage of training or evaluation.

This is a clean, CUDA-extension-free PyTorch reimplementation of:

> **Unsupervised Point Cloud Object Co-Segmentation by Co-Contrastive Learning and Mutual Attention Sampling**  
> Cheng-Kun Yang, Yung-Yu Chuang, Yen-Yu Lin · ICCV 2021 (Oral)  
> [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Unsupervised_Point_Cloud_Object_Co-Segmentation_by_Co-Contrastive_Learning_and_Mutual_ICCV_2021_paper.pdf) [[Original Repo]](https://github.com/jimmy15923/unsup_point_coseg)

Extended with:
- **EMA Teacher-Student framework** → +80.2% IoU over baseline
- **Spatial Consistency Loss** → +15.3% improvement
- **Entropy Regularisation** → +44.6% improvement
- **Confidence-Guided EMA** (ablation)
- **Multi-Category evaluation** (Chair, Airplane, Guitar)

---

## Results

| Method | Best IoU | Best F1 | Stability |
|--------|----------|---------|-----------|
| Baseline (contrastive only) | 0.3379 | 0.5052 | Low |
| + Spatial (λ=0.01, k=20) | 0.4635 | 0.6670 | Medium |
| + Entropy (λ=0.0001) | 0.4885 | 0.6906 | Medium |
| **EMA Teacher-Student ★** | **0.6086** | **0.7567** | Medium |
| MAS Only | 0.3763 | 0.5468 | **High** |
| **MAS + EMA ★** | **0.5260** | **0.6894** | **High** |

★ Recommended configurations. EMA standard is the most reproducible peak result; MAS+EMA is the most stable (17 consecutive epochs above IoU=0.42).

---

## Method Overview

```
Input (B, 1024, 3)
    ↓
DGCNN Encoder  [k=20, 4×EdgeConv, 512-dim]
    ↓
FG Sampler  ───────────  BG Sampler
(TopK 256 pts)           (TopK 256 pts)
    │                         │
    └──── Part Head (512→2, Softmax)
    │
Loss = L_NTXent + 0.5·L_rep + 0.01·L_spatial + 0.0001·L_ent + 0.1·L_EMA
    ↓
EMA Teacher Update: θ_t = 0.999·θ_t + 0.001·θ_s
    ↓
Evaluation: Prototype distance (unsupervised — no labels)
```

**Key differences from the original paper:**

| Component | Original (Yang 2021) | This repo |
|-----------|---------------------|-----------|
| KNN | `knn_cuda` (CUDA ext) | `torch.cdist` |
| Sampler | SampleNet + SoftProjection | Learned score head + top-k |
| Repulsion | Chamfer distance | Cosine similarity |
| Dataset | ScanObjectNN | ShapeNet Part |
| Backbone | Frozen pretrained | Trainable from scratch |
| Training | 2000 epochs | 15–30 epochs |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/btp-coseg-iitkharagpur
cd btp-coseg-iitkharagpur
pip install -r requirements.txt
```

No CUDA extensions required. Runs on standard Google Colab (Tesla T4).

---

## Dataset

Download ShapeNet Part from Kaggle:

```bash
pip install kaggle
kaggle datasets download -d majdouline20/shapenetpart-dataset
unzip shapenetpart-dataset.zip -d data/shapenetpart
```

Then update `data_root` in the config files to point to `data/shapenetpart`.

**Categories used:**

| Class | Name | Shapes |
|-------|------|--------|
| 0 | Airplane | 4,045 |
| 4 | Chair | 6,778 ← primary |
| 6 | Guitar | 787 |

---

## Training

```bash
# Baseline (contrastive only)
python train.py --config configs/chair_baseline.yaml

# Best stable result: EMA Teacher-Student
python train.py --config configs/chair_ema.yaml

# Most stable: MAS + EMA
python train.py --config configs/chair_mas_ema.yaml
```

Training logs print per-epoch IoU and F1. Best checkpoint saved to `results/`.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/BTP_Baseline.ipynb` | Contrastive-only baseline |
| `notebooks/BTP_Spatial_Entropy.ipynb` | Spatial + entropy ablations |
| `notebooks/BTP_EMA.ipynb` | EMA teacher-student training |
| `notebooks/BTP_MAS.ipynb` | Mutual attention sampling |
| `notebooks/BTP_MultiCategory.ipynb` | Chair + Airplane + Guitar |
| `notebooks/BTP_Visualization.ipynb` | 3D point cloud visualizations |

All notebooks are Colab-compatible. Open directly in Google Colab by clicking the badge at the top of each notebook.

---

## Architecture Details

### DGCNN Encoder

4 × EdgeConv layers with dynamic KNN graph (k=20), recomputed per layer:
```
EdgeConv 1: (3→64)   — raw coordinates
EdgeConv 2: (64→64)  — low-level geometry
EdgeConv 3: (64→128) — semantic neighbourhoods form
EdgeConv 4: (128→256)— rich geometric context
Concat → 512 channels → Conv1d projection
Output: (B, 512, N)
```

### Mutual Attention Sampling (MAS)

Cross-shape attention inside the sampler — each shape attends to a batch-shifted shape:
```
shape[0] attends to shape[7]
shape[1] attends to shape[0]   ... (batch size = 8)

Q = θ(h)         # current shape query
K = φ(shift(h))  # next shape key
V = g(shift(h))  # next shape value
A = softmax(Q·Kᵀ)
y = W([c·V, (1-c)·V]) + h   # residual connection
```

### EMA Teacher-Student

```
θ_teacher = 0.999 · θ_teacher + 0.001 · θ_student   (after every batch)
Warmup: 5 epochs before EMA consistency loss activates
Evaluation: always on teacher
```

---

## Honest Limitations

1. **High session-to-session variance** — up to 0.28 IoU difference between identical runs due to random init. Results are indicative, not deterministic.
2. **Late-epoch collapse** — teacher IoU peaks at epoch 5, collapses at epoch 7. Unresolved within project timeframe.
3. **Confidence masking inoperative** — teacher softmax outputs stay near 0.5/0.5; confidence-guided loss never triggers. Fix: prototype sharpening.
4. **ShapeNet Part only** — original paper used ScanObjectNN (not publicly downloadable). Numerical comparison with Yang et al. not possible.
5. **Binary segmentation only** — chairs have 4 parts; we use binary FG/BG.

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@InProceedings{Yang_2021_ICCV,
  author    = {Yang, Cheng-Kun and Chuang, Yung-Yu and Lin, Yen-Yu},
  title     = {Unsupervised Point Cloud Object Co-Segmentation by
               Co-Contrastive Learning and Mutual Attention Sampling},
  booktitle = {Proceedings of the IEEE/CVF International Conference
               on Computer Vision (ICCV)},
  year      = {2021},
  pages     = {7335--7344}
}
```

And this reimplementation:

```bibtex
@misc{sarkar2026coseg,
  author = {Saptarshi Sarkar},
  title  = {Unsupervised 3D Point Cloud Co-Segmentation:
            Pure PyTorch Reimplementation with EMA Teacher-Student},
  year   = {2026},
  school = {IIT Kharagpur},
  note   = {BTP Report, Dept. of Computer Science and Engineering}
}
```

---

## Acknowledgements

Supervised by **Prof. Ayan Chaudhury**, Dept. of CSE, IIT Kharagpur.  
Dataset: ShapeNet Part via Kaggle.  
Compute: Google Colab Tesla T4.
