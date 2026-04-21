# HeteroGAT-Rank

Operational Runtime Behavior Mining for Open-Source Supply Chain Security

Python 
Golang 
CUDA 

## Defined Canonical Node and Edge Types

- Node Tyoes:
  - PKG: package identity (eco, name, version)
  - PROC: process (for QUT) or a placeholder “package-runtime” proc (for label_data)
  - FILE: file/path (or QUT buckets like TMP_DIR, HOME_DIR)
  - NET: network endpoint (IP, DOMAIN, PORT)
  - SYSCALL: syscall name (QUT)
  - CMD: command string (label_data)
- Edge Types:
  - LOAD: PKG → PROC
  - INVOKE: PROC → SYSCALL (QUT)
  - EXEC: PROC → CMD (label_data) and/or PROC → PROC (QUT filetop)
  - READ / WRITE / DELETE: PROC → FILE (label_data file flags; QUT file buckets)
  - CONNECT: PROC → NET (label_data sockets; QUT network if present)
  - DNS_QUERY / RESOLVE: PROC → NET(domain) (label_data DNS)
- Edge Attributes:
  - phase: import vs install 
  - raw_type / dns_types: DNS query types (A, AAAA)
  - bytes/count: for QUT aggregated stats; optional for label_data

## Environment Setting Up:

- General Package Support
- Download Pyenv:
- Configure Python Version
- Other packages
- Resource Required:
  - Graph generation was conducted on a high-memory instance from Vast.ai2 (base-image:cuda-12.8.1-auto), equipped with 104 CPUs (Intel Xeon®

Platinum 8272L), up to 2.91 TB RAM, and 1100 GB disk. 

```
- Model training and feature analysis were performed on a GPU-optimized instance (base-image:cuda-12.4-auto), provisioned with 128 CPUs (Intel Xeon® Platinum 8462Y+), 1.97 TB RAM, two H100 SXM GPUs (159 GB total), and 1002 GB disk.
```

## Quick Test

Below is a minimal end-to-end smoke run on **small-scale canonical graphs**.

### 0) Run unit tests

```bash
python -m unittest discover -s tests -p "test_*.py" -q
```

### 1) Generate small-scale canonical graphs (QUT-DV25 + OSPTrack)

This will generate both:

- `*.events.json`: canonical nodes/edges (human-readable)
- `*.graph.pt`: safe `data_dict` generated from `HeteroData.to_dict()` (for training)

```bash
# QUT-DV25: join multiple processed tables by Package_Name
python scripts/generate_qut_canonical_graphs.py \
  --out artifacts/qut_2k \
  --limit-packages 2000 \
  --workers 8

# OSPTrack: parse label_data.pkl (recommended)
python scripts/generate_osptrack_canonical_graphs.py \
  --out artifacts/osp_2k \
  --prefer-pkl --limit-rows 2000 \
  --workers 8
```

To generate **full-scale** canonical graphs, set the limits to 0:

```bash
# OSPTrack full-scale (0 = all rows)
python scripts/generate_osptrack_canonical_graphs.py \
  --out artifacts/osp_all \
  --prefer-pkl --limit-rows 0 \
  --workers 8

# QUT-DV25 full-scale (0 = all packages)
python scripts/generate_qut_canonical_graphs.py \
  --out artifacts/qut_all \
  --limit-packages 0 \
  --workers 8
```

### 2) Train a strong baseline GNN (R-GCN)

```bash
python comp/gnn_baselines/train_rgcn.py \
  --graphs artifacts/qut_2k artifacts/osp_2k \
  --epochs 5 \
  --batch-size 8 \
  --save-ckpt artifacts/checkpoints/rgcn_smoke.pt
```

### 3) Train the edge-mask explainer (PGExplainer-style)

Note: for best explanations, pre-train/freeze the backbone first (Step 2).

```bash
python ranking_explain/train_pgexplainer.py \
  --graphs artifacts/qut_2k artifacts/osp_2k \
  --epochs 5 \
  --batch-size 4 \
  --backbone-ckpt artifacts/checkpoints/rgcn_smoke.pt \
  --save-ckpt artifacts/checkpoints/pgexp_smoke.pt
```

### 4) Produce hunting output (top-K important edges)

Pick any generated `*.graph.pt` and print its top-K edges.

```bash
python ranking_explain/run_hunt.py \
  --graph "artifacts/qut_2k/10Cent10-999.0.4.tar.gz.graph.pt" \
  --k 20 \
  --backbone-ckpt artifacts/checkpoints/rgcn_smoke.pt \
  --explainer-ckpt artifacts/checkpoints/pgexp_smoke.pt \
  --filter-net --filter-net-ip \
  --filter-system-files --filter-tmp-tempfile --filter-cmd-noise \
  --rerank-tmp-suspicious --demote-load --dedup-dst --max-per-etype 8
```

## Data Split and Statistical Analysis

### 1) Generate Split (Qut/OSP/Cross)

```
python scripts/make_splits.py \
  --qut artifacts/qut_a \
  --osp artifacts/osp_a \
  --out splits_a \
  --seed 42
```

### 2) Graph Type Statistic

```
python scripts/stats_canonical_graphs.py \
  --graphs artifacts/qut_a artifacts/osp_a \
  --out artifacts/stats_a
```

### 3) Train RGCN with split

```
# within QUT
python comp/gnn_baselines/train_rgcn.py \
  --train-list splits_a/qut_train.txt \
  --test-list splits_a/qut_test.txt \
  --epochs 5 --batch-size 8

# within OSP
python comp/gnn_baselines/train_rgcn.py \
  --train-list splits_a/osp_train.txt \
  --test-list splits_a/osp_test.txt \
  --epochs 5 --batch-size 8

# cross-domain
python comp/gnn_baselines/train_rgcn.py \
  --train-list splits_a/cross_train_qut.txt \
  --test-list splits_a/cross_test_osp.txt \
  --epochs 5 --batch-size 8

```

## Baseline Running

### 1) different training data

```
# all data
python comp/gnn_baselines/train_rgcn.py   --train-list splits_a/osp_train.txt   --test-list splits_a/osp_test.txt   --epochs 5 --batch-size 8 --seed 42

# data excluding non-empty (remove all load)
python comp/gnn_baselines/train_rgcn.py   --train-list splits_osp_nonempty/train.txt   --test-list splits_osp_nonempty/test.txt   --epochs 5 --batch-size 8 --seed 42

# reweight all data
python comp/gnn_baselines/train_rgcn.py   --train-list splits_a/osp_train.txt   --test-list splits_a/osp_test.txt   --reweight   --epochs 5 --batch-size 8 --seed 42

```

### 2) Training

```
# train with save backbone
python -m comp.gnn_baselines.train_rgcn --graphs artifacts/osp_a --epochs 5 --save-ckpt artifacts/checkpoints/rgcn_osp_a.pt

# Train + save explainer
python -m ranking_explain.train_pgexplainer --graphs artifacts/osp_a --epochs 5 --backbone-ckpt artifacts/checkpoints/rgcn_osp_a.pt --save-ckpt artifacts/checkpoints/pgexp_osp_a.pt

# hunt with checkpoints
python -m ranking_explain.run_hunt --graph "artifacts/osp_a/..." --k 20 --backbone-ckpt artifacts/checkpoints/rgcn_osp_a.pt --explainer-ckpt artifacts/checkpoints/pgexp_osp_a.pt

```

## Hunting Running

- Rarity Stats Generation (benign document frequency)

```
# OSPTrack benign stats
python scripts/build_benign_rarity_stats.py \
  --graphs artifacts/osp_all \
  --label 0 \
  --out artifacts/stats/benign_rarity_stats_osp_full.json
  # optional (recommended for OSP): --normalize osp

# QUT benign stats
python scripts/build_benign_rarity_stats.py \
  --graphs artifacts/qut_all \
  --label 0 \
  --out artifacts/stats/benign_rarity_stats_qut_full.json

```

### Evaluate (Benchmark metrics + Hunt/Rarity metrics)

#### 1) Benchmark: classification metrics (Acc/Prec/Rec/F1/AUROC/AUPRC + P@K/R@K)

First generate a fixed split (recommended for reproducibility):

```bash
python scripts/make_splits.py \
  --qut artifacts/qut_all \
  --osp artifacts/osp_all \
  --out splits_full \
  --seed 42
```

Train/eval R-GCN on QUT and save the backbone checkpoint:

```bash
python comp/gnn_baselines/train_rgcn.py \
  --graphs artifacts/qut_all \
  --train-list splits_full/qut_train.txt \
  --test-list splits_full/qut_test.txt \
  --epochs 5 \
  --batch-size 8 \
  --ks 10,50,100 \
  --save-ckpt artifacts/checkpoints/rgcn_qut_full.pt
```

Note: `train_rgcn.py` only saves the backbone. To produce explanations you also need a PGExplainer checkpoint:

```bash
python ranking_explain/train_pgexplainer.py \
  --graphs artifacts/qut_all \
  --epochs 5 \
  --batch-size 4 \
  --backbone-ckpt artifacts/checkpoints/rgcn_qut_full.pt \
  --save-ckpt artifacts/checkpoints/pgexp_qut_full.pt
```

#### 2) Hunt/Rarity: anomaly-score ranking metrics (AUROC/AUPRC + P@K/R@K)

This evaluates the graph-level anomaly score computed from top-K explanation edges,
and compares **base** vs **rarity-adjusted** scores.

```bash
python scripts/eval_hunt_rarity.py \
  --test-list splits_full/qut_test.txt \
  --backbone-ckpt artifacts/checkpoints/rgcn_qut_full.pt \
  --explainer-ckpt artifacts/checkpoints/pgexp_qut_full.pt \
  --rarity-stats artifacts/stats/benign_rarity_stats_qut_full.json \
  --k 20 \
  --ks 10,50,100 \
  --filter-system-files --filter-tmp-tempfile --filter-cmd-noise \
  --dedup-dst --max-per-etype 8
```

```

# ================ for OSPTrack dataset ===============

# malicious examples [pypi::oauth-less@1.0, pypi::systemdemon@2.9.graph.pt, npm::rechtspraak.namenlijst@0.9.0, pypi::adv2099m2@1.0.0]
python -m ranking_explain.run_hunt \
  --graph "artifacts/osp_a/pypi::systemdemon@2.9.graph.pt" --k 20 \
  --backbone-ckpt artifacts/checkpoints/rgcn_osp_a.pt \
  --explainer-ckpt artifacts/checkpoints/pgexp_osp_a.pt \
  --filter-net --dedup-dst --max-per-etype 5

# with filter function for explict white list
python -m ranking_explain.run_hunt \
  --graph "artifacts/osp_a/pypi::systemdemon@2.9.graph.pt" --k 20 \
  --backbone-ckpt artifacts/checkpoints/rgcn_osp_a.pt \
  --explainer-ckpt artifacts/checkpoints/pgexp_osp_a.pt \
  --filter-net --filter-domains "pypi.org,files.pythonhosted.org,pythonhosted.org" \
  --dedup-dst --max-per-etype 3

# merge diverse IP addresses with temporal files
python -m ranking_explain.run_hunt \
  --graph "artifacts/osp_y1_3/pypi::adv2099m2@1.0.0.graph.pt" --k 80 \
  --backbone-ckpt artifacts/checkpoints/rgcn_osp_a.pt \
  --explainer-ckpt artifacts/checkpoints/pgexp_osp_a.pt \
  --filter-net --filter-net-ip \
  --filter-system-files --system-file-prefixes "/dev/,dev/,pipe:[,host:[,socket:[,__pycache__/,/{dev=,{dev=" \
  --filter-tmp-tempfile \
  --filter-cmd-noise \
  --rerank-tmp-suspicious \
  --demote-load \
  --dedup-dst --max-per-etype 8

# ================ for QUT_DV25 =================





```

## Distributed Configuration

```
accelerate config
# this machine
# multi-GPU
# 1 machine
# default choice for other options
# 8 GPUs
# default choice for other options
# !no mixed precision --- in order to run sparse matrix

```

