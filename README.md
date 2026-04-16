# HeteroGAT-Rank
Operational Runtime Behavior Mining for Open-Source Supply Chain Security

![Python](https://img.shields.io/badge/Python3-3.10-brightgreen.svg) 
![Golang](https://img.shields.io/badge/Go1.22.2-brightblue.svg) 
![CUDA](https://img.shields.io/badge/CUDA12.4-brightred.svg) 

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

    ```
    sudo apt update
    
    sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev

    ```

- Download Pyenv:
    ```
    # for linux
    curl https://pyenv.run | bash

    # add the following lines to ~/.profile and ~/.bashrc
    export PYENV_ROOT="$HOME/.pyenv"
    [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init - bash)"
    
    # add the following to ~/.bashrc:
    eval "$(pyenv virtualenv-init -)"
    
    # reactivate bash files
    source ~/.bashrc
    source ~/.profile
    ```

- Configure Python Version
    ```
    pyenv install 3.10.1
    pyenv global 3.10.1
    # activate environment
    pyenv virtualenv 3.10.1 DDGRL
    pyenv local DDGRL

    # install libraries
    pip install -r requirements-base.txt
    pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
    
    ## (optional) for quick test
    pip3 install pandas==2.2.3 tqdm==4.67.1 sentence-transformers==3.4.1 matplotlib==3.10.3 seaborn==0.13.2 shap==0.47.2 xgboost==3.0.2 accelerate==1.7.0 ace_tools==0.0 torch==2.6.0 torch-geometric==2.6.1 dask==2023.5.0 dask[distributed]==2023.5.0

    ```

- Other packages
    ```
    sudo apt-get install libffi-dev # avoid _ctypes error
    sudo apt-get install libbz2-dev # avoid _bz2 error
    sudo apt install golang-go

    ```

- Resource Required:

    - Graph generation was conducted on a high-memory instance from Vast.ai2 (base-image:cuda-12.8.1-auto), equipped with 104 CPUs (Intel Xeon®
Platinum 8272L), up to 2.91 TB RAM, and 1100 GB disk. 

    - Model training and feature analysis were performed on a GPU-optimized instance (base-image:cuda-12.4-auto), provisioned with 128 CPUs (Intel Xeon® Platinum 8462Y+), 1.97 TB RAM, two H100 SXM GPUs (159 GB total), and 1002 GB disk.

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
python scripts/generate_osptrack_canonical_graphs.py \
  --out artifacts/osp_2k \
  --prefer-pkl --limit-rows 2000 \
  --workers 8

# OSPTrack: stream label_data.csv in chunks
python scripts/generate_qut_canonical_graphs.py \
  --out artifacts/qut_2k \
  --limit-packages 2000 \
  --workers 8
```

### 2) Train a strong baseline GNN (R-GCN)

```bash
python comp/gnn_baselines/train_rgcn.py \
  --graphs artifacts/qut_a artifacts/osp_a \
  --epochs 5 \
  --batch-size 8
```

### 3) Train the edge-mask explainer (PGExplainer-style)

Note: for best explanations, pre-train/freeze the backbone first (Step 2).

```bash
python ranking_explain/train_pgexplainer.py \
  --graphs artifacts/qut_a artifacts/osp_a \
  --epochs 5 \
  --batch-size 4
```

### 4) Produce hunting output (top-K important edges)

Pick any generated `*.graph.pt` and print its top-K edges.

```bash
python ranking_explain/run_hunt.py \
  --graph "artifacts/qut_a/10Cent10-999.0.4.tar.gz.graph.pt" \
  --k 20
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

- Rarity Stats Generation
```
# for osptrack
python scripts/build_benign_rarity_stats.py \
  --graphs artifacts/osp_a \
  --label 0 \
  --out artifacts/stats/benign_rarity_stats_osp_full.json

# for qut data
python scripts/build_benign_rarity_stats.py \
  --graphs artifacts/qut_a \
  --label 0 \
  --out artifacts/stats/benign_rarity_stats_qut_full.json

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

## Experiment Note
```
# download large download in multiple processes
sudo apt update
sudo apt install aria2
# need cookie to enable right file download
aria2c -k 1M -x 8 -s 8 \
--header="Cookie: username-mixing-graphics-agenda-librarian-trycloudflare-com=\"2|1:0|10:1750668312|59:username-mixing-graphics-agenda-librarian-trycloudflare-com|200:eyJ1c2VybmFtZSI6ICJlYmQ5YjgwZDk0YTM0ZjIwYWI0NDM0NjE5MTlhODU3YiIsICJuYW1lIjogIkFub255bW91cyBBbWFsdGhlYSIsICJkaXNwbGF5X25hbWUiOiAiQW5vbnltb3VzIEFtYWx0aGVhIiwgImluaXRpYWxzIjogIkFBIiwgImNvbG9yIjogbnVsbH0=|4a393cbf532b62f3e2d481a6f394a6cec07587f1e95ffc80b8cafe11cb304cb3\"; _xsrf=2|854af5bb|5b2a4314559e61e83185255caa3ccea4|1750668312" \
"https://mixing-graphics-agenda-librarian.trycloudflare.com/files/workspace/DDGRL.zip?_xsrf=2%7C854af5bb%7C5b2a4314559e61e83185255caa3ccea4%7C1750668312"

# download from google drive
pip3 install gdown
gdown --folder "{shared_link}"

```

