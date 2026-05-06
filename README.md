# PFT3A: Prior-Free Tabular Test-Time Adaptation on TableShift

Course project for *Advanced Machine Learning*. This repository contains a
reproduction and empirical study of **PFT3A** (Prior-Free Tabular Test-Time
Adaptation) on the **TableShift** distribution-shift benchmark, together with
all training, hyper-parameter search, and adaptation pipelines used to
produce the results reported in the accompanying write-up.

---

## 1. Overview

The pipeline has three stages:

1. **Source-model training with hyper-parameter optimisation.**
   `tune_csv.py` runs a Ray Tune sweep (HyperOpt TPE search algorithm + ASHA
   early-stopping scheduler) over the search space defined in
   `tableshift/configs/hparams.py`, optimising **validation balanced
   accuracy** on the in-distribution split. The best configuration is then
   refit and the checkpoint is written to
   `models/<experiment>/<arch>/checkpoint.pt`, together with a
   `metrics.json` log.
2. **Prior-Free test-time adaptation.** `TTA_main_csv.py` loads the source
   checkpoint and adapts *all* trainable parameters of the network on the
   unlabelled OOD-test stream using the PFT3A objective (online
   entropy-regularised KL minimisation against an estimated target prior;
   see `tableshift/PFT3A_src/PFT3A.py`). Adapted predictions and metrics are
   written to `metrics_tta.json`.
3. **Layer-Norm-only TTA variant.** `TTA_main_csv_ln.py` is a Tent-style
   restriction of stage 2: only the affine parameters of LayerNorm /
   BatchNorm layers are updated; the rest of the network is frozen. Outputs
   are written to `metrics_tta_ln.json`. We use this variant as a
   stability-vs-strength ablation.

Throughout, the OOD-test split is **never** seen by training, model
selection, or hyper-parameter search, in keeping with the strict OOD
protocol.

---

## 2. Repository layout

```
.
├── environment.yml                # Conda env spec (Python 3.8)
├── requirements.txt               # Pip dependencies
├── tune_csv.py / tune_csv.sh      # Stage 1: HPO + source-model training
├── TTA_main_csv.py / TTA_main.sh  # Stage 2: full-parameter PFT3A TTA
├── TTA_main_csv_ln.py             # Stage 3: LayerNorm-only TTA variant
├── TTA_main_ln.sh
├── tableshift/                    # Library code
│   ├── PFT3A_src/                 # PFT3A.py (full) + PFT3A_ln.py (LN-only)
│   ├── core/                      # CSVTabularDataset, splitters, ...
│   ├── configs/                   # benchmark_configs.py, hparams.py
│   ├── models/                    # MLP / FT-Transformer / TabTransformer / ...
│   ├── datasets/                  # raw dataset adapters
│   └── third_party/               # SAINT, NODE, ...
└── models/                        # All metric logs from our runs
    └── <experiment>/<arch>/
        ├── metrics.json           # source-trained, val + id_test + ood_test
        ├── metrics_tta.json       # after full-parameter PFT3A TTA
        └── metrics_tta_ln.json    # (acsunemployment/fttrans only) LN-only TTA
```

`<experiment>` ranges over the six TableShift tasks
`{acsfoodstamps, acsunemployment, assistments, brfss_diabetes, nhanes_lead, physionet}`.
`<arch>` ranges over `{mlp, tabtrans, fttrans}`, corresponding to the
`mlp`, `tabtransformer`, and `ft_transformer` backbones.

> **Note on shipped artefacts.** Trained `checkpoint.pt` files are
> intentionally excluded from this submission to keep the archive small;
> only the JSON metric logs are included so that all numbers in the report
> are exactly reproducible from this repository. Re-running the commands in
> §4 regenerates the checkpoints locally.

---

## 3. Environment

The code targets Python 3.8 with PyTorch and Ray. Two equivalent setup
options are provided:

```bash
# Option A — Conda
conda env create -f environment.yml
conda activate pft3a

# Option B — Pip
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-enabled GPU is required to reproduce the deep-tabular runs in
practical time. CPU-only execution is possible but slow.

---

## 4. Data

This project consumes the pre-extracted CSV splits of the six TableShift
tasks, organised as `{experiment}_X{split}.csv` / `{experiment}_y{split}.csv`
with `split ∈ {train, val, idtest, ood}`, loaded by
`tableshift.core.csv_dataset.CSVTabularDataset`.

Please override `--data_dir` if your CSVs live elsewhere.
TableShift induces **covariate / sub-population shift only** — the label
sets in the ID and OOD splits are identical (binary classification for all
six tasks).

---

## 5. How to reproduce

All commands below are run from the repository root.

### 5.1 Source-model training and HPO

```bash
# (Edit tune_csv.sh to choose <experiment>, <model>, num_samples, ...)
bash tune_csv.sh
```

`tune_csv.sh` is a thin wrapper around:

```bash
python tune_csv.py \
    --experiment brfss_diabetes \
    --model ft_transformer \
    --num_samples 16 \
    --max_epochs 20 \
    --gpus_per_trial 0.5 \
    --batch_size 4096
```

* `--num_samples`: total number of HPO trials drawn by HyperOpt.
* `--max_epochs`: per-trial training budget; ASHA prunes unpromising trials
  early.
* `--batch_size`: when omitted, batch size is also part of the search space
  (`{128, 256, 512, 1024}`); fixing it pins the value across all trials.

After the sweep, the best configuration is refit and the resulting
checkpoint and metric log are written to
`models/<experiment>/<arch>/{checkpoint.pt, metrics.json}`.

### 5.2 Full-parameter PFT3A test-time adaptation

```bash
bash TTA_main.sh
# or, equivalently:
python TTA_main_csv.py \
    --experiment assistments \
    --model ft_transformer \
    --n_epochs 1 \
    --batch_size 2048
```

Loads `models/<experiment>/<arch>/checkpoint.pt`, runs PFT3A on the
OOD-test stream, and writes
`models/<experiment>/<arch>/metrics_tta.json`.

### 5.3 LayerNorm-only TTA variant

```bash
bash TTA_main_ln.sh
# or:
python TTA_main_csv_ln.py \
    --experiment acsunemployment \
    --model ft_transformer \
    --n_epochs 3 \
    --batch_size 1024
```

Same as 5.2 but only LayerNorm / BatchNorm affine parameters receive
gradients. Outputs `metrics_tta_ln.json`. We report this variant on
`acsunemployment + ft_transformer` as a representative ablation.

---

## 6. The `models/` directory

Each `metrics.json` (Stage 1) records:

* The HPO winning configuration (`best_config`).
* The training schedule actually used (`n_epochs`, `batch_size`, `n_train`).
* Per-split metrics on `validation`, `id_test`, `ood_validation`, and
  `ood_test`: accuracy, balanced accuracy, and macro-F1.

Each `metrics_tta.json` / `metrics_tta_ln.json` (Stages 2–3) records the
post-adaptation OOD-test metrics, the TTA configuration (`n_epochs`,
`batch_size`), and the fraction of OOD samples actually used by the
PFT3A reliability gate (`tta.fraction = total_used / total_seen`).

These JSON files are the single source of truth for every number in the
written report.

---

## 7. Acknowledgements

This project builds on:

* **TableShift** — Gardner et al., *TableShift: A Benchmark for
  Distribution Shift in Tabular Data*, NeurIPS 2023.
  ([github.com/mlfoundations/tableshift](https://github.com/mlfoundations/tableshift))
* **PFT3A** — *Prior-Free Tabular Test-Time Adaptation*. The
  implementation under `tableshift/PFT3A_src/` follows the algorithmic
  description from the paper.
* **Ray Tune** and **HyperOpt** for hyper-parameter optimisation.
* **Tent** (Wang et al., ICLR 2021) — design inspiration for the
  LayerNorm-only adaptation variant in `TTA_main_csv_ln.py`.
