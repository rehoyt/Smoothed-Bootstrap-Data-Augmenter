# Smoothed-Bootstrap TSTR / Privacy Evaluation

Companion code for **"Tabular Medical Data Augmentation Using Smoothed Bootstrapping"**
(Hoyt R, Jagarapu J, Angelis D). This repository contains the Python used to measure
the **utility** and **privacy** of smoothed-bootstrap augmented tabular data.

## What it does

For each dataset, the real data are split by **repeated stratified k-fold** cross-validation
(5 folds × 5 repeats). Within every fold, synthetic data are generated **only from the
training portion** and the following conditions are compared on the untouched real test fold:

| Condition | Training data | Role |
|-----------|---------------|------|
| **TRTR** | real (train fold) | reference ceiling |
| **TSTR** | synthetic only | gold-standard utility |
| **TR+S** | real + synthetic | augmentation / deployment |
| NAIVE | plain bootstrap (no smoothing) | lower anchor |

Classification reports AUROC, AUPRC, F1, and Brier score; regression reports RMSE, MAE, R².
**Privacy** is evaluated with a membership-inference attack (MIA AUROC), distance-to-closest-record
(DCR), and nearest-neighbour distance ratio (NNDR). A **bandwidth sweep** traces the
privacy–utility tradeoff.

## Files

- `tstr_privacy_eval.py` — the evaluation engine (generators, TSTR/TRTR/TR+S harness,
  MIA/DCR/NNDR privacy, bandwidth sweep). Importable as a library.
- `reproduce_tables.py` — runs the full 5×5 evaluation on the ten study datasets and
  prints/writes the paper's utility and privacy tables.
- `requirements.txt`, `LICENSE` (MIT), `CITATION.cff`.

## Install

```bash
pip install -r requirements.txt
```

## Reproduce the paper's tables

The ten datasets are publicly available (UCI Machine Learning Repository and Kaggle; see the
paper for exact sources) and are **not** redistributed here. Download them, apply any
study-specific cleaning (imputation, down-sampling) described in the manuscript, place the
CSV or XLSX files in a `data/` folder, and run:

```bash
python reproduce_tables.py
```

Files are matched to datasets by a keyword in the filename. The script repairs Excel's
auto-date corruption of range labels (e.g. `9-May` → `5-9`) and removes the target-leakage
columns noted in the paper (`Density` from Body fat, `time` from Heart Failure). Per-fold and
privacy results are written to `results/`.

## Use as a library

```python
import pandas as pd, tstr_privacy_eval as T

df = pd.read_csv("mydata.csv")
cfg = T.EvalConfig(task="classification", n_target=1000,
                   n_splits=5, n_repeats=5, bandwidth=0.25)
res = T.evaluate_dataset(df, target="outcome", cfg=cfg, categorical=["sex"])
print(res.groupby("condition")["auroc"].mean())       # TRTR / TSTR / TR+S / NAIVE
print(res.attrs["privacy"][["mia_auc", "dcr_ratio_median"]].mean())
```

## Notes

- SMOTE and CTGAN/TVAE generators are included as optional comparators; the paper's
  synthpop / Gaussian-copula comparison was run in a separate R/SDV pipeline.
- The membership-inference result is an empirical attack, not a formal differential-privacy
  guarantee. Do not treat any output as safe to release without review, and do not run
  protected health information through a hosted deployment.

## Citation

See `CITATION.cff`. Code is released under the MIT License; see `LICENSE`.
