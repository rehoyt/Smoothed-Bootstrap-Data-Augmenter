#!/usr/bin/env python3
# ===========================================================================
#  TSTR / TRTR / TR+S utility and membership-inference privacy evaluation
#  for smoothed-bootstrap tabular data augmentation.
#
#  Companion code for:
#    Hoyt R, Jagarapu J, Angelis D. "Tabular Medical Data Augmentation Using
#    Smoothed Bootstrapping." (manuscript).
#
#  Authors : Robert Hoyt, Jawahar Jagarapu, Dimitrios Angelis
#  Contact : rehoyt@gmail.com
#  License : MIT (see LICENSE)
#  Repo    : https://github.com/rehoyt/   (update to the final repository URL)
#
#  Scope: this module implements the utility protocol (TSTR, TRTR, TR+S, and a
#  naive-resample lower anchor) and the privacy checks (membership-inference
#  attack [MIA], distance-to-closest-record [DCR], nearest-neighbour distance
#  ratio [NNDR]) reported in the paper, plus the bandwidth sweep. SMOTE and
#  CTGAN/TVAE generators are included as optional comparators and are NOT part
#  of the paper's synthpop / Gaussian-copula comparison (run separately).
# ===========================================================================
"""
TSTR utility evaluation for smoothed-bootstrap data augmentation.

Implements the protocol: for each real dataset, run repeated stratified k-fold CV.
Within every fold, generate synthetic data FROM THE TRAINING FOLD ONLY (smoothed
bootstrap), then compare four training conditions on the untouched real test fold:

    TRTR      : train real,            test real   (reference ceiling)
    TSTR      : train synthetic only,  test real   (gold-standard utility)
    TR+S      : train real + synthetic, test real  (augmentation / deployment)
    NAIVE     : train plain bootstrap,  test real   (lower anchor, no smoothing)

Results are aggregated across folds/repeats/seeds and across datasets, with a
Wilcoxon signed-rank test on paired per-dataset scores (TSTR vs TRTR, TR+S vs TRTR).

Key guardrails baked in:
  * Synthetic data is fit ONLY on the training fold; the test fold is never touched
    by the generator or by preprocessing .fit().
  * Preprocessing is fit on training data and applied to the test fold.
  * Multiple generation seeds per fold to separate signal from a lucky draw.

Author: protocol reference implementation. Dependencies: numpy, pandas,
scikit-learn, scipy.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# --------------------------------------------------------------------------- #
# Smoothed bootstrap generator
# --------------------------------------------------------------------------- #
def smoothed_bootstrap(
    X: np.ndarray,
    y: np.ndarray,
    n_target: int,
    task: str,
    numeric_mask: np.ndarray,
    rng: np.random.Generator,
    bandwidth: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``n_target`` synthetic rows via smoothed bootstrap.

    Rows are resampled with replacement; Gaussian noise scaled by ``bandwidth``
    times each numeric feature's per-class std is added to numeric columns.
    Categorical columns (numeric_mask == False) are copied without noise.
    For classification, sampling is stratified so class balance is preserved.

    NB: X, y here are the TRAINING FOLD ONLY.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n_features = X.shape[1]
    Xs = np.empty((n_target, n_features), dtype=float)
    ys = np.empty(n_target, dtype=y.dtype)

    if task == "classification":
        classes, counts = np.unique(y, return_counts=True)
        proportions = counts / counts.sum()
        alloc = np.floor(proportions * n_target).astype(int)
        alloc[-1] = n_target - alloc[:-1].sum()  # make it sum exactly
        groups = [(c, X[y == c]) for c in classes]
    else:
        # single "group" for regression; bin-based smoothing on y too
        alloc = [n_target]
        groups = [(None, X)]

    row = 0
    for (cls, Xg), n_take in zip(groups, alloc):
        if n_take <= 0:
            continue
        # per-(group) numeric std for noise scaling; guard against zero std
        std = Xg.std(axis=0)
        std[std == 0] = 1e-9
        idx = rng.integers(0, Xg.shape[0], size=n_take)
        sampled = Xg[idx].copy()
        noise = rng.normal(0.0, 1.0, size=sampled.shape) * (bandwidth * std)
        noise[:, ~numeric_mask] = 0.0  # do not perturb categorical columns
        sampled += noise
        Xs[row:row + n_take] = sampled
        if task == "classification":
            ys[row:row + n_take] = cls
        else:
            # regression: carry the resampled targets, lightly smoothed.
            # For regression there is a single group, so idx indexes full y.
            y_std = max(float(y.std()), 1e-9)
            ys[row:row + n_take] = y[idx] + rng.normal(0.0, bandwidth * y_std, size=n_take)
        row += n_take
    return Xs, ys


def naive_bootstrap(X, y, n_target, rng):
    """Plain resample with replacement, no smoothing (lower-anchor condition)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    idx = rng.integers(0, X.shape[0], size=n_target)
    return X[idx].copy(), y[idx].copy()


# --------------------------------------------------------------------------- #
# Baseline generators for method comparison: SMOTE and CTGAN / TVAE
# --------------------------------------------------------------------------- #
def _smote_group(Xg, rng, k, n_take, numeric_mask):
    """Generate n_take SMOTE-interpolated rows within a single group of rows."""
    n = Xg.shape[0]
    if n == 1:                       # nothing to interpolate toward; copy
        return np.repeat(Xg, n_take, axis=0)
    kk = int(min(k, n - 1))
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(Xg)   # +1 includes self
    neigh = nn.kneighbors(Xg, return_distance=False)[:, 1:]  # drop self column
    base_idx = rng.integers(0, n, size=n_take)
    pick = rng.integers(0, kk, size=n_take)
    nbr_idx = neigh[base_idx, pick]
    gap = rng.random(size=(n_take, 1))
    base, nbr = Xg[base_idx], Xg[nbr_idx]
    out = base + gap * (nbr - base)                     # interpolate all cols
    out[:, ~numeric_mask] = base[:, ~numeric_mask]      # keep base categoricals (SMOTE-NC style)
    return out


def smote_generate(X, y, n_target, task, numeric_mask, rng, k=5):
    """SMOTE-interpolation generator (Chawla et al., 2002), generalized to
    whole-dataset augmentation and to regression.

    Standard SMOTE only synthesizes minority-class points to rebalance classes.
    Here it is used as a general augmenter:
      * Classification: each class is oversampled in proportion to its size so
        the class balance of the real data is preserved (not rebalanced), up to
        n_target rows total. Interpolation is done within-class.
      * Regression: interpolation is global in feature space, and the target is
        interpolated between the same two neighbours with the same gap (a SMOTER-
        style extension).
    Categorical columns (numeric_mask == False) take the base point's value
    rather than being interpolated (SMOTE-NC style).
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    n_features = X.shape[1]
    Xs = np.empty((n_target, n_features)); ys = np.empty(n_target, dtype=y.dtype)

    if task == "classification":
        classes, counts = np.unique(y, return_counts=True)
        alloc = np.floor(counts / counts.sum() * n_target).astype(int)
        alloc[-1] = n_target - alloc[:-1].sum()
        row = 0
        for cls, n_take in zip(classes, alloc):
            if n_take <= 0:
                continue
            Xg = X[y == cls]
            Xs[row:row + n_take] = _smote_group(Xg, rng, k, n_take, numeric_mask)
            ys[row:row + n_take] = cls
            row += n_take
        return Xs, ys

    # regression: interpolate features and target together
    n = X.shape[0]
    kk = int(min(k, n - 1))
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(X)
    neigh = nn.kneighbors(X, return_distance=False)[:, 1:]
    base_idx = rng.integers(0, n, size=n_target)
    pick = rng.integers(0, kk, size=n_target)
    nbr_idx = neigh[base_idx, pick]
    gap = rng.random(size=(n_target, 1))
    base, nbr = X[base_idx], X[nbr_idx]
    Xs = base + gap * (nbr - base)
    Xs[:, ~numeric_mask] = base[:, ~numeric_mask]
    g = gap.ravel()
    ys = y[base_idx] + g * (y[nbr_idx] - y[base_idx])
    return Xs, ys


def deep_generate(method, X, y, n_target, task, numeric_mask, rng, epochs=100):
    """CTGAN / TVAE generator (Xu et al., 2019), via the `ctgan` package.

    Learns a global generative model of the joint (features, target) table and
    samples fresh rows from it, rather than perturbing individual real records.
    Requires `pip install ctgan`. Import is lazy so the rest of the module works
    without it installed.
    """
    try:
        from ctgan import CTGAN, TVAE
    except ImportError as e:
        raise ImportError("The 'ctgan' package is required for method="
                          f"'{method}'. Install with: pip install ctgan") from e
    import torch

    seed = int(rng.integers(0, 2**31 - 1))
    np.random.seed(seed); torch.manual_seed(seed)

    cols = [f"x{i}" for i in range(X.shape[1])]
    tcol = "__target__"
    df = pd.DataFrame(np.asarray(X, dtype=float), columns=cols)
    df[tcol] = np.asarray(y)
    discrete = [cols[i] for i in range(len(cols)) if not numeric_mask[i]]
    if task == "classification":
        discrete = discrete + [tcol]

    Model = CTGAN if method == "ctgan" else TVAE
    model = Model(epochs=epochs)
    model.fit(df, discrete_columns=discrete)
    samp = model.sample(n_target)
    Xs = samp[cols].to_numpy(dtype=float)
    ys = samp[tcol].to_numpy()
    if task == "classification":
        ys = ys.astype(y.dtype)
    return Xs, ys


def generate_synthetic(method, Xtr, ytr, n_target, task, numeric_mask, rng, cfg):
    """Dispatch to the configured synthetic-data generator.

    method in {smoothed_bootstrap, smote, ctgan, tvae}. Returns (X_syn, y_syn).
    """
    if method == "smoothed_bootstrap":
        return smoothed_bootstrap(Xtr, ytr, n_target, task, numeric_mask, rng,
                                  cfg.bandwidth)
    if method == "smote":
        return smote_generate(Xtr, ytr, n_target, task, numeric_mask, rng,
                              k=cfg.smote_k)
    if method in ("ctgan", "tvae"):
        return deep_generate(method, Xtr, ytr, n_target, task, numeric_mask, rng,
                             epochs=cfg.ctgan_epochs)
    raise ValueError(f"Unknown generator method: {method!r}")


# --------------------------------------------------------------------------- #
# Privacy check: distance to closest record (DCR)
# --------------------------------------------------------------------------- #
def privacy_metrics(
    X_syn: np.ndarray,
    X_train: np.ndarray,
    X_holdout: np.ndarray | None = None,
    dup_quantile: float = 0.01,
) -> dict:
    """Distance-to-closest-record (DCR) privacy diagnostics for synthetic data.

    Guards against a smoothed bootstrap that simply memorizes its seed points.
    All distances are computed in a space standardized on the TRAINING fold, so
    features contribute comparably.

    Reported:
      dcr_syn_*     : min / 5th-pctile / median Euclidean distance from each
                      synthetic row to its nearest REAL TRAINING row.
      dcr_holdout_* : same, for a real hold-out set vs training (a natural
                      reference for "how close should genuine unseen records be").
      dcr_ratio_median : median(dcr_syn) / median(dcr_holdout). Values well below
                      1.0 indicate synthetic rows sit closer to training data than
                      genuine unseen records do -> memorization / privacy risk.
      frac_near_dup : fraction of synthetic rows whose nearest-training distance
                      falls below the ``dup_quantile`` quantile of the training
                      set's own nearest-neighbor distances (near-duplicate share).

    Rule of thumb: dcr_ratio_median >= ~1 and a low frac_near_dup indicate the
    generator is not memorizing. Interpret bandwidth as the main control knob.
    """
    scaler = StandardScaler().fit(X_train)          # fit on training only
    Xt = scaler.transform(X_train)
    Xs = scaler.transform(X_syn)

    nn = NearestNeighbors(n_neighbors=2).fit(Xt)

    # synthetic -> two nearest training records (for DCR and NNDR)
    d2_syn, _ = nn.kneighbors(Xs, n_neighbors=2)
    d_syn = d2_syn[:, 0]
    # NNDR: nearest / second-nearest. Low values flag a synthetic row that
    # matches ONE specific training record much better than any other -> that
    # training patient is re-identifiable even if the row is not a duplicate.
    nndr = d2_syn[:, 0] / np.clip(d2_syn[:, 1], 1e-12, None)

    # training -> nearest *other* training record (self-excluded) to set a
    # near-duplicate threshold that reflects the data's intrinsic granularity
    d_tt, _ = NearestNeighbors(n_neighbors=2).fit(Xt).kneighbors(Xt, n_neighbors=2)
    d_train_nn = d_tt[:, 1]                          # column 0 is the point itself
    dup_thresh = np.quantile(d_train_nn, dup_quantile)

    out = {
        "dcr_syn_min": float(np.min(d_syn)),
        "dcr_syn_p05": float(np.quantile(d_syn, 0.05)),
        "dcr_syn_median": float(np.median(d_syn)),
        "frac_near_dup": float(np.mean(d_syn <= dup_thresh)),
        "nndr_median": float(np.median(nndr)),
        "frac_nndr_low": float(np.mean(nndr < 0.5)),
    }

    if X_holdout is not None and len(X_holdout) > 0:
        d_hold, _ = nn.kneighbors(scaler.transform(X_holdout), n_neighbors=1)
        d_hold = d_hold.ravel()
        out["dcr_holdout_median"] = float(np.median(d_hold))
        out["dcr_holdout_p05"] = float(np.quantile(d_hold, 0.05))
        med_hold = max(np.median(d_hold), 1e-12)
        out["dcr_ratio_median"] = float(np.median(d_syn) / med_hold)
    else:
        out["dcr_holdout_median"] = np.nan
        out["dcr_holdout_p05"] = np.nan
        out["dcr_ratio_median"] = np.nan
    return out


def membership_inference_attack(
    X_syn: np.ndarray,
    X_members: np.ndarray,
    X_nonmembers: np.ndarray,
) -> dict:
    """Distance-based membership inference attack (MIA) against synthetic data.

    This is the recognized gold-standard privacy test for synthetic data, and it
    catches leakage that duplicate- or DCR-only checks miss. The attacker asks:
    'given only the synthetic data, can I tell whether a particular real record
    was in the generator's TRAINING set?' If yes, the synthetic release leaks
    membership -- a real privacy breach under HIPAA-style reasoning.

    Design (fair, holdout-controlled):
      * members     = real records used to generate the synthetic data (train fold)
      * non-members = real records NOT used (an equal-status hold-out, e.g. the
                      test fold)
      * attack score for each record = -distance to its nearest synthetic row.
        Records close to the synthetic manifold are guessed to be members.
      * We report the ROC AUC of that score at separating members from
        non-members, evaluated on a BALANCED sample so AUC is not inflated by
        class imbalance.

    Interpretation:
      mia_auc  ~ 0.50  -> no membership signal (good; indistinguishable)
      mia_auc  > 0.60  -> meaningful leakage; tighten the smoothing bandwidth
      mia_advantage = 2*(AUC-0.5), in [0,1]; the attacker's edge over guessing
      mia_tpr_at_fpr_10 : true-positive rate at 10% false-positive rate. Worst-case
                      metric preferred by privacy auditors -- how many real patients
                      are confidently re-identified while rarely crying wolf.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    n = min(len(X_members), len(X_nonmembers))
    if n < 5:
        return {"mia_auc": np.nan, "mia_advantage": np.nan,
                "mia_tpr_at_fpr_10": np.nan, "mia_n_per_group": int(n)}

    rng = np.random.default_rng(0)
    mem = X_members[rng.choice(len(X_members), n, replace=False)]
    non = X_nonmembers[rng.choice(len(X_nonmembers), n, replace=False)]

    scaler = StandardScaler().fit(np.vstack([mem, non]))
    nn = NearestNeighbors(n_neighbors=1).fit(scaler.transform(X_syn))
    d_mem = nn.kneighbors(scaler.transform(mem), n_neighbors=1)[0].ravel()
    d_non = nn.kneighbors(scaler.transform(non), n_neighbors=1)[0].ravel()

    y_true = np.r_[np.ones(n), np.zeros(n)]
    score = -np.r_[d_mem, d_non]                 # closer to synthetic => member
    auc = float(roc_auc_score(y_true, score))

    fpr, tpr, _ = roc_curve(y_true, score)
    tpr_at_10 = float(np.interp(0.10, fpr, tpr))

    return {
        "mia_auc": auc,
        "mia_advantage": float(2 * abs(auc - 0.5)),
        "mia_tpr_at_fpr_10": tpr_at_10,
        "mia_n_per_group": int(n),
    }


# --------------------------------------------------------------------------- #
# Model panel and metrics
# --------------------------------------------------------------------------- #
def model_panel(task: str, seed: int) -> dict:
    if task == "classification":
        return {
            "logreg": LogisticRegression(max_iter=2000),
            "rf": RandomForestClassifier(n_estimators=300, random_state=seed),
            "gb": GradientBoostingClassifier(random_state=seed),
        }
    return {
        "ridge": Ridge(),
        "rf": RandomForestRegressor(n_estimators=300, random_state=seed),
        "gb": GradientBoostingRegressor(random_state=seed),
    }


def clf_metrics(y_true, proba, pred) -> dict:
    out = {"f1": f1_score(y_true, pred, average="binary" if
                          len(np.unique(y_true)) == 2 else "macro")}
    try:
        if len(np.unique(y_true)) == 2:
            out["auroc"] = roc_auc_score(y_true, proba)
            out["auprc"] = average_precision_score(y_true, proba)
            out["brier"] = brier_score_loss(y_true, proba)
        else:
            out["auroc"] = roc_auc_score(y_true, proba, multi_class="ovr")
            out["auprc"] = np.nan
            out["brier"] = np.nan
    except ValueError:
        out.update(auroc=np.nan, auprc=np.nan, brier=np.nan)
    return out


def reg_metrics(y_true, pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
    }


# --------------------------------------------------------------------------- #
# Preprocessing (fit on training data only)
# --------------------------------------------------------------------------- #
def build_preprocessor(numeric_idx, categorical_idx):
    return ColumnTransformer([
        ("num", StandardScaler(), list(numeric_idx)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical_idx)),
    ], remainder="drop")


# --------------------------------------------------------------------------- #
# Core evaluation
# --------------------------------------------------------------------------- #
@dataclass
class EvalConfig:
    task: str                       # "classification" or "regression"
    n_target: int = 1000            # augment each training fold up to this size
    n_splits: int = 5
    n_repeats: int = 5
    gen_seeds: tuple = (0, 1, 2)    # synthetic generations per fold
    bandwidth: float = 0.25         # used only by method="smoothed_bootstrap"
    method: str = "smoothed_bootstrap"   # generator: smoothed_bootstrap|smote|ctgan|tvae
    smote_k: int = 5                # neighbours for the SMOTE generator
    ctgan_epochs: int = 100         # training epochs for ctgan/tvae
    random_state: int = 42


def _fit_score(model, Xtr, ytr, Xte, yte, pre, task):
    """Fit preprocessing+model on (Xtr,ytr), score on (Xte,yte)."""
    pipe = Pipeline([("pre", clone(pre)), ("model", clone(model))])
    pipe.fit(Xtr, ytr)
    if task == "classification":
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(Xte)
            proba = proba[:, 1] if proba.shape[1] == 2 else proba
        else:
            proba = pipe.decision_function(Xte)
        pred = pipe.predict(Xte)
        return clf_metrics(yte, proba, pred)
    return reg_metrics(yte, pipe.predict(Xte))


def evaluate_dataset(
    df: pd.DataFrame,
    target: str,
    cfg: EvalConfig,
    categorical: list[str] | None = None,
    dataset_name: str = "dataset",
) -> pd.DataFrame:
    """Run the full TSTR protocol on one dataset. Returns a tidy DataFrame.

    Text columns are handled automatically: any non-numeric feature column is
    treated as categorical and label-encoded to integer codes, and a non-numeric
    classification target (e.g. the thyroid dataset's Yes/No 'Recurred') is
    label-encoded to 0/1. This keeps every dataset on the same footing whether the
    labels arrive as 0/1 or as text.
    """
    categorical = set(categorical or [])
    feature_cols = [c for c in df.columns if c != target]

    # Any non-numeric feature column is categorical, even if not listed as such.
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            categorical.add(c)

    work = df.copy()
    for c in categorical:
        if c in work.columns:
            work[c] = LabelEncoder().fit_transform(work[c].astype(str))

    numeric_cols = [c for c in feature_cols if c not in categorical]
    numeric_idx = [feature_cols.index(c) for c in numeric_cols]
    categorical_idx = [feature_cols.index(c) for c in feature_cols if c in categorical]
    numeric_mask = np.array([c not in categorical for c in feature_cols])

    X = work[feature_cols].to_numpy(dtype=float)
    if cfg.task == "classification" and not pd.api.types.is_numeric_dtype(df[target]):
        y = LabelEncoder().fit_transform(df[target].astype(str))   # Yes/No -> 0/1
    elif cfg.task == "classification":
        y = df[target].to_numpy()
    else:
        y = df[target].to_numpy(dtype=float)
    pre = build_preprocessor(numeric_idx, categorical_idx)

    # stratification target (bin the response for regression)
    if cfg.task == "classification":
        strat = y
    else:
        strat = pd.qcut(y, q=min(10, len(np.unique(y))), labels=False, duplicates="drop")

    rskf = RepeatedStratifiedKFold(
        n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, random_state=cfg.random_state
    )
    rows = []
    priv_rows = []
    for fold_id, (tr, te) in enumerate(rskf.split(X, strat)):
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        panel = model_panel(cfg.task, seed=cfg.random_state)

        # --- privacy (DCR) check per generation seed: synthetic vs training,
        #     with the untouched real test fold as the natural hold-out ref --- #
        for gs in cfg.gen_seeds:
            rng = np.random.default_rng(cfg.random_state + 1000 * gs + fold_id)
            Xsyn_p, _ = generate_synthetic(
                cfg.method, Xtr, ytr, cfg.n_target, cfg.task, numeric_mask, rng, cfg)
            pm = privacy_metrics(Xsyn_p, Xtr, X_holdout=Xte)
            mia = membership_inference_attack(
                Xsyn_p, X_members=Xtr, X_nonmembers=Xte)
            priv_rows.append(dict(dataset=dataset_name, fold=fold_id,
                                  gen_seed=gs, **pm, **mia))

        for mname, model in panel.items():
            # --- TRTR: reference ceiling -------------------------------- #
            m = _fit_score(model, Xtr, ytr, Xte, yte, pre, cfg.task)
            rows.append(dict(dataset=dataset_name, fold=fold_id, model=mname,
                             condition="TRTR", gen_seed=-1, **m))

            # --- synthetic conditions, averaged over generation seeds --- #
            for gs in cfg.gen_seeds:
                rng = np.random.default_rng(cfg.random_state + 1000 * gs + fold_id)

                Xsyn, ysyn = generate_synthetic(
                    cfg.method, Xtr, ytr, cfg.n_target, cfg.task, numeric_mask, rng, cfg)
                Xnv, ynv = naive_bootstrap(Xtr, ytr, cfg.n_target, rng)

                # TSTR: synthetic only
                m = _fit_score(model, Xsyn, ysyn, Xte, yte, pre, cfg.task)
                rows.append(dict(dataset=dataset_name, fold=fold_id, model=mname,
                                 condition="TSTR", gen_seed=gs, **m))

                # TR+S: real + synthetic
                Xaug = np.vstack([Xtr, Xsyn]); yaug = np.concatenate([ytr, ysyn])
                m = _fit_score(model, Xaug, yaug, Xte, yte, pre, cfg.task)
                rows.append(dict(dataset=dataset_name, fold=fold_id, model=mname,
                                 condition="TR+S", gen_seed=gs, **m))

                # NAIVE: plain bootstrap (lower anchor)
                m = _fit_score(model, Xnv, ynv, Xte, yte, pre, cfg.task)
                rows.append(dict(dataset=dataset_name, fold=fold_id, model=mname,
                                 condition="NAIVE", gen_seed=gs, **m))
    results = pd.DataFrame(rows)
    # Per-fold privacy diagnostics travel alongside utility results.
    results.attrs["privacy"] = pd.DataFrame(priv_rows)
    return results


# --------------------------------------------------------------------------- #
# Aggregation across datasets + significance testing
# --------------------------------------------------------------------------- #
def summarize(results: pd.DataFrame, primary_metric: str) -> pd.DataFrame:
    """Mean +/- std per (dataset, model, condition) for the primary metric."""
    g = (results.groupby(["dataset", "model", "condition"])[primary_metric]
         .agg(["mean", "std", "count"]).reset_index())
    return g


def utility_ratio(results: pd.DataFrame, primary_metric: str,
                  higher_is_better: bool = True) -> pd.DataFrame:
    """Per-dataset TSTR/TRTR and TR+S/TRTR ratios (averaged over folds/models)."""
    per = (results.groupby(["dataset", "condition"])[primary_metric]
           .mean().unstack("condition"))
    out = pd.DataFrame(index=per.index)
    ref = per["TRTR"]
    for cond in ["TSTR", "TR+S", "NAIVE"]:
        if cond in per:
            out[f"{cond}/TRTR"] = (per[cond] / ref) if higher_is_better \
                else (ref / per[cond])
    return out.reset_index()


def summarize_privacy(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-fold DCR privacy diagnostics (from results.attrs)."""
    priv = results.attrs.get("privacy")
    if priv is None or len(priv) == 0:
        return pd.DataFrame()
    cols = ["dcr_ratio_median", "frac_near_dup", "nndr_median", "frac_nndr_low",
            "mia_auc", "mia_advantage", "mia_tpr_at_fpr_10"]
    cols = [c for c in cols if c in priv.columns]
    return (priv.groupby("dataset")[cols].mean().reset_index())


def wilcoxon_vs_reference(results: pd.DataFrame, primary_metric: str,
                          conditions=("TSTR", "TR+S")) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test across datasets: condition vs TRTR."""
    per = (results.groupby(["dataset", "condition"])[primary_metric]
           .mean().unstack("condition"))
    ref = per["TRTR"].to_numpy()
    rows = []
    for cond in conditions:
        if cond not in per:
            continue
        vals = per[cond].to_numpy()
        mask = ~(np.isnan(vals) | np.isnan(ref))
        try:
            stat, p = stats.wilcoxon(vals[mask], ref[mask])
        except ValueError:
            stat, p = np.nan, np.nan
        rows.append(dict(condition=cond, n_datasets=int(mask.sum()),
                         median_diff=float(np.median(vals[mask] - ref[mask])),
                         wilcoxon_stat=stat, p_value=p))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Bandwidth sweep: privacy (MIA AUC) vs utility (TSTR/TRTR) tradeoff
# --------------------------------------------------------------------------- #
@dataclass
class DatasetSpec:
    """One dataset for the sweep."""
    name: str
    df: pd.DataFrame
    target: str
    categorical: list = field(default_factory=list)


def bandwidth_sweep(
    datasets: list[DatasetSpec],
    bandwidths,
    task: str,
    n_target: int = 1000,
    n_splits: int = 5,
    n_repeats: int = 2,
    gen_seeds: tuple = (0, 1, 2),
    random_state: int = 42,
) -> pd.DataFrame:
    """Sweep the smoothing bandwidth and trace the privacy/utility tradeoff.

    For each bandwidth, runs the full protocol on every dataset, then averages
    ACROSS datasets (and folds/repeats/seeds). Returns one tidy row per
    bandwidth with:

      mia_auc        : mean membership-inference AUC (privacy risk; want ~0.50)
      mia_auc_std    : between-dataset spread (small-n estimates are noisy)
      util_ratio     : mean TSTR utility as a fraction of the TRTR ceiling, where
                       higher is always better (AUROC ratio for classification,
                       inverse-RMSE ratio for regression)
      tstr, trtr     : the raw primary-metric means behind util_ratio
      frac_near_dup, nndr_median : supporting proximity diagnostics

    The recommended operating point is the smallest bandwidth whose mean MIA AUC
    is at/near 0.50 while util_ratio remains acceptable for your use case.
    """
    higher_is_better = (task == "classification")
    primary = "auroc" if task == "classification" else "rmse"
    rows = []
    for bw in bandwidths:
        per_ds_mia, per_ds_ratio, per_ds_tstr, per_ds_trtr = [], [], [], []
        per_ds_dup, per_ds_nndr = [], []
        for spec in datasets:
            cfg = EvalConfig(task=task, n_target=n_target, n_splits=n_splits,
                             n_repeats=n_repeats, gen_seeds=gen_seeds,
                             bandwidth=float(bw), random_state=random_state)
            res = evaluate_dataset(spec.df, spec.target, cfg,
                                   categorical=spec.categorical,
                                   dataset_name=spec.name)
            means = res.groupby("condition")[primary].mean()
            tstr, trtr = means.get("TSTR", np.nan), means.get("TRTR", np.nan)
            ratio = (tstr / trtr) if higher_is_better else (trtr / tstr)
            per_ds_tstr.append(tstr); per_ds_trtr.append(trtr)
            per_ds_ratio.append(ratio)
            priv = res.attrs["privacy"]
            per_ds_mia.append(priv["mia_auc"].mean())
            per_ds_dup.append(priv["frac_near_dup"].mean())
            per_ds_nndr.append(priv["nndr_median"].mean())
        rows.append(dict(
            bandwidth=float(bw),
            mia_auc=float(np.nanmean(per_ds_mia)),
            mia_auc_std=float(np.nanstd(per_ds_mia)),
            util_ratio=float(np.nanmean(per_ds_ratio)),
            util_ratio_std=float(np.nanstd(per_ds_ratio)),
            tstr=float(np.nanmean(per_ds_tstr)),
            trtr=float(np.nanmean(per_ds_trtr)),
            frac_near_dup=float(np.nanmean(per_ds_dup)),
            nndr_median=float(np.nanmean(per_ds_nndr)),
        ))
    return pd.DataFrame(rows)


def recommend_bandwidth(sweep: pd.DataFrame, mia_threshold: float = 0.55,
                        min_util_ratio: float = 0.90) -> dict:
    """Pick the smallest bandwidth meeting a privacy bar with acceptable utility.

    Default policy: mean MIA AUC <= 0.55 (near indistinguishable) AND utility at
    least 90% of the TRTR ceiling. Returns the chosen row, or the best-privacy
    row with a warning flag if no bandwidth satisfies both.
    """
    ok = sweep[(sweep.mia_auc <= mia_threshold) &
               (sweep.util_ratio >= min_util_ratio)].sort_values("bandwidth")
    if len(ok):
        r = ok.iloc[0]
        return {"bandwidth": float(r.bandwidth), "mia_auc": float(r.mia_auc),
                "util_ratio": float(r.util_ratio), "satisfied": True}
    best = sweep.sort_values("mia_auc").iloc[0]
    return {"bandwidth": float(best.bandwidth), "mia_auc": float(best.mia_auc),
            "util_ratio": float(best.util_ratio), "satisfied": False,
            "note": "No bandwidth met both bars; consider a different generator."}


def plot_sweep(sweep: pd.DataFrame, task: str, path: str = "bandwidth_sweep.png"):
    """Dual-axis plot: MIA AUC (privacy) and TSTR/TRTR utility vs bandwidth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    c_priv, c_util = "#C0392B", "#1F6FB2"

    ax1.errorbar(sweep.bandwidth, sweep.mia_auc, yerr=sweep.mia_auc_std,
                 color=c_priv, marker="o", capsize=3, label="MIA AUC (privacy risk)")
    ax1.axhline(0.5, color=c_priv, ls=":", lw=1, alpha=0.7)
    ax1.text(sweep.bandwidth.iloc[0], 0.505, "no leakage (0.5)",
             color=c_priv, fontsize=8, va="bottom")
    ax1.set_xlabel("Smoothing bandwidth")
    ax1.set_ylabel("Membership-inference AUC", color=c_priv)
    ax1.tick_params(axis="y", labelcolor=c_priv)
    ax1.set_ylim(0.45, 1.02)

    ax2 = ax1.twinx()
    ax2.errorbar(sweep.bandwidth, sweep.util_ratio, yerr=sweep.util_ratio_std,
                 color=c_util, marker="s", capsize=3, label="TSTR/TRTR utility")
    ax2.axhline(1.0, color=c_util, ls=":", lw=1, alpha=0.6)
    ax2.set_ylabel("TSTR utility (fraction of TRTR ceiling)", color=c_util)
    ax2.tick_params(axis="y", labelcolor=c_util)

    rec = recommend_bandwidth(sweep)
    if rec.get("satisfied"):
        ax1.axvline(rec["bandwidth"], color="green", ls="--", lw=1.2)
        ax1.text(rec["bandwidth"], 0.98, f"  rec. bw={rec['bandwidth']:.2f}",
                 color="green", fontsize=8, va="top")

    ax1.set_title(f"Privacy-utility tradeoff across bandwidths ({task})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Method comparison: smoothed bootstrap vs SMOTE vs CTGAN/TVAE
# --------------------------------------------------------------------------- #
def compare_methods(
    datasets: list,
    methods=("smoothed_bootstrap", "smote", "ctgan"),
    task: str = "classification",
    n_target: int = 1000,
    n_splits: int = 5,
    n_repeats: int = 2,
    gen_seeds: tuple = (0,),
    bandwidth: float = 1.0,
    ctgan_epochs: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run several generators through the same TSTR + MIA harness and tabulate
    the utility-vs-privacy result for each, averaged across datasets.

    Every method is evaluated with identical cross-validation, model panel, and
    privacy attack, so the comparison is apples-to-apples. For smoothed bootstrap
    the given `bandwidth` is used (pick one from your sweep, e.g. the recommended
    operating point). SMOTE and CTGAN/TVAE ignore bandwidth.

    Returns one row per method with mean TSTR/TRTR utility ratio and mean MIA AUC
    (privacy risk), plus supporting proximity diagnostics.
    """
    higher = (task == "classification")
    primary = "auroc" if task == "classification" else "rmse"
    rows = []
    for method in methods:
        u, mia, dup, nndr, tstr_v, trtr_v = [], [], [], [], [], []
        for spec in datasets:
            cfg = EvalConfig(task=task, n_target=n_target, n_splits=n_splits,
                             n_repeats=n_repeats, gen_seeds=gen_seeds,
                             bandwidth=bandwidth, method=method,
                             ctgan_epochs=ctgan_epochs, random_state=random_state)
            res = evaluate_dataset(spec.df, spec.target, cfg,
                                   categorical=spec.categorical,
                                   dataset_name=spec.name)
            means = res.groupby("condition")[primary].mean()
            t, r = means.get("TSTR", np.nan), means.get("TRTR", np.nan)
            u.append((t / r) if higher else (r / t))
            tstr_v.append(t); trtr_v.append(r)
            p = res.attrs["privacy"]
            mia.append(p["mia_auc"].mean()); dup.append(p["frac_near_dup"].mean())
            nndr.append(p["nndr_median"].mean())
        rows.append(dict(
            method=method,
            util_ratio=float(np.nanmean(u)), util_ratio_std=float(np.nanstd(u)),
            mia_auc=float(np.nanmean(mia)), mia_auc_std=float(np.nanstd(mia)),
            tstr=float(np.nanmean(tstr_v)), trtr=float(np.nanmean(trtr_v)),
            frac_near_dup=float(np.nanmean(dup)), nndr_median=float(np.nanmean(nndr)),
        ))
    return pd.DataFrame(rows)


def plot_comparison(comp: pd.DataFrame, task: str, path: str = "method_comparison.png"):
    """Scatter of privacy (MIA AUC) vs utility (TSTR/TRTR) with one point per method.

    The ideal method sits toward the bottom (low leakage) and right (high utility)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for _, r in comp.iterrows():
        ax.errorbar(r.util_ratio, r.mia_auc, xerr=r.util_ratio_std,
                    yerr=r.mia_auc_std, marker="o", capsize=3, ms=8)
        ax.annotate(r.method, (r.util_ratio, r.mia_auc),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.text(ax.get_xlim()[0], 0.505, "no leakage (0.5)", fontsize=8, color="grey")
    ax.set_xlabel("TSTR / TRTR utility (higher is better)")
    ax.set_ylabel("Membership-inference AUC (lower is better)")
    ax.set_title(f"Privacy vs utility by generation method ({task})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Demo / self-test on synthetic toy datasets
# --------------------------------------------------------------------------- #
def _demo():
    from sklearn.datasets import make_classification, make_regression

    print("Running demo on 3 toy datasets per task (small n)...\n")
    all_clf, all_reg = [], []
    for i in range(3):
        Xc, yc = make_classification(n_samples=300, n_features=12, n_informative=6,
                                     weights=[0.7, 0.3], random_state=i)
        dfc = pd.DataFrame(Xc, columns=[f"f{j}" for j in range(Xc.shape[1])])
        dfc["target"] = yc
        cfg_c = EvalConfig(task="classification", n_target=1000,
                           n_splits=5, n_repeats=1, gen_seeds=(0, 1))
        all_clf.append(evaluate_dataset(dfc, "target", cfg_c,
                                        dataset_name=f"clf_{i}"))

        Xr, yr = make_regression(n_samples=300, n_features=12, n_informative=6,
                                 noise=12.0, random_state=i)
        dfr = pd.DataFrame(Xr, columns=[f"f{j}" for j in range(Xr.shape[1])])
        dfr["target"] = yr
        cfg_r = EvalConfig(task="regression", n_target=1000,
                           n_splits=5, n_repeats=1, gen_seeds=(0, 1))
        all_reg.append(evaluate_dataset(dfr, "target", cfg_r,
                                        dataset_name=f"reg_{i}"))

    def _combine(frames):
        combined = pd.concat(frames, ignore_index=True)
        priv = pd.concat([f.attrs.get("privacy", pd.DataFrame()) for f in frames],
                         ignore_index=True)
        combined.attrs["privacy"] = priv          # attrs are lost by concat
        return combined

    clf = _combine(all_clf)
    reg = _combine(all_reg)

    print("=== CLASSIFICATION (AUROC) ===")
    print(summarize(clf, "auroc").round(3).to_string(index=False))
    print("\nUtility ratios (higher better):")
    print(utility_ratio(clf, "auroc", higher_is_better=True).round(3).to_string(index=False))
    print("\nWilcoxon vs TRTR:")
    print(wilcoxon_vs_reference(clf, "auroc").round(4).to_string(index=False))
    print("\nPrivacy / DCR (ratio>=~1 and low frac_near_dup = safe):")
    print(summarize_privacy(clf).round(3).to_string(index=False))

    print("\n=== REGRESSION (RMSE) ===")
    print(summarize(reg, "rmse").round(3).to_string(index=False))
    print("\nUtility ratios (lower RMSE better -> ratio TRTR/cond):")
    print(utility_ratio(reg, "rmse", higher_is_better=False).round(3).to_string(index=False))
    print("\nWilcoxon vs TRTR:")
    print(wilcoxon_vs_reference(reg, "rmse").round(4).to_string(index=False))
    print("\nDemo complete.")


def _sweep_demo(out_png="bandwidth_sweep_demo.png"):
    """Demonstrate the bandwidth sweep on a few toy classification datasets."""
    from sklearn.datasets import make_classification
    specs = []
    for i in range(3):
        X, y = make_classification(n_samples=300, n_features=12, n_informative=6,
                                   weights=[0.7, 0.3], random_state=i)
        df = pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])])
        df["target"] = y
        specs.append(DatasetSpec(name=f"clf_{i}", df=df, target="target"))

    grid = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
    print(f"Sweeping bandwidths {grid} over {len(specs)} datasets...\n")
    sweep = bandwidth_sweep(specs, grid, task="classification",
                            n_target=1000, n_splits=3, n_repeats=1, gen_seeds=(0, 1))
    print(sweep.round(3).to_string(index=False))
    rec = recommend_bandwidth(sweep)
    print("\nRecommended operating point:", {k: (round(v, 3) if isinstance(v, float)
                                                  else v) for k, v in rec.items()})
    p = plot_sweep(sweep, task="classification", path=out_png)
    print("Saved plot to", p)
    return sweep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true", help="run built-in toy demo")
    ap.add_argument("--sweep", action="store_true",
                    help="run bandwidth privacy/utility sweep")
    ap.add_argument("--bandwidths", nargs="*", type=float,
                    default=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0],
                    help="bandwidth grid for the sweep")
    ap.add_argument("--csv", help="path to a CSV dataset")
    ap.add_argument("--target", help="target column name")
    ap.add_argument("--task", choices=["classification", "regression"])
    ap.add_argument("--categorical", nargs="*", default=[],
                    help="names of categorical feature columns")
    ap.add_argument("--n-target", type=int, default=1000)
    ap.add_argument("--out", default="tstr_results.csv")
    args = ap.parse_args()

    if args.sweep and args.csv:
        df = pd.read_csv(args.csv)
        spec = DatasetSpec(name=args.csv, df=df, target=args.target,
                           categorical=args.categorical)
        sweep = bandwidth_sweep([spec], args.bandwidths, task=args.task,
                                n_target=args.n_target)
        print(sweep.round(3).to_string(index=False))
        print("\nRecommended:", recommend_bandwidth(sweep))
        sweep.to_csv(args.out.replace(".csv", "_sweep.csv"), index=False)
        plot_sweep(sweep, task=args.task,
                   path=args.out.replace(".csv", "_sweep.png"))
        print("Saved sweep table and plot.")
    elif args.sweep:
        _sweep_demo()
    elif args.demo or not args.csv:
        _demo()
    else:
        df = pd.read_csv(args.csv)
        cfg = EvalConfig(task=args.task, n_target=args.n_target)
        res = evaluate_dataset(df, args.target, cfg, categorical=args.categorical,
                               dataset_name=args.csv)
        res.to_csv(args.out, index=False)
        metric = "auroc" if args.task == "classification" else "rmse"
        print(summarize(res, metric).round(3).to_string(index=False))
        print("\nPrivacy / DCR (ratio>=~1 and low frac_near_dup = safe):")
        print(summarize_privacy(res).round(3).to_string(index=False))
        priv_out = args.out.replace(".csv", "_privacy.csv")
        res.attrs["privacy"].to_csv(priv_out, index=False)
        print("\nSaved per-fold results to", args.out, "and privacy to", priv_out)
