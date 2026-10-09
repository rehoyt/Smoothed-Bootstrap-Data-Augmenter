#!/usr/bin/env python3
"""
Reproduce the paper's utility (TSTR/TRTR/TR+S) and privacy (MIA) tables.

Place the ten study datasets (CSV or XLSX) in ./data/ and run:

    python reproduce_tables.py

Each file is matched to its dataset by a keyword in the filename, the same
preprocessing used in the study is applied (Excel-date repair of range labels,
removal of target-leakage columns), and the smoothed-bootstrap evaluation is
run at 5-fold cross-validation repeated 5 times (bandwidth 0.25, augmentation to
1,000 rows). Per-dataset and aggregate tables are written to ./results/.

Datasets are publicly available (UCI Machine Learning Repository and Kaggle);
see the manuscript for the exact sources. They are not redistributed here.
"""
import os, re, glob
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")

import tstr_privacy_eval as T

# --- dataset configuration: filename keyword -> (name, target, task, categorical) ---
CONFIG = {
    "thyroid": ("Thyroid Cancer Recurrence", "Recurred", "classification",
                ["Gender", "Smoking", "Hx Smoking", "Hx Radiothreapy", "Thyroid Function",
                 "Physical Examination", "Adenopathy", "Pathology", "Focality", "Risk",
                 "T", "N", "M", "Stage", "Response"]),
    "wisconsin": ("Breast Cancer Wisconsin", "Class", "classification", []),
    "recurren": ("Breast Cancer Recurrence", "target", "classification",
                 ["age", "menopause", "tumor-size", "inv-nodes", "node-caps",
                  "breast", "breast-quad", "irradiat"]),
    "parkinson": ("Parkinson's disease", "class", "classification", []),
    "gallstone": ("Gallstones", "Gallstone Status", "classification", []),
    "kidney": ("Chronic Kidney Disease", "class", "classification",
               ["htn", "dm", "cad", "appetite", "pedal_edema", "anemia"]),
    "heart_disease": ("Heart Disease Prediction", "heart disease", "classification",
                      ["gender", "chest pain", "rest ECG", "slope peak exc ST",
                       "thallium stress test"]),
    "failure": ("Heart Failure Prediction", "DEATH_EVENT", "classification", []),
    "hepatitis": ("Hepatitis", "Class", "classification", ["SEX"]),
    "bodyfat": ("Body fat", "class", "regression", []),
    "fat": ("Body fat", "class", "regression", []),
}
# target-leakage columns removed before modeling (see paper)
DROP = {"bodyfat": ["Density"], "fat": ["Density"], "failure": ["time"]}
ORDER = ["thyroid", "wisconsin", "recurren", "parkinson", "gallstone", "kidney",
         "heart_disease", "failure", "hepatitis", "bodyfat", "fat"]

# --- repair Excel's auto-date corruption of range labels ("9-May" -> "5-9") ---
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_DATE = re.compile(r"^\s*(\d{1,2})[-/ ]([A-Za-z]{3})\s*$|^\s*([A-Za-z]{3})[-/ ](\d{1,2})\s*$")

def _fix(v):
    if not isinstance(v, str):
        return v
    m = _DATE.match(v)
    if not m:
        return v
    if m.group(1):
        num, mon = int(m.group(1)), m.group(2).lower()
    else:
        mon, num = m.group(3).lower(), int(m.group(4))
    if mon not in _MONTHS:
        return v
    a, b = sorted((num, _MONTHS[mon]))
    return f"{a}-{b}"

def repair_excel_dates(df):
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(_fix)
    return df

def match(fn):
    low = os.path.basename(fn).lower().replace("-", "_").replace(" ", "_")
    for kw in ORDER:
        if kw in low:
            return kw
    return None

def load(path):
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    return repair_excel_dates(df)


def main():
    files = sorted(glob.glob("data/*.csv") + glob.glob("data/*.xlsx"))
    if not files:
        raise SystemExit("No datasets found. Put the CSV/XLSX files in ./data/.")
    os.makedirs("results", exist_ok=True)
    util_rows, priv_rows = [], []
    for f in files:
        kw = match(f)
        if kw is None:
            print(f"  [skip] unrecognized filename: {f}"); continue
        name, target, task, categ = CONFIG[kw]
        df = load(f)
        for c in DROP.get(kw, []):
            if c in df.columns:
                df = df.drop(columns=c)
        categ = [c for c in categ if c in df.columns]
        cfg = T.EvalConfig(task=task, n_target=1000, n_splits=5, n_repeats=5,
                           gen_seeds=(0,), bandwidth=0.25, method="smoothed_bootstrap")
        res = T.evaluate_dataset(df, target, cfg, categorical=categ, dataset_name=name)
        res.to_csv(f"results/{name.replace(' ', '_')}_folds.csv", index=False)
        res.attrs["privacy"].to_csv(f"results/{name.replace(' ', '_')}_privacy.csv", index=False)
        util_rows.append(res); priv_rows.append(res.attrs["privacy"])
        metric = "auroc" if task == "classification" else "rmse"
        m = res.groupby("condition")[metric].mean()
        print(f"{name:28s} {metric} TRTR={m.get('TRTR', np.nan):.3f} "
              f"TSTR={m.get('TSTR', np.nan):.3f}  MIA={res.attrs['privacy']['mia_auc'].mean():.3f}")

    allu = pd.concat(util_rows, ignore_index=True)
    allp = pd.concat(priv_rows, ignore_index=True)
    allu.to_csv("results/all_folds.csv", index=False)
    allp.to_csv("results/all_privacy.csv", index=False)

    # ---- aggregate Table 5 (classification) ----
    clf = allu[allu.dataset != "Body fat"]
    piv = clf.groupby(["dataset", "condition"])[["auroc", "auprc", "f1", "brier"]].mean()
    print("\n=== Table 5 (classification, mean over datasets) ===")
    for metric, higher in [("auroc", True), ("auprc", True), ("f1", True), ("brier", False)]:
        p = piv[metric].unstack()
        ratio = (p["TSTR"] / p["TRTR"]) if higher else (p["TRTR"] / p["TSTR"])
        trs = (p["TR+S"] / p["TRTR"]) if higher else (p["TRTR"] / p["TR+S"])
        print(f"{metric:6s} TRTR={p['TRTR'].mean():.3f} TSTR={p['TSTR'].mean():.3f} "
              f"TSTR/TRTR={ratio.mean():.3f} ({ratio.min():.2f}-{ratio.max():.2f}) "
              f"TR+S/TRTR={trs.mean():.3f}")
    cp = allp[allp.dataset != "Body fat"].groupby("dataset").mia_auc.mean()
    print(f"\nPrivacy: mean classification MIA AUROC = {cp.mean():.3f} "
          f"({cp.min():.2f}-{cp.max():.2f})")
    print("\nPer-fold and privacy CSVs written to ./results/")


if __name__ == "__main__":
    main()
