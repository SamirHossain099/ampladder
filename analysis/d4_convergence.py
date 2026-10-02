"""Deviation D4 (POST HOC): does the shared classifier converge, and do the conclusions survive if it does?

PREREGISTRATION.md, D4. For every within-session cell and rungs L1-L5, refit on the same data at
the benchmark's tol = 1e-3 (must reproduce the primary AUROC: a determinism check) and at tol = 1e-6,
max_iter = 10000, recording AUROC and lbfgs iterations for both. Then:
  * iterations at 1e-3 per rung;
  * overall within-session AUROC, converged minus default, per rung (hierarchical bootstrap);
  * H1 (L5 - L4) and H3 (L2 - BrainBERT per-window) on converged fits.
Appends per cell to results/d4/convergence.csv, so it resumes after an interruption.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from analyze import hier_boot, load_leaderboard, overall  # noqa: E402
from decode import Session, neuroprobe_auroc  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNGS = ("L1", "L2", "L3", "L4", "L5")
TOLS = (1e-3, 1e-6)


def fit(Xtr, ytr, Xte, yte, tol):
    # Mirror decode.fit_score exactly (dtype kept, scaler fit on train) so the tol = 1e-3 refit can
    # reproduce the primary AUROC: an early-stopped solver lands elsewhere if float32 becomes float64.
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(random_state=42, max_iter=10000, tol=tol).fit(Xtr, ytr)
    return neuroprobe_auroc(clf, Xte, yte), int(clf.n_iter_[0])


def run(a):
    idx = pd.read_csv(a.index)
    idx = idx[idx.split == "within"]
    out = os.path.join(a.out, "convergence.csv")
    done = set()
    if os.path.exists(out):
        prev = pd.read_csv(out)
        done = set(map(tuple, prev[["test_subject", "test_trial", "task", "fold"]].itertuples(index=False, name=None)))
    cache = {}
    t0 = time.time()
    groups = list(idx.groupby(["test_subject", "test_trial", "task", "fold"], sort=True))
    for n, (key, g) in enumerate(groups):
        if key in done:
            continue
        s, t = key[0], key[1]
        if (s, t) not in cache:
            cache.clear()                       # one session's L5 memmap at a time
            cache[(s, t)] = Session(a.features, s, t)
        ses = cache[(s, t)]
        tr = g[g.role == "train"].sort_values("item")
        te = g[g.role == "test"].sort_values("item")
        rows = []
        for rung in RUNGS:
            Xtr = ses.features(rung, tr.window_from.to_numpy()).reshape(len(tr), -1)
            Xte = ses.features(rung, te.window_from.to_numpy()).reshape(len(te), -1)
            row = dict(test_subject=s, test_trial=t, task=key[2], fold=key[3], rung=rung, n_features=Xtr.shape[1])
            for tol in TOLS:
                auc, it = fit(Xtr, tr.label.to_numpy(), Xte, te.label.to_numpy(), tol)
                row[f"auroc_{tol:.0e}"] = auc
                row[f"iter_{tol:.0e}"] = it
            rows.append(row)
        pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        if n % 20 == 0:
            print(f"{n + 1}/{len(groups)} {key} ({time.time() - t0:.0f} s)", flush=True)


def analyze(a):
    d = pd.read_csv(os.path.join(a.out, "convergence.csv"))
    res = {"note": "POST HOC (D4)", "iterations_at_1e-3": {}, "converged_minus_default": {}, "determinism": {}}
    for rung, g in d.groupby("rung"):
        it = g["iter_1e-03"]
        res["iterations_at_1e-3"][rung] = dict(median=float(it.median()), min=int(it.min()), max=int(it.max()))
    # Determinism: tol 1e-3 refits must equal the primary decode, cell by cell.
    for rung in RUNGS:
        p = pd.read_csv(os.path.join(HERE, "results", "decode", f"{rung}.csv"))
        p = p[p.split == "within"][["test_subject", "test_trial", "task", "fold", "test_auroc"]]
        m = d[d.rung == rung].merge(p, on=["test_subject", "test_trial", "task", "fold"])
        res["determinism"][rung] = dict(cells=int(len(m)), max_abs_diff=float((m["auroc_1e-03"] - m.test_auroc).abs().max()))

    def cells(col, name):
        c = (d.groupby(["rung", "test_subject", "test_trial", "task"])[col].mean().reset_index()
             .rename(columns={"rung": "model", col: "auroc"}))
        c["split"] = "within"
        c["model"] = c["model"] + name
        return c

    allc = pd.concat([cells("auroc_1e-03", ""), cells("auroc_1e-06", "_conv"),
                      load_leaderboard(a.leaderboard)], ignore_index=True)
    for rung in RUNGS:
        x, y = overall(allc, rung + "_conv", "within"), overall(allc, rung, "within")
        common = x.index.intersection(y.index)
        m, lo, hi = hier_boot(x[common] - y[common], f"D4_{rung}")
        res["converged_minus_default"][rung] = dict(mean=m, ci_lo=lo, ci_hi=hi, default=float(y[common].mean()),
                                                    converged=float(x[common].mean()))
    for name, (p, q) in {"H1_conv_L5_minus_L4": ("L5_conv", "L4_conv"),
                         "H3_conv_L2_minus_brainbert_perwindow": ("L2_conv", "brainbert_perwindow")}.items():
        x, y = overall(allc, p, "within"), overall(allc, q, "within")
        common = x.index.intersection(y.index)
        m, lo, hi = hier_boot(x[common] - y[common], f"D4_{name}")
        res[name] = dict(mean=m, ci_lo=lo, ci_hi=hi)
        if name.startswith("H3"):
            res[name]["equivalent_within_0.02"] = bool(lo > -0.02 and hi < 0.02)
    with open(os.path.join(a.out, "d4_results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(HERE, "results", "index", "neuroprobe_lite_index.csv.gz"))
    ap.add_argument("--features", default="D:/ieeg07/features")
    ap.add_argument("--leaderboard", default="D:/ieeg07/np/leaderboard")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "d4"))
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not a.analyze_only:
        run(a)
    analyze(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
