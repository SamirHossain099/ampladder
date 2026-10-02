"""D1 step 1: score every electrode of a session the way Neuroprobe Lite's selection did.

PREREGISTRATION.md D1. The shipped rule read `linear_voltage_single_electrode` results: one
electrode at a time, raw voltage (no Laplacian), within-session, linear classifier, test AUROC,
averaged over tasks, time bins and folds. This script computes the stated stand-in for each Lite
session: the same, on the one-second bin and Neuroprobe Lite's two within-session folds, for **every**
electrode of the subject that Neuroprobe keeps (corrupted, trigger and coordinate-less electrodes are
excluded by its own subject class), not only the Lite ones.

Output: results/d1/scores_s{S}_t{T}.csv, one row per (electrode, task, fold), plus the per-electrode
mean that the selection rule consumes.

Run with IEEG07_THREADS=1: each worker fits one small logistic regression at a time, so parallelism
comes from processes, not BLAS threads.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402

import h5py  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 2048


def _work(args):
    h5_path, keys_labels, wf, groups = args
    import h5py
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    rows = []
    with h5py.File(h5_path, "r") as f:
        for key, label in keys_labels:
            # float32 at once, then windows through a strided view: no int64 index grid (181 MB for a
            # long session) and no float64 copy kept alive. Same values as before: Neuroprobe casts to
            # float32 before slicing too. Memory per worker roughly halves, which matters when other
            # work shares the machine (an OOM on a 205-electrode subject on 2026-10-01).
            trace = f["data"][key][:].astype(np.float32)
            X_all = np.lib.stride_tricks.sliding_window_view(trace, N)[wf].copy()   # (n_windows, 2048)
            del trace
            for (task, fold), (tr_pos, ytr, te_pos, yte) in groups.items():
                Xtr = np.array(X_all[tr_pos])
                Xte = np.array(X_all[te_pos])
                sc = StandardScaler(copy=False)
                Xtr = sc.fit_transform(Xtr)
                Xte = sc.transform(Xte)
                clf = LogisticRegression(random_state=42, max_iter=10000, tol=1e-3).fit(Xtr, ytr)
                auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
                rows.append(dict(electrode=label, task=task, fold=fold, test_auroc=float(auc)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--trial", type=int, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--np-root", default="D:/ieeg07/np")
    ap.add_argument("--data-root", default="D:/ieeg07/braintreebank")
    ap.add_argument("--index", default=os.path.join(HERE, "results", "index", "neuroprobe_lite_index.csv.gz"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "d1"))
    ap.add_argument("--max-electrodes", type=int, default=None, help="testing only")
    a = ap.parse_args()
    if resources.THREADS != 1:
        print(f"warning: BLAS threads = {resources.THREADS}; set IEEG07_THREADS=1 for this script", flush=True)

    os.environ["ROOT_DIR_BRAINTREEBANK"] = a.data_root
    sys.path.insert(0, a.np_root)
    from neuroprobe.braintreebank_subject import BrainTreebankSubject

    subj = BrainTreebankSubject(a.subject, allow_corrupted=False, cache=False)
    labels = list(subj.electrode_labels)
    if a.max_electrodes:
        labels = labels[: a.max_electrodes]
    keys = [(subj.h5_neural_data_keys[e], e) for e in labels]

    idx = pd.read_csv(a.index)
    idx = idx[(idx.split == "within") & (idx.test_subject == a.subject) & (idx.test_trial == a.trial)]
    wf = np.sort(idx.window_from.unique()).astype(np.int64)
    pos = {w: i for i, w in enumerate(wf)}
    groups = {}
    for (task, fold), g in idx.groupby(["task", "fold"]):
        tr = g[g.role == "train"].sort_values("item")
        te = g[g.role == "test"].sort_values("item")
        groups[(task, fold)] = (np.array([pos[w] for w in tr.window_from]), tr.label.to_numpy(),
                                np.array([pos[w] for w in te.window_from]), te.label.to_numpy())

    h5_path = os.path.join(a.data_root, f"sub_{a.subject}_trial{a.trial:03}.h5")
    with h5py.File(h5_path, "r") as f:
        length = f["data"][keys[0][0]].shape[0]
    # Per worker at peak: float64 read + float32 trace + window block + one fold's train/test copies.
    per_worker = length * (8 + 4) + len(wf) * N * 4 * 2
    resources.require_ram(per_worker * a.workers, f"{a.workers} scoring workers", headroom=0.7)
    chunks = [keys[i::a.workers] for i in range(a.workers) if keys[i::a.workers]]
    t0 = time.time()
    print(f"s{a.subject}_t{a.trial}: {len(keys)} electrodes x {len(groups)} task-folds, "
          f"{len(wf)} windows, {len(chunks)} workers", flush=True)
    with mp.get_context("spawn").Pool(len(chunks)) as pool:
        parts = pool.map(_work, [(h5_path, c, wf, groups) for c in chunks])
    df = pd.DataFrame([r for p in parts for r in p])
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, f"scores_s{a.subject}_t{a.trial}.csv"), index=False)
    mean = df.groupby("electrode").test_auroc.mean()
    mean.to_csv(os.path.join(a.out, f"electrode_mean_s{a.subject}_t{a.trial}.csv"), header=["mean_auroc"])
    print(f"done in {time.time() - t0:.0f} s; electrode mean AUROC median {mean.median():.4f}, "
          f"max {mean.max():.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
