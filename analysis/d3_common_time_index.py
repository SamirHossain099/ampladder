"""Deviation D3 (POST HOC): within-session folds cut at one common time for both classes.

PREREGISTRATION.md, D3. Neuroprobe cuts each class at its own time quantile (classes are sampled
separately, sorted in time, interleaved, then split by index), so scored positives and negatives can
occupy different stretches of a recording. This builds the alternative: the same items -- same
windows, same labels -- with folds cut at one time boundary shared by both classes.

For each within-session (session, task):
  * items = every item of that dataset (taken from Neuroprobe's own fold-0 rows: train + val + test);
  * sort by window start; fold A = items before the median start time, fold B = the rest;
  * fold 0 holds out A (as KFold's first fold holds out the first half); fold 1 holds out B;
  * the held-out fold is halved at its own median time: earlier half `val`, later half `test`
    (Neuroprobe scores the second half of the held-out fold; this mirrors it in time).
Rows keep the index schema, so `decode.py --index <this file> --splits within` runs unchanged.
`item` is time order. Output: results/d3/index_common_time.csv.gz.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def common_time_folds(items):
    """items: DataFrame of one dataset's items (window_from, label, ...). Returns rows with fold/role."""
    it = items.sort_values(["window_from", "base_index"]).reset_index(drop=True)
    t = it.window_from.to_numpy()
    cut = np.median(t)
    A = it[t < cut]
    B = it[t >= cut]
    out = []
    for fold, (held, train) in enumerate(((A, B), (B, A))):
        h = held.sort_values("window_from")
        hcut = np.median(h.window_from.to_numpy())
        val = h[h.window_from < hcut]
        test = h[h.window_from >= hcut]
        for role, part in (("train", train), ("val", val), ("test", test)):
            p = part.sort_values("window_from").copy()
            p["fold"] = fold
            p["role"] = role
            p["item"] = np.arange(len(p))
            out.append(p)
    return pd.concat(out, ignore_index=True)


def main():
    src = os.path.join(HERE, "results", "index", "neuroprobe_lite_index.csv.gz")
    out_dir = os.path.join(HERE, "results", "d3")
    idx = pd.read_csv(src)
    w = idx[(idx.split == "within") & (idx.fold == 0)]
    rows = []
    for (s, t, task), g in w.groupby(["test_subject", "test_trial", "task"]):
        items = g.drop_duplicates("base_index")[["split", "test_subject", "test_trial", "task", "src_subject",
                                                "src_trial", "base_index", "window_from", "window_to", "label"]]
        n_expected = g.base_index.nunique()
        f = common_time_folds(items)
        assert f[f.fold == 0].base_index.nunique() == n_expected, "every item must appear in fold 0"
        rows.append(f)
    df = pd.concat(rows, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "index_common_time.csv.gz"), index=False)
    bal = df[df.role == "test"].groupby(["test_subject", "test_trial", "task", "fold"]).label.nunique()
    print(f"{len(df)} rows; scored test sets with both classes: {int((bal == 2).sum())} of {len(bal)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
