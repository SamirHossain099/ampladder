"""Audit a trial-based benchmark's folds for time leakage, from its index alone.

No neural data is needed. The input is one row per (cell, fold, role, item) with the item's start time
and label -- what any decoding benchmark can export -- and the functions answer three questions:

* single_class_share: what fraction of scored test items sit in a stretch of the recording that holds
  only one class? There, anything that drifts slowly over a session separates the classes.
* train_test_adjacency: what fraction of scored test items have a training item starting within t
  seconds? Within the window length they share raw samples.
* common_time_folds: rebuild the folds so both classes are cut at one shared time, to test whether a
  result survives once the construction cannot leak time.

Written for, and validated on, Neuroprobe Lite (Zahorodnii et al. 2025), whose within-session folds cut
each class at its own time quantile; see the accompanying paper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CELL = ["test_subject", "test_trial", "task", "fold"]


def _scored(g):
    return g[g.role == "test"]


def single_class_share(index: pd.DataFrame, time_col: str = "window_from", cell=CELL) -> pd.DataFrame:
    """Per cell: share of scored test items lying outside the time span both classes occupy."""
    rows = []
    for key, g in index.groupby(cell):
        te = _scored(g)
        t1, t0 = te.loc[te.label == 1, time_col], te.loc[te.label == 0, time_col]
        if t1.empty or t0.empty:
            raise ValueError(f"cell {key}: scored set lacks a class")
        lo, hi = max(t1.min(), t0.min()), min(t1.max(), t0.max())
        t = te[time_col]
        rows.append(dict(zip(cell, key), single_class_share=float(((t < lo) | (t > hi)).mean())))
    return pd.DataFrame(rows)


def train_test_adjacency(index: pd.DataFrame, within_seconds=(1.0, 5.0, 30.0), fs: float = 2048.0,
                         time_col: str = "window_from", cell=CELL) -> pd.DataFrame:
    """Per cell: share of scored test items with a training item starting within each horizon."""
    rows = []
    for key, g in index.groupby(cell):
        te = np.sort(_scored(g)[time_col].to_numpy()) / fs
        tr = np.sort(g.loc[g.role == "train", time_col].to_numpy()) / fs
        if len(tr) == 0 or len(te) == 0:
            raise ValueError(f"cell {key}: empty train or test")
        pos = np.searchsorted(tr, te)
        left = np.abs(te - tr[np.clip(pos - 1, 0, len(tr) - 1)])
        right = np.abs(tr[np.clip(pos, 0, len(tr) - 1)] - te)
        near = np.minimum(left, right)
        rows.append({**dict(zip(cell, key)), **{f"within_{w:g}s": float((near < w).mean()) for w in within_seconds}})
    return pd.DataFrame(rows)


def common_time_folds(items: pd.DataFrame, time_col: str = "window_from", id_col: str = "base_index") -> pd.DataFrame:
    """Two folds cut at one median time shared by both classes; the held-out fold is halved in time
    and its later half scored (mirroring a scheme that scores the second half of the held-out fold).
    `items` is every item of one (session, task) dataset; returns rows with fold, role and item order.
    """
    it = items.sort_values([time_col, id_col]).reset_index(drop=True)
    t = it[time_col].to_numpy()
    cut = np.median(t)
    A, B = it[t < cut], it[t >= cut]
    out = []
    for fold, (held, train) in enumerate(((A, B), (B, A))):
        h = held.sort_values(time_col)
        hcut = np.median(h[time_col].to_numpy())
        for role, part in (("train", train), ("val", h[h[time_col] < hcut]), ("test", h[h[time_col] >= hcut])):
            p = part.sort_values(time_col).copy()
            p["fold"], p["role"], p["item"] = fold, role, np.arange(len(p))
            out.append(p)
    return pd.concat(out, ignore_index=True)
