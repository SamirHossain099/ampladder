"""Decode every rung of the ladder with Neuroprobe's own classifier, on Neuroprobe's own splits.

PREREGISTRATION.md sections 4-6. For each (split, test session, task, fold) the train and test rows
come from `results/index/neuroprobe_lite_index.csv.gz` -- the windows, labels and fold roles that
Neuroprobe's `generate_splits_*` produced -- and every rung is scored the same way:

    StandardScaler(copy=False) fit on train  ->  LogisticRegression(random_state=42,
    max_iter=10000, tol=1e-3)  ->  test AUROC, computed in eval_population.py's one-hot form.

Rows enter the classifier in the dataset's iteration order (`item`), as eval_population.py builds
them, because the solver's floating-point sums depend on order; this is what lets L5 reproduce the
leaderboard cell by cell rather than approximately.

Cross-subject: per-electrode features are averaged within Desikan-Killiany regions shared by the
two subjects using Neuroprobe's `combine_regions`, as the published baseline does.

**NaN policy: refuse.** The pre-registration does not specify imputation. A NaN feature (a
zero-variance window has no log variance and no exponent) stops the run with a count, so a policy can
be chosen and logged as a deviation before any decoding outcome exists. It never silently imputes.

Output: results/decode/{rung}.csv, one row per (split, test session, task, fold).
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNGS = ("L0", "L1", "L2", "L3", "L4", "L5")
# S2 (PREREGISTRATION.md section 8): all eight exponents, on plateau-free windows only.
S2_RUNG = "L2neg"
SMALLEST_SCALE = 8
N_SAMPLES = 2048
Q_ALL = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0])
Q_PRIMARY = np.isin(Q_ALL, [1.0, 2.0, 3.0, 5.0])          # PREREGISTRATION.md section 4


class Session:
    """Features of one session, addressed by window start sample."""

    def __init__(self, feat_dir, subject, trial):
        tag = f"s{subject}_t{trial}"
        self.meta = json.load(open(os.path.join(feat_dir, f"{tag}_meta.json"), encoding="utf-8"))
        self.wf = np.load(os.path.join(feat_dir, f"{tag}_window_from.npy"))
        self._dir, self._tag = feat_dir, tag
        flat = os.path.join(feat_dir, f"{tag}_flat.npy")
        self.flat = np.load(flat) if os.path.exists(flat) else None
        self.labels = self.meta["electrode_labels"]
        self.subject, self.trial = subject, trial

    # Arrays load on first use, so a feature directory may hold only what its analysis needs
    # (S1's directories hold exponents only). A missing array fails when it is asked for.
    @property
    def L1(self):
        if not hasattr(self, "_L1"):
            self._L1 = np.load(os.path.join(self._dir, f"{self._tag}_L1.npy"))
        return self._L1

    @property
    def hq(self):
        if not hasattr(self, "_hq"):
            self._hq = np.load(os.path.join(self._dir, f"{self._tag}_hq.npy"))
        return self._hq

    @property
    def L5(self):
        if not hasattr(self, "_L5"):
            self._L5 = np.load(os.path.join(self._dir, f"{self._tag}_L5.npy"), mmap_mode="r")
        return self._L5

    def rows(self, window_from):
        pos = np.searchsorted(self.wf, window_from)
        if (pos >= len(self.wf)).any() or (self.wf[np.minimum(pos, len(self.wf) - 1)] != window_from).any():
            raise KeyError(f"window not extracted for session {self.subject}/{self.trial}")
        return pos

    def features(self, rung, window_from):
        r = self.rows(window_from)
        if rung == "L0":
            return (np.asarray(window_from, dtype=np.float64) / 2048.0)[:, None, None]
        if rung == "L1":
            return self.L1[r][:, :, None]
        if rung == "L2":
            return self.hq[r][:, :, Q_PRIMARY]
        if rung == "L3":
            return np.concatenate([self.L1[r][:, :, None], self.hq[r][:, :, Q_PRIMARY]], axis=2)
        if rung == "L5":
            return np.asarray(self.L5[r], dtype=np.float32)
        if rung == S2_RUNG:
            return self.hq[r]                                   # q = -3, -2, -1, 0, 1, 2, 3, 5
        if rung == "L4":
            x = np.asarray(self.L5[r], dtype=np.float64)
            s = x.sum(axis=(2, 3), keepdims=True)
            if (s <= 0).any():
                raise ValueError("an electrode-window with zero spectral mass cannot be shape-normalised")
            return (x / s).astype(np.float32)
        raise ValueError(rung)

    def not_constant(self, window_from):
        """True where no electrode is constant across the whole window (S3 on raw voltage, D5)."""
        if self.flat is None:
            raise FileNotFoundError("flat-run QC file missing; --exclude-constant needs it")
        return self.flat[self.rows(window_from)].max(axis=1) < N_SAMPLES

    def plateau_free(self, window_from):
        """True where no electrode has a constant run reaching the smallest scale (S2's QC)."""
        if self.flat is None:
            raise FileNotFoundError("flat-run QC file missing; S2 cannot be run without it")
        return self.flat[self.rows(window_from)].max(axis=1) < SMALLEST_SCALE


def region_labels(np_root, data_root, subject, labels):
    os.environ["ROOT_DIR_BRAINTREEBANK"] = data_root
    sys.path.insert(0, os.path.join(np_root, "examples"))
    sys.path.insert(0, np_root)
    from neuroprobe.braintreebank_subject import BrainTreebankSubject
    s = BrainTreebankSubject(subject, allow_corrupted=False, cache=False)
    s.set_electrode_subset(labels)
    return s.get_all_electrode_metadata()["DesikanKilliany"].to_numpy()


def neuroprobe_auroc(clf, X, y):
    """eval_population.py's one-hot AUROC, kept literally."""
    probs = clf.predict_proba(X)
    keep = np.isin(y, clf.classes_)
    y, probs = y[keep], probs[keep]
    onehot = np.zeros((len(y), len(clf.classes_)))
    for i, lab in enumerate(y):
        onehot[i, np.where(clf.classes_ == lab)[0][0]] = 1
    if len(clf.classes_) > 2:
        return roc_auc_score(onehot, probs, multi_class="ovr", average="macro")
    return roc_auc_score(onehot, probs)


def fit_score(Xtr, ytr, Xte, yte):
    Xtr = Xtr.reshape(len(Xtr), -1)
    Xte = Xte.reshape(len(Xte), -1)
    for name, X in (("train", Xtr), ("test", Xte)):
        bad = ~np.isfinite(X)
        if bad.any():
            raise ValueError(f"{int(bad.sum())} non-finite {name} feature values; NaN policy is refuse")
    scaler = StandardScaler(copy=False)
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)
    clf = LogisticRegression(random_state=42, max_iter=10000, tol=1e-3)
    clf.fit(Xtr, ytr)
    return neuroprobe_auroc(clf, Xte, yte), neuroprobe_auroc(clf, Xtr, ytr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="+", default=list(RUNGS))
    ap.add_argument("--splits", nargs="+", default=["within", "cross_session", "cross_subject"])
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--sessions", nargs="*", default=None, help="e.g. 1_1 2_0 (test sessions)")
    ap.add_argument("--index", default=os.path.join(HERE, "results", "index", "neuroprobe_lite_index.csv.gz"))
    ap.add_argument("--features", default="D:/ieeg07/features")
    ap.add_argument("--np-root", default="D:/ieeg07/np")
    ap.add_argument("--data-root", default="D:/ieeg07/braintreebank")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "decode"))
    ap.add_argument("--exclude-constant", action="store_true",
                    help="drop windows in which any electrode is constant for the whole window, from train "
                         "and test, for every rung, counted in n_excluded_plateau (S3 on raw voltage, D5)")
    a = ap.parse_args()

    idx = pd.read_csv(a.index)
    idx = idx[idx.split.isin(a.splits)]
    if a.tasks:
        idx = idx[idx.task.isin(a.tasks)]
    if a.sessions:
        keep = {tuple(map(int, s.split("_"))) for s in a.sessions}
        idx = idx[[(s, t) in keep for s, t in zip(idx.test_subject, idx.test_trial)]]

    sys.path.insert(0, os.path.join(a.np_root, "examples"))
    os.environ["ROOT_DIR_BRAINTREEBANK"] = a.data_root
    from eval_utils import combine_regions

    cache, regions = {}, {}

    def session(s, t):
        if (s, t) not in cache:
            cache[(s, t)] = Session(a.features, s, t)
        return cache[(s, t)]

    def regions_of(ses):
        k = (ses.subject, ses.trial)
        if k not in regions:
            regions[k] = region_labels(a.np_root, a.data_root, ses.subject, ses.labels)
        return regions[k]

    os.makedirs(a.out, exist_ok=True)
    keys = ["split", "test_subject", "test_trial", "task", "fold"]
    groups = list(idx.groupby(keys, sort=True))
    t0 = time.time()
    for rung in a.rungs:
        out_path = os.path.join(a.out, f"{rung}.csv")
        done = set()
        if os.path.exists(out_path):
            prev = pd.read_csv(out_path)
            done = set(map(tuple, prev[keys].itertuples(index=False, name=None)))
        for n, (key, g) in enumerate(groups):
            if key in done:
                continue
            tr = g[g.role == "train"].sort_values("item")
            te = g[g.role == "test"].sort_values("item")
            (trs, trt), = tr[["src_subject", "src_trial"]].drop_duplicates().itertuples(index=False, name=None)
            (tes, tet), = te[["src_subject", "src_trial"]].drop_duplicates().itertuples(index=False, name=None)
            str_, ste = session(trs, trt), session(tes, tet)
            n_excl = 0
            if rung == S2_RUNG:
                ktr, kte = str_.plateau_free(tr.window_from.to_numpy()), ste.plateau_free(te.window_from.to_numpy())
                n_excl = int((~ktr).sum() + (~kte).sum())
                tr, te = tr[ktr], te[kte]
            elif a.exclude_constant:
                ktr, kte = str_.not_constant(tr.window_from.to_numpy()), ste.not_constant(te.window_from.to_numpy())
                n_excl = int((~ktr).sum() + (~kte).sum())
                tr, te = tr[ktr], te[kte]
            Xtr = str_.features(rung, tr.window_from.to_numpy())
            Xte = ste.features(rung, te.window_from.to_numpy())
            if key[0] == "cross_subject" and rung != "L0":
                Xtr, Xte, _ = combine_regions(Xtr, Xte, regions_of(str_), regions_of(ste))
            auc, train_auc = fit_score(Xtr, tr.label.to_numpy(), Xte, te.label.to_numpy())
            row = dict(zip(keys, key), rung=rung, n_train=len(tr), n_test=len(te), n_features=int(np.prod(Xtr.shape[1:])),
                       test_auroc=auc, train_auroc=train_auc, n_excluded_plateau=n_excl)
            pd.DataFrame([row]).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)
            if n % 25 == 0:
                print(f"{rung} {n + 1}/{len(groups)} {key} auc={auc:.4f} ({time.time() - t0:.0f} s)", flush=True)
    print("=== done ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
