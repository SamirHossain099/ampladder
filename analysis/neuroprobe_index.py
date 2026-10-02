"""Record every Neuroprobe Lite window, label and fold, using Neuroprobe's own classes.

PREREGISTRATION.md section 2: windows, labels and folds are the benchmark's, not this project's.
This script asks Neuroprobe for them -- `generate_splits_*` with `output_indices=True` -- and writes
one row per (split, evaluation, fold, role, item). Feature extraction then reads exactly those
windows, so no window boundary or fold assignment is ever re-implemented here.

Two details of the upstream code this depends on, verified by reading it (commit b901984):
* With `output_indices=True` the dataset still calls `subject.load_neural_data`, which opens the
  HDF5 file. Only indices are needed, so that call is replaced by a no-op here; this lets the index
  be built before all 75 GB have downloaded. The index cannot change as a result: the indices come
  from annotation tables shipped with the package, never from the neural data.
* `eval_population.py` scores `fold["test_dataset"]`, the *second* half of each held-out fold;
  `val_dataset` is unused by the linear pipeline. Both roles are recorded, and `role` says which.

Output: results/index/neuroprobe_lite_index.csv.gz and a summary JSON with row counts and the number
of unique windows per session (the unit of feature extraction).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LITE = [(1, 1), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1), (4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1)]
CROSS_SUBJECT_TRAIN = (2, 4)


def unwrap(ds):
    """Follow Subset.dataset links down to the BrainTreebankSubjectTrialBenchmarkDataset, composing
    the index maps on the way, so each item can be attributed to its source session."""
    idx = None
    while hasattr(ds, "indices") and hasattr(ds, "dataset"):
        here = list(ds.indices)
        idx = here if idx is None else [here[i] for i in idx]
        ds = ds.dataset
    if idx is None:
        idx = list(range(len(ds)))
    return ds, idx


def rows_for(ds_wrapped, split, test_session, task, fold, role):
    base, idx = unwrap(ds_wrapped)
    out = []
    for order, i in enumerate(idx):
        (w_from, w_to), label = base.__getitem__(i, force_output_indices=True)
        out.append(dict(split=split, test_subject=test_session[0], test_trial=test_session[1],
                        task=task, fold=fold, role=role, src_subject=base.subject_id,
                        src_trial=base.trial_id, item=order, base_index=i,
                        window_from=int(w_from), window_to=int(w_to), label=int(label)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--np-root", default="D:/ieeg07/np")
    ap.add_argument("--data-root", default="D:/ieeg07/braintreebank")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "index"))
    ap.add_argument("--tasks", nargs="*", default=None)
    a = ap.parse_args()
    os.environ["ROOT_DIR_BRAINTREEBANK"] = a.data_root
    sys.path.insert(0, a.np_root)

    import torch  # noqa: F401  (neuroprobe imports it; imported here to keep the order explicit)
    from neuroprobe import train_test_splits as tts
    from neuroprobe.braintreebank_subject import BrainTreebankSubject
    from neuroprobe.config import NEUROPROBE_LITE_ELECTRODES, NEUROPROBE_TASKS

    # Indices only: never open the neural data file.
    BrainTreebankSubject.load_neural_data = lambda self, *args, **kwargs: None

    tasks = a.tasks or NEUROPROBE_TASKS
    subjects = {}

    def subject(sid):
        if sid not in subjects:
            s = BrainTreebankSubject(sid, allow_corrupted=False, cache=False)
            s.set_electrode_subset(NEUROPROBE_LITE_ELECTRODES[s.subject_identifier])
            subjects[sid] = s
        return subjects[sid]

    rows = []
    for task in tasks:
        for ses in LITE:
            sub = subject(ses[0])
            folds = tts.generate_splits_within_session(sub, ses[1], task, output_indices=True,
                                                       output_dict=False, lite=True)
            for k, f in enumerate(folds):
                rows += rows_for(f["train_dataset"], "within", ses, task, k, "train")
                rows += rows_for(f["val_dataset"], "within", ses, task, k, "val")
                rows += rows_for(f["test_dataset"], "within", ses, task, k, "test")

            folds = tts.generate_splits_cross_session(sub, ses[1], task, output_indices=True,
                                                      output_dict=False, lite=True)
            for k, f in enumerate(folds):
                rows += rows_for(f["train_dataset"], "cross_session", ses, task, k, "train")
                rows += rows_for(f["val_dataset"], "cross_session", ses, task, k, "val")
                rows += rows_for(f["test_dataset"], "cross_session", ses, task, k, "test")

            if ses[0] != CROSS_SUBJECT_TRAIN[0]:
                allsub = {ses[0]: sub, CROSS_SUBJECT_TRAIN[0]: subject(CROSS_SUBJECT_TRAIN[0])}
                folds = tts.generate_splits_cross_subject(allsub, ses[0], ses[1], task,
                                                          output_indices=True, output_dict=False,
                                                          lite=True)
                for k, f in enumerate(folds):
                    rows += rows_for(f["train_dataset"], "cross_subject", ses, task, k, "train")
                    rows += rows_for(f["val_dataset"], "cross_subject", ses, task, k, "val")
                    rows += rows_for(f["test_dataset"], "cross_subject", ses, task, k, "test")
        print(f"{task}: {len(rows)} rows so far", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "neuroprobe_lite_index.csv.gz")
    df.to_csv(path, index=False)

    # Guards that must hold if the index means what it claims.
    assert (df.window_to - df.window_from == 2048).all(), "every window must be exactly 1 s"
    bal = df[df.role == "test"].groupby(["split", "test_subject", "test_trial", "task", "fold"]).label.nunique()
    assert (bal == 2).all(), "every scored test set must contain both classes"

    uniq = (df[["src_subject", "src_trial", "window_from"]].drop_duplicates()
            .groupby(["src_subject", "src_trial"]).size())
    summary = dict(rows=int(len(df)), tasks=list(tasks),
                   rows_by_split_role=df.groupby(["split", "role"]).size().to_dict().__repr__(),
                   unique_windows_per_session={f"{s}_{t}": int(n) for (s, t), n in uniq.items()},
                   unique_windows_total=int(uniq.sum()),
                   electrodes_per_subject={k: len(v) for k, v in NEUROPROBE_LITE_ELECTRODES.items()})
    with open(os.path.join(a.out, "index_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
