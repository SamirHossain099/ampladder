"""H4's label timescale tau, per task, from annotations only (PREREGISTRATION.md section 7).

For each Lite session and task, the task's underlying annotation is placed on a 1-second grid over
the recording: the mean of the variable over words starting in that second, or, for indicator tasks,
the fraction of those words in the positive class; for Speech, whether any word starts in that
second. tau is the first lag (s) at which the NaN-aware autocorrelation falls below 1/e, capped at
600 s. A task's tau is its median over the 12 sessions. No neural data is read.

The variable behind each task is taken the way `neuroprobe/datasets.py` (commit b901984) derives
the label, so tau describes the quantity the benchmark actually thresholds:

  pitch            enhanced_pitch, from pitch_volume_features/<movie>.json
  volume           rms                 delta_volume      delta_rms
  frame_brightness mean_pixel_brightness
  global_flow      max_global_magnitude local_flow       max_vector_magnitude
  gpt2_surprisal   gpt2_surprisal      word_length       word_length
  onset            is_onset == 1       face_num          face_num > 0
  word_index       idx_in_sentence == 0 (the positive class; negatives are == 1)
  word_head_pos    bin_head == 0       word_part_speech  pos == "VERB"
  word_gap         start minus previous word's end, within a sentence
  speech           any word starting in the second (negatives are non-verbal chunks)
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
CAP = 600
FS = 2048


def acf_timescale(v, cap=CAP):
    """First lag k >= 1 where the NaN-aware autocorrelation of v drops below 1/e (capped)."""
    v = np.asarray(v, dtype=float)
    for k in range(1, min(cap, len(v) - 2) + 1):
        a, b = v[:-k], v[k:]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 20:
            return float("nan")
        a, b = a[ok], b[ok]
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        if np.corrcoef(a, b)[0, 1] < 1 / np.e:
            return float(k)
    return float(cap)


def word_variable(task, words, pvf):
    if task == "pitch":
        return np.array([pvf[f"{float(t):.5f}"]["enhanced_pitch"] for t in words["start"]], dtype=float)
    simple = {"volume": "rms", "delta_volume": "delta_rms", "frame_brightness": "mean_pixel_brightness",
              "global_flow": "max_global_magnitude", "local_flow": "max_vector_magnitude",
              "gpt2_surprisal": "gpt2_surprisal", "word_length": "word_length"}
    if task in simple:
        return words[simple[task]].to_numpy(dtype=float)
    if task == "onset":
        return (words["is_onset"].to_numpy() == 1).astype(float)
    if task == "face_num":
        return (words["face_num"].to_numpy().astype(int) > 0).astype(float)
    if task == "word_index":
        return (words["idx_in_sentence"].to_numpy().astype(int) == 0).astype(float)
    if task == "word_head_pos":
        return (words["bin_head"].to_numpy().astype(int) == 0).astype(float)
    if task == "word_part_speech":
        return (words["pos"].to_numpy() == "VERB").astype(float)
    if task == "word_gap":
        same = words["sentence"].to_numpy()[1:] == words["sentence"].to_numpy()[:-1]
        gap = words["start"].to_numpy()[1:] - words["end"].to_numpy()[:-1]
        return np.concatenate([[np.nan], np.where(same, gap, np.nan)])
    raise ValueError(task)


def grid(times_s, values, length_s):
    bins = np.floor(times_s).astype(int)
    keep = (bins >= 0) & (bins < length_s) & np.isfinite(values)
    s = np.bincount(bins[keep], weights=values[keep], minlength=length_s)
    c = np.bincount(bins[keep], minlength=length_s)
    with np.errstate(invalid="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--np-root", default="D:/ieeg07/np")
    ap.add_argument("--data-root", default="D:/ieeg07/braintreebank")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "label_timescale"))
    a = ap.parse_args()
    os.environ["ROOT_DIR_BRAINTREEBANK"] = a.data_root
    sys.path.insert(0, a.np_root)
    from neuroprobe.braintreebank_subject import BrainTreebankSubject
    from neuroprobe.config import (BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING, NEUROPROBE_TASKS,
                                   PITCH_VOLUME_FEATURES_DIR)
    from neuroprobe.datasets import BrainTreebankSubjectTrialBenchmarkDataset

    BrainTreebankSubject.load_neural_data = lambda self, *x, **k: None
    rows = []
    for s, t in LITE:
        subj = BrainTreebankSubject(s, cache=False)
        # Build one dataset only to obtain Neuroprobe's merged word table (all_words_df).
        ds = BrainTreebankSubjectTrialBenchmarkDataset(subj, t, dtype=None, eval_name="gpt2_surprisal",
                                                       output_indices=True, output_dict=False)
        words = ds.all_words_df.sort_values("est_idx").reset_index(drop=True)
        movie = BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING[f"btbank{s}_{t}"]
        raw = json.load(open(os.path.join(PITCH_VOLUME_FEATURES_DIR, f"{movie}_pitch_volume_features.json")))
        pvf = {f"{float(k):.5f}": v for k, v in raw.items()}
        times = words["est_idx"].to_numpy() / FS
        length = int(np.ceil(times.max())) + 2
        for task in NEUROPROBE_TASKS:
            if task == "speech":
                g = grid(times, np.ones(len(times)), length)
                g = np.where(np.isfinite(g), 1.0, 0.0)
            else:
                g = grid(times, word_variable(task, words, pvf), length)
            rows.append(dict(subject=s, trial=t, task=task, tau_s=acf_timescale(g),
                             grid_seconds=length, finite_seconds=int(np.isfinite(g).sum())))
        print(f"{s}_{t} done", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "tau_by_session.csv"), index=False)
    med = df.groupby("task").tau_s.median().sort_values()
    med.to_csv(os.path.join(a.out, "tau_by_task.csv"), header=["tau_s_median"])
    print(med.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
