"""Deviation D1: how much does selecting electrodes on the evaluated session inflate the benchmark?

PREREGISTRATION.md, D1. Neuroprobe Lite's electrodes were chosen by ranking contiguous probe runs by
their mean single-electrode test AUROC, measured on the very sessions the benchmark scores (FINDINGS
F7). D1 rebuilds that rule twice per session -- once scored on the evaluated session (circular),
once scored only on its partner session from the same subject (cross-fitted) -- and compares the
two subsets on the ladder.

The selection rule is ported VERBATIM from `analyses/select_neuroprobe_electrode_subsets.ipynb`
(Neuroprobe commit b901984, cell 3: `stem_electrode_name`, `find_streaks`, the streak ranking and
`select_electrodes`), so a difference between the subsets can only come from what the scores were
measured on. `tests/test_d1_selection.py` pins the port's behaviour.

The scoring is the stated stand-in (D1, "Stated limits"): within-session, single-electrode,
linear-on-raw-voltage test AUROC on the one-second bin with Neuroprobe Lite's two folds, averaged over
the 15 tasks and both folds -- where the shipped rule used 15 time bins and 5 folds.
"""
from __future__ import annotations

import numpy as np

LITE_N = 120


# --- Ported verbatim from select_neuroprobe_electrode_subsets.ipynb, cell 3 -----------------------
def stem_electrode_name(name):
    # names look like 'O1aIb4', 'O1aIb5', 'O1aIb6', 'O1aIb7'
    # names look like 'T1b2
    found_stem_end = False
    stem, num = [], []
    for c in reversed(name):
        if c.isalpha():
            found_stem_end = True
        if found_stem_end:
            stem.append(c)
        else:
            num.append(c)
    return ''.join(reversed(stem)), int(''.join(reversed(num)))


def find_streaks(nums):
    if not nums:
        return []
    nums = sorted(nums)
    streaks = []
    current_streak = [nums[0]]
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:
            current_streak.append(nums[i])
        else:
            if len(current_streak) > 0:
                streaks.append(current_streak)
            current_streak = [nums[i]]
    streaks.append(current_streak)
    return streaks


def select_electrodes_from_scores(electrode_labels, scores, n_electrodes=LITE_N):
    """The notebook's procedure, with `subject_electrode_neuroprobe_mean_performance[subject_id]`
    replaced by `scores` (a dict label -> mean AUROC). Logic unchanged, including skipping
    single-electrode streaks and the `>= n_electrodes - 1` stopping condition."""
    stem_nums = [stem_electrode_name(e) for e in electrode_labels]
    stems = [x[0] for x in stem_nums]
    probes = {stem: [num for (s, num) in stem_nums if s == stem] for stem in set(stems)}

    probe_streaks = {}
    for probe, nums in probes.items():
        streaks = find_streaks(nums)
        probe_streaks[probe] = sorted(streaks, key=len, reverse=True)
    all_streaks = []
    for probe, streaks in probe_streaks.items():
        for streak in streaks:
            all_streaks.append((probe, streak))

    streak_performances = []
    for probe, streak in all_streaks:
        performances = []
        for num in streak:
            if f"{probe}{num}" in scores:
                performances.append(scores[f"{probe}{num}"])
        streak_performances.append((probe, streak, np.nanmean(performances) if performances else np.nan))
    # The notebook sorts with Python's sort on floats; a NaN mean would make that order undefined,
    # so refuse rather than reproduce an accident.
    if any(np.isnan(p) for _, _, p in streak_performances):
        raise ValueError("a probe run has no scored electrode; the notebook's sort is undefined here")
    streak_performances.sort(key=lambda x: x[2], reverse=True)
    all_streaks = [x[:2] for x in streak_performances]

    selected_electrodes = []
    total_electrodes = 0
    for probe, streak in all_streaks:
        if len(streak) == 1:
            continue
        for num in streak:
            selected_electrodes.append(f"{probe}{num}")
            total_electrodes += 1
            if total_electrodes == n_electrodes:
                break
        if total_electrodes >= n_electrodes - 1:
            break
    return selected_electrodes
# --- end of port ------------------------------------------------------------------------------------


def overlap(a, b):
    """Jaccard overlap of two electrode subsets, and the size of each."""
    sa, sb = set(a), set(b)
    return dict(n_a=len(sa), n_b=len(sb), shared=len(sa & sb), jaccard=len(sa & sb) / max(1, len(sa | sb)))
