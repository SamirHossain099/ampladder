"""Test H1-H5 and report D1/D2, exactly as PREREGISTRATION.md sections 6-7 and 10 specify.

Written before any rung was decoded (git history shows it), so the statistics are fixed in the same
sense the hypotheses are.

Units
-----
* cell     test AUROC for one (split, session, task), mean over that cell's folds;
* overall  mean over the 15 tasks for one (split, session); a session lacking any task is excluded
           from that comparison, and the exclusion is reported;
* inference: hierarchical bootstrap -- resample subjects with replacement, then sessions within each
  drawn subject -- 10,000 resamples. Seeds are CRC32 of the comparison's name: a pure function,
  never Python's process-randomised hash() (which broke reproducibility in projects 02 and 08).

Every comparison is a paired difference, A minus B, on sessions both cover. No ratios.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import zlib  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_BOOT = 10_000
SPLIT_DIR = {"within": "Within-Session", "cross_session": "Cross-Session", "cross_subject": "Cross-Subject"}
LEADERBOARD = {
    "pub_linear_lap_stft": "Linear_Laplacian_rereferencing_spectrogram_Andrii_Zahorodnii_02_05_2026",
    "brainbert_perwindow": "BrainBERT_frozen_offtheshelf_perwindow_STFT_zscoring_Andrii_Zahorodnii_02_05_2026",
    "brainbert_untrained_perwindow": "BrainBERT_untrained_frozen_offtheshelf_perwindow_STFT_zscoring_Andrii_Zahorodnii_02_05_2026",
    "brainbert_globalz": "BrainBERT_frozen_global_z_Wang_et_al_2023_Geeling_Chau_02_05_2026",
    "popt_perwindow": "PopulationTransformer_offtheshelf_perwindow_STFT_zscoring_Andrii_Zahorodnii_02_05_2026",
    "popt_globalz": "PopulationTransformer_LaplacianSTFT_global_z_Chau_et_al_2025_Geeling_Chau_28_04_2026",
    "diver1_tiny_frozen": "DIVER-1(0.1s,tiny,frozen)_Yonghyeon_Gwon_09_01_2026",
    "diver1_tiny": "DIVER-1(0.1s,tiny)_Yonghyeon_Gwon_09_01_2026",
    "mapa": "MAPA_Ben_Tang_09_09_2026",
}
EQUIV = 0.02          # H3 equivalence margin, fixed in section 7


def seed_for(name):
    return zlib.crc32(name.encode("utf-8"))


def load_decoded(decode_dir):
    frames = [pd.read_csv(p) for p in glob.glob(os.path.join(decode_dir, "L*.csv"))]
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames)
    cell = (d.groupby(["rung", "split", "test_subject", "test_trial", "task"]).test_auroc.mean()
            .reset_index().rename(columns={"rung": "model", "test_auroc": "auroc"}))
    return cell


def load_leaderboard(lb_root):
    rows = []
    for name, folder in LEADERBOARD.items():
        for split, sub in SPLIT_DIR.items():
            for f in glob.glob(os.path.join(lb_root, folder, sub, "population_*.json")):
                task = os.path.basename(f)[len("population_"):-len(".json")]
                for key, v in json.load(open(f, encoding="utf-8"))["evaluation_results"].items():
                    s, t = key.replace("btbank", "").split("_")
                    folds = v["population"]["one_second_after_onset"]["folds"]
                    rows.append(dict(model=name, split=split, test_subject=int(s), test_trial=int(t),
                                     task=task, auroc=float(np.mean([x["test_roc_auc"] for x in folds])),
                                     n_folds=len(folds)))
    return pd.DataFrame(rows)


def overall(cells, model, split, n_tasks=15):
    c = cells[(cells.model == model) & (cells.split == split)]
    g = c.groupby(["test_subject", "test_trial"])
    out = g.auroc.mean()[g.task.nunique() == n_tasks]
    return out                       # index (subject, trial)


def hier_boot(values, name, n=N_BOOT):
    """values: Series indexed by (subject, trial). Returns (mean, lo, hi) of the session mean."""
    rng = np.random.default_rng(seed_for(name))
    by_sub = {}
    for (s, t), v in values.items():
        by_sub.setdefault(s, []).append(v)
    subs = sorted(by_sub)
    stats = np.empty(n)
    for i in range(n):
        draw = []
        for s in rng.choice(subs, size=len(subs), replace=True):
            vs = by_sub[s]
            draw.extend(rng.choice(vs, size=len(vs), replace=True))
        stats[i] = np.mean(draw)
    return float(np.mean(values.values)), float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def paired(cells, a, b, split, name):
    oa, ob = overall(cells, a, split), overall(cells, b, split)
    common = oa.index.intersection(ob.index)
    d = (oa[common] - ob[common])
    m, lo, hi = hier_boot(d, name)
    return dict(comparison=name, split=split, a=a, b=b, n_sessions=int(len(common)),
                sessions_dropped=sorted(set(map(str, oa.index.union(ob.index))) - set(map(str, common))),
                mean_diff=m, ci_lo=lo, ci_hi=hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", default=os.path.join(HERE, "results", "decode"))
    ap.add_argument("--leaderboard", default="D:/ieeg07/np/leaderboard")
    ap.add_argument("--tau", default=os.path.join(HERE, "results", "label_timescale", "tau_by_task.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "analysis"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cells = pd.concat([load_decoded(a.decode), load_leaderboard(a.leaderboard)], ignore_index=True)
    cells.to_csv(os.path.join(a.out, "cells.csv"), index=False)
    res = dict(n_boot=N_BOOT, equivalence_margin=EQUIV, tests={}, context={}, reproduction={})
    have = set(cells.model)

    # Reproduction of every published linear cell by this project's L5.
    if {"L5", "pub_linear_lap_stft"} <= have:
        m = cells[cells.model == "L5"].merge(cells[cells.model == "pub_linear_lap_stft"],
                                             on=["split", "test_subject", "test_trial", "task"],
                                             suffixes=("_mine", "_pub"))
        diff = (m.auroc_mine - m.auroc_pub).abs()
        res["reproduction"] = dict(cells=int(len(m)), max_abs_diff=float(diff.max()),
                                   median_abs_diff=float(diff.median()),
                                   within_0p01=int((diff <= 0.01).sum()))

    w = "within"
    if {"L5", "L4"} <= have:
        res["tests"]["H1_L5_minus_L4"] = paired(cells, "L5", "L4", w, "H1")
    if "L1" in have:
        m, lo, hi = hier_boot(overall(cells, "L1", w), "H2")
        res["tests"]["H2_L1_overall"] = dict(mean=m, ci_lo=lo, ci_hi=hi, predicted_above=0.55)
    if {"L2", "brainbert_perwindow"} <= have:
        t = paired(cells, "L2", "brainbert_perwindow", w, "H3_brainbert")
        t["equivalent_within_margin"] = bool(t["ci_lo"] > -EQUIV and t["ci_hi"] < EQUIV)
        res["tests"]["H3_L2_minus_brainbert_perwindow"] = t
    if {"L2", "popt_perwindow"} <= have:
        t = paired(cells, "L2", "popt_perwindow", w, "H3_popt_unpaired")
        t["note"] = "PopT per-window reports 1 fold on 173/180 cells; test sets are not paired (D2)"
        res["tests"]["H3s_L2_minus_popt_perwindow_UNPAIRED"] = t
    if {"L2", "L4"} <= have and os.path.exists(a.tau):
        tau = pd.read_csv(a.tau).set_index("task").tau_s_median
        per_task = (cells[(cells.model == "L2") & (cells.split == w)].set_index(["test_subject", "test_trial", "task"]).auroc
                    - cells[(cells.model == "L4") & (cells.split == w)].set_index(["test_subject", "test_trial", "task"]).auroc).dropna()
        tm = per_task.groupby(level="task").mean()
        rho = float(spearmanr(tau.reindex(tm.index), tm).correlation)
        rng = np.random.default_rng(seed_for("H4"))
        sess = per_task.reset_index()
        subs = sorted(sess.test_subject.unique())
        boots = []
        for _ in range(2000):
            parts = []
            for s in rng.choice(subs, size=len(subs), replace=True):
                trials = sorted(sess[sess.test_subject == s].test_trial.unique())
                for t in rng.choice(trials, size=len(trials), replace=True):
                    parts.append(sess[(sess.test_subject == s) & (sess.test_trial == t)])
            bt = pd.concat(parts).groupby("task").auroc.mean()
            boots.append(spearmanr(tau.reindex(bt.index), bt).correlation)
        res["tests"]["H4_tau_vs_L2_minus_L4"] = dict(rho=rho, ci_lo=float(np.nanquantile(boots, .025)),
                                                     ci_hi=float(np.nanquantile(boots, .975)),
                                                     per_task=tm.to_dict(), tau=tau.to_dict(),
                                                     note="9 of 15 tasks tie at tau = 1 s (FINDINGS F10)")
    if "L0" in have:
        c0 = cells[(cells.model == "L0") & (cells.split == w)]
        per = {}
        for task, g in c0.groupby("task"):
            m, lo, hi = hier_boot(g.set_index(["test_subject", "test_trial"]).auroc, f"H5_{task}")
            per[task] = dict(mean=m, ci_lo=lo, ci_hi=hi, confounded=bool(lo > 0.5 or hi < 0.5))
        res["tests"]["H5_L0_time_only"] = per

    for name, (x, y) in {"brainbert_globalz_minus_perwindow": ("brainbert_globalz", "brainbert_perwindow"),
                         "popt_globalz_minus_perwindow_UNPAIRED": ("popt_globalz", "popt_perwindow"),
                         "brainbert_pretrained_minus_untrained": ("brainbert_perwindow", "brainbert_untrained_perwindow"),
                         "pub_linear_minus_brainbert_perwindow": ("pub_linear_lap_stft", "brainbert_perwindow")}.items():
        if {x, y} <= have:
            res["context"][name] = paired(cells, x, y, w, name)

    with open(os.path.join(a.out, "tests.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
