"""Deviation D3 (POST HOC): Neuroprobe's fold construction versus common-time folds.

PREREGISTRATION.md, D3. Outcomes fixed before running:
  * clock: L0 per task under each scheme, as AUROC and as |AUROC - 0.5| (a reversing trend pushes
    AUROC below 0.5 and averaging cancels it), and the paired difference between schemes;
  * neural rungs: overall within-session AUROC, Neuroprobe folds minus common-time folds, paired by
    session, hierarchical bootstrap. No direction predicted.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402

import pandas as pd  # noqa: E402

from analyze import hier_boot, load_decoded, overall  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neuroprobe", default=os.path.join(HERE, "results", "decode"))
    ap.add_argument("--common", default=os.path.join(HERE, "results", "d3", "decode"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "d3", "d3_results.json"))
    a = ap.parse_args()
    npf = load_decoded(a.neuroprobe)
    npf = npf[npf.split == "within"]
    ctf = load_decoded(a.common)
    res = {"note": "POST HOC (D3)", "clock_L0_by_task": {}, "neural_overall": {}}

    if {"L0"} <= set(npf.model) & set(ctf.model):
        a0 = npf[npf.model == "L0"].set_index(["test_subject", "test_trial", "task"]).auroc
        b0 = ctf[ctf.model == "L0"].set_index(["test_subject", "test_trial", "task"]).auroc
        for task in sorted(set(a0.index.get_level_values("task"))):
            x, y = a0.xs(task, level="task"), b0.xs(task, level="task")
            common = x.index.intersection(y.index)
            x, y = x[common], y[common]
            row = {}
            for name, v in (("np_auroc", x), ("ct_auroc", y), ("np_abs_dev", (x - 0.5).abs()),
                            ("ct_abs_dev", (y - 0.5).abs()), ("abs_dev_np_minus_ct", (x - 0.5).abs() - (y - 0.5).abs())):
                m, lo, hi = hier_boot(v, f"D3_L0_{task}_{name}")
                row[name] = dict(mean=m, ci_lo=lo, ci_hi=hi)
            res["clock_L0_by_task"][task] = row
        ad = (a0 - 0.5).abs().groupby(level=["test_subject", "test_trial"]).mean()
        bd = (b0 - 0.5).abs().groupby(level=["test_subject", "test_trial"]).mean()
        m, lo, hi = hier_boot(ad - bd, "D3_L0_absdev_overall")
        res["clock_L0_overall_abs_dev_np_minus_ct"] = dict(mean=m, ci_lo=lo, ci_hi=hi,
                                                           np_mean=float(ad.mean()), ct_mean=float(bd.mean()))

    for rung in ("L1", "L2", "L3", "L4", "L5"):
        if rung in set(npf.model) and rung in set(ctf.model):
            x, y = overall(npf, rung, "within"), overall(ctf, rung, "within")
            common = x.index.intersection(y.index)
            m, lo, hi = hier_boot(x[common] - y[common], f"D3_{rung}")
            res["neural_overall"][rung] = dict(np_minus_ct=m, ci_lo=lo, ci_hi=hi, n_sessions=int(len(common)),
                                               np_mean=float(x[common].mean()), ct_mean=float(y[common].mean()))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "clock_L0_by_task"}, indent=2))
    rows = [(t, r["np_auroc"]["mean"], r["ct_auroc"]["mean"], r["np_abs_dev"]["mean"], r["ct_abs_dev"]["mean"],
             r["abs_dev_np_minus_ct"]["ci_lo"], r["abs_dev_np_minus_ct"]["ci_hi"]) for t, r in res["clock_L0_by_task"].items()]
    if rows:
        print(pd.DataFrame(rows, columns=["task", "np_auc", "ct_auc", "np_|d|", "ct_|d|", "diff_lo", "diff_hi"])
              .sort_values("np_|d|").round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
