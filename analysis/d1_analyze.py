"""D1 step 4: selection inflation per rung, and the one comparison D1 predicts.

inflation(rung) = overall within-session AUROC on the circular subset minus on the cross-fitted
subset, paired by session, hierarchical bootstrap (analyze.hier_boot). PREREGISTRATION.md D1 predicts
no size, and one direction: inflation is larger for the amplitude-bearing rungs (L1, L5) than for the
amplitude-free ones (L2, L4). Tested as the per-session difference
  mean(inflation L1, L5) - mean(inflation L2, L4)
with the same bootstrap.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import pandas as pd  # noqa: E402

from analyze import hier_boot, load_decoded, overall  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNGS = ("L1", "L2", "L4", "L5")


def main(d1=os.path.join(HERE, "results", "d1")):
    circ = load_decoded(os.path.join(d1, "decode_circular"))
    cross = load_decoded(os.path.join(d1, "decode_crossfit"))
    infl, res = {}, {"inflation": {}}
    for r in RUNGS:
        oc, ox = overall(circ, r, "within"), overall(cross, r, "within")
        common = oc.index.intersection(ox.index)
        infl[r] = oc[common] - ox[common]
        m, lo, hi = hier_boot(infl[r], f"D1_{r}")
        res["inflation"][r] = dict(mean=m, ci_lo=lo, ci_hi=hi, n_sessions=int(len(common)),
                                   circular_mean=float(oc[common].mean()), crossfit_mean=float(ox[common].mean()))
    common = infl["L1"].index
    for r in RUNGS:
        common = common.intersection(infl[r].index)
    contrast = (infl["L1"][common] + infl["L5"][common]) / 2 - (infl["L2"][common] + infl["L4"][common]) / 2
    m, lo, hi = hier_boot(contrast, "D1_amplitude_contrast")
    res["amplitude_bearing_minus_amplitude_free"] = dict(mean=m, ci_lo=lo, ci_hi=hi, predicted="positive")
    overl = pd.read_csv(os.path.join(d1, "subset_overlap.csv"))
    res["subset_overlap"] = {c: float(overl[c].mean()) for c in overl.columns if c.endswith("jaccard")}
    with open(os.path.join(d1, "d1_results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
