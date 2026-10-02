"""Descriptive, not registered: does H3's equivalence hold under every S1 estimator setting?

PREREGISTRATION.md section 8 registers S1 as "L2 recomputed over upper scale x order; all six
reported". It does not register H3 under each setting. Because H3 is the hypothesis most exposed to
the estimator (it compares the battery itself with BrainBERT), this script reports L2 minus
BrainBERT per-window for all six settings with the same paired hierarchical bootstrap and margin,
labelled descriptive. Writes results/s1/h3_by_setting.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import pandas as pd  # noqa: E402

from analyze import EQUIV, load_decoded, paired  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = [(128, 1), (256, 1), (512, 1), (128, 2), (256, 2), (512, 2)]     # (256, 1) is the primary


def main():
    cells = pd.read_csv(os.path.join(HERE, "results", "analysis", "cells.csv"))
    bb = cells[cells.model == "brainbert_perwindow"]
    out = {"registered": False, "equivalence_margin": EQUIV, "settings": {}}
    for smax, order in SETTINGS:
        if (smax, order) == (256, 1):
            l2 = cells[cells.model == "L2"]
        else:
            l2 = load_decoded(os.path.join(HERE, "results", "s1", f"decode_s{smax}_o{order}"))
        c = pd.concat([l2[l2.model == "L2"], bb], ignore_index=True)
        t = paired(c, "L2", "brainbert_perwindow", "within", f"S1_H3_{smax}_{order}")
        t["equivalent_within_margin"] = bool(-EQUIV < t["ci_lo"] and t["ci_hi"] < EQUIV)
        t["primary"] = (smax, order) == (256, 1)
        out["settings"][f"smax{smax}_order{order}"] = t
        print(f"s_max {smax} order {order}: {t['mean_diff']:+.4f} [{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}] "
              f"equivalent={t['equivalent_within_margin']}", flush=True)
    with open(os.path.join(HERE, "results", "s1", "h3_by_setting.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
