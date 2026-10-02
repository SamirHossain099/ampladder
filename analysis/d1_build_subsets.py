"""D1 step 2: build the circular and cross-fitted electrode subsets from the electrode scores.

For each Lite session S with partner session P (the other Lite session of the same subject):
  circular    = select(scores measured on S)   -- the evaluated session scores its own electrodes
  cross-fitted = select(scores measured on P)  -- S never influences its own electrode choice
using the verbatim port of Neuroprobe's rule (`d1_selection.select_electrodes_from_scores`).

Also reports each subset's overlap with the shipped Lite list, which shows how far the D1 stand-in
rule (1-s bin, 2 folds) is from the shipped one (15 bins, 5 folds, all Full sessions).

Writes results/d1/subsets_circular.json, subsets_crossfit.json ({"s{S}_t{T}": [labels]}) and
results/d1/subset_overlap.csv.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d1_selection import overlap, select_electrodes_from_scores  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTNER = {(1, 1): (1, 2), (1, 2): (1, 1), (2, 0): (2, 4), (2, 4): (2, 0), (3, 0): (3, 1), (3, 1): (3, 0),
           (4, 0): (4, 1), (4, 1): (4, 0), (7, 0): (7, 1), (7, 1): (7, 0), (10, 0): (10, 1), (10, 1): (10, 0)}


def main(np_root="D:/ieeg07/np", data_root="D:/ieeg07/braintreebank", d1=os.path.join(HERE, "results", "d1")):
    os.environ["ROOT_DIR_BRAINTREEBANK"] = data_root
    sys.path.insert(0, np_root)
    from neuroprobe.braintreebank_subject import BrainTreebankSubject
    from neuroprobe.config import NEUROPROBE_LITE_ELECTRODES

    def scores(s, t):
        return pd.read_csv(os.path.join(d1, f"electrode_mean_s{s}_t{t}.csv")).set_index("electrode").mean_auroc.to_dict()

    circ, cross, rows = {}, {}, []
    for (s, t), (ps, pt) in PARTNER.items():
        labels = list(BrainTreebankSubject(s, allow_corrupted=False, cache=False).electrode_labels)
        tag = f"s{s}_t{t}"
        circ[tag] = select_electrodes_from_scores(labels, scores(s, t))
        cross[tag] = select_electrodes_from_scores(labels, scores(ps, pt))
        shipped = [e for e in NEUROPROBE_LITE_ELECTRODES[f"btbank{s}"] if e in labels]
        rows.append(dict(session=tag, n_available=len(labels),
                         **{f"circ_vs_cross_{k}": v for k, v in overlap(circ[tag], cross[tag]).items()},
                         **{f"circ_vs_shipped_{k}": v for k, v in overlap(circ[tag], shipped).items()},
                         **{f"cross_vs_shipped_{k}": v for k, v in overlap(cross[tag], shipped).items()}))
    json.dump(circ, open(os.path.join(d1, "subsets_circular.json"), "w"), indent=1)
    json.dump(cross, open(os.path.join(d1, "subsets_crossfit.json"), "w"), indent=1)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(d1, "subset_overlap.csv"), index=False)
    print(df[["session", "n_available", "circ_vs_cross_jaccard", "circ_vs_shipped_jaccard",
              "cross_vs_shipped_jaccard"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
